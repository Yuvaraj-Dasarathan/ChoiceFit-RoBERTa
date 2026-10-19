"""Stream JSONL data for single-device training of the shared choice scorer."""
import argparse
import json
import os
from itertools import islice
from pathlib import Path
from time import monotonic

import torch
from transformers import set_seed

from .data import audit_splits, batches, buffered_shuffle
from .model import ChoiceModel, iter_data
from .checkpoint import file_digest, load_recovery, save_recovery, rng_state, restore_rng


def targets(rows, device):
    probabilities = [torch.tensor(
        [r["target"][key] if "target" in r else float(key == r["label"])
         for key in r["choices"]], dtype=torch.float32, device=device) for r in rows]
    return torch.nn.utils.rnn.pad_sequence(probabilities, batch_first=True, padding_value=0.0)


def choice_loss(logits, probabilities):
    # Padded logits are -inf and padded targets are zero. Mask BEFORE the
    # product to avoid 0 * -inf producing NaN for variable choice counts.
    log_probabilities = logits.log_softmax(-1).masked_fill(probabilities == 0, 0.0)
    return -(probabilities * log_probabilities).sum(-1)


def classification_head_stats(model):
    """Small, useful summaries of the trainable scalar scorer.

    Logging every RoBERTa parameter would mean serialising roughly 125 million
    values at every update.  These norms show whether the newly-created head is
    changing without turning a learning trace into an enormous opaque file.
    """
    head = getattr(model.model, "classifier", None)
    if head is None:
        return {}
    values = {}
    for name, parameter in head.named_parameters():
        value = parameter.detach().float().norm().cpu().item()
        values[f"classifier_{name.replace('.', '_')}_l2"] = value
    return values


def trace_examples(sink, epoch, next_update, batch, logits, probabilities, losses):
    """Write one compact, pre-update learning observation for each message."""
    predicted_probabilities = logits.detach().softmax(-1).cpu().tolist()
    scores = logits.detach().cpu().tolist()
    target_probabilities = probabilities.detach().cpu().tolist()
    per_example_losses = losses.detach().cpu().tolist()
    for row, scores_for_row, predicted_for_row, target_for_row, loss in zip(
            batch, scores, predicted_probabilities, target_probabilities, per_example_losses):
        keys = list(row["choices"])
        size = len(keys)
        scores_for_row = scores_for_row[:size]
        predicted_for_row = predicted_for_row[:size]
        target_for_row = target_for_row[:size]
        predicted_index = max(range(size), key=predicted_for_row.__getitem__)
        sink({"event": "example", "epoch": epoch, "next_update": next_update,
              "choice_ids": keys, "target": dict(zip(keys, target_for_row)),
              "scores": dict(zip(keys, scores_for_row)),
              "probabilities": dict(zip(keys, predicted_for_row)),
              "predicted_choice": keys[predicted_index], "loss": loss})


@torch.inference_mode()
def evaluate(model, rows, batch_size, pair_batch_size=16, prediction_sink=None):
    model.model.eval()
    correct = hard_examples = examples = modal_correct = 0
    total_loss = target_mass = brier = 0.0
    for batch in batches(rows, batch_size):
        logits = model.logits(batch, pair_batch_size=pair_batch_size)
        labels = targets(batch, model.device)
        losses = choice_loss(logits, labels)
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite evaluation loss")
        total_loss += losses.sum().item()
        predicted = logits.argmax(-1)
        mass = labels.gather(1, predicted[:, None]).squeeze(1)
        hard = (labels == 1).sum(-1) == 1
        correct += ((mass == 1) & hard).sum().item()
        hard_examples += hard.sum().item()
        modal_correct += (mass >= labels.max(-1).values - 1e-6).sum().item()
        target_mass += mass.sum().item()
        brier += ((logits.softmax(-1) - labels) ** 2).sum().item()
        if prediction_sink is not None:
            predictions = logits.softmax(-1).cpu().tolist()
            for row, probabilities in zip(batch, predictions):
                keys = list(row["choices"])
                probabilities = probabilities[:len(keys)]
                best = max(range(len(keys)), key=probabilities.__getitem__)
                prediction_sink.write(json.dumps({
                    "id": row.get("id"), "source": row.get("source"),
                    "message": row["message"], "question": row.get("question"),
                    "choices": row["choices"], "target": row.get("target", row.get("label")),
                    "choice": keys[best], "description": row["choices"][keys[best]],
                    "probabilities": dict(zip(keys, probabilities))}) + "\n")
        examples += len(batch)
    if not examples:
        raise ValueError("Evaluation dataset is empty")
    return {"accuracy": correct / hard_examples if hard_examples else None,
            "hard_examples": hard_examples, "soft_examples": examples - hard_examples,
            "top_target_accuracy": modal_correct / examples,
            "mean_selected_target_probability": target_mass / examples,
            "brier_score": brier / examples, "loss": total_loss / examples, "examples": examples}


def train_epoch(model, optimizer, rows, batch_size, max_pairs, accumulation_steps, log_every=100,
                progress=None, checkpoint_every=0, checkpoint_callback=None, epoch=None,
                trace_sink=None, update_callback=None):
    model.model.train()
    iterator = iter(batches(rows, batch_size, max_pairs))
    progress = progress or {}
    total_loss = progress.get("total_loss", 0.0)
    examples = progress.get("examples", 0)
    updates = progress.get("updates", 0)
    skipped = 0
    started = monotonic()
    while group := list(islice(iterator, accumulation_steps)):
        if skipped < progress.get("updates", 0):
            skipped += 1
            continue
        # Weight by messages, not microbatches: pair budgets and the last batch
        # can produce unequal microbatch sizes.
        group_examples = sum(len(batch) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for batch in group:
            logits = model.logits(batch)
            probabilities = targets(batch, model.device)
            losses = choice_loss(logits, probabilities)
            loss_sum = losses.sum()
            if not torch.isfinite(loss_sum):
                raise ValueError("Non-finite training loss; stopping before optimizer update")
            if trace_sink is not None:
                trace_examples(trace_sink, epoch, updates + 1, batch, logits, probabilities, losses)
            (loss_sum / group_examples).backward()
            total_loss += loss_sum.detach().item()
            group_loss += loss_sum.detach().item()
            examples += len(batch)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.model.parameters(), 1.0, error_if_nonfinite=True).detach().cpu().item()
        optimizer.step()
        updates += 1
        if checkpoint_callback and checkpoint_every and updates % checkpoint_every == 0:
            checkpoint_callback({"updates": updates, "examples": examples, "total_loss": total_loss})
        if log_every and updates % log_every == 0:
            print(json.dumps({"updates": updates, "messages": examples,
                              "train_loss": total_loss / examples,
                              "elapsed_seconds": round(monotonic() - started, 1)}), flush=True)
        if update_callback is not None:
            keep_training = update_callback({
                "event": "update", "epoch": epoch, "update": updates,
                "messages_seen": examples, "update_loss": group_loss / group_examples,
                "mean_train_loss": total_loss / examples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": gradient_norm,
                **classification_head_stats(model)})
            if keep_training is False:
                return {"train_loss": total_loss / examples, "examples": examples, "updates": updates,
                        "train_seconds": round(monotonic() - started, 2), "stopped_early": True}
    if not examples:
        raise ValueError("Training dataset is empty")
    return {"train_loss": total_loss / examples, "examples": examples, "updates": updates,
            "train_seconds": round(monotonic() - started, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(os.getenv("SM_CHANNEL_TRAIN", "data/typed-local"), "train.jsonl"))
    parser.add_argument("--validation", default=os.path.join(os.getenv("SM_CHANNEL_VALIDATION", "data/typed-local"), "validation.jsonl"))
    parser.add_argument("--output", default=os.getenv("SM_MODEL_DIR", "artifacts/typed-local-v1"))
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum messages per microbatch")
    parser.add_argument("--max-pairs", type=int, default=32, help="Maximum candidate pairs per training microbatch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade extra computation for lower activation memory")
    parser.add_argument("--shuffle-buffer", type=int, default=10000, help="Maximum buffered training records")
    parser.add_argument("--eval-pair-batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100, help="Optimizer updates between progress reports")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", help="Private directory for automatic full-state recovery (SageMaker: /opt/ml/checkpoints)")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Optimizer updates between recovery checkpoints")
    parser.add_argument("--convergence-jsonl", help="Write per-example and per-update learning observations here")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run validation every N optimizer updates (0 means only at epoch end)")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after N validation checks without a lower loss (0 disables it)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum validation-loss decrease that counts as an improvement")
    parser.add_argument("--early-stopping-min-validations", type=int, default=0,
                        help="Do not early-stop until this many post-training validation checks complete")
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.max_pairs, args.gradient_accumulation_steps,
                args.shuffle_buffer, args.eval_pair_batch_size, args.log_every, args.learning_rate, args.checkpoint_every)
    if min(positive) <= 0:
        parser.error("Epochs, batch/buffer limits, accumulation steps, log interval, and learning rate must be positive")
    if args.eval_every < 0:
        parser.error("eval-every must be zero or positive")
    if (args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0 or
            args.early_stopping_min_validations < 0):
        parser.error("Early-stopping values must be zero or positive")
    if not 16 <= args.max_length <= 512:
        parser.error("max-length must be between 16 and 512")
    output = Path(args.output)
    recovery = load_recovery(args.checkpoint_dir)
    if output.exists() and any(output.iterdir()) and recovery is None:
        parser.error("Output directory is not empty; choose a new --output to preserve existing checkpoints")
    set_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    counts = audit_splits(args.train, args.validation, args.max_pairs)
    signature = {key: getattr(args, key) for key in (
        "epochs", "batch_size", "max_pairs", "gradient_accumulation_steps", "gradient_checkpointing",
        "shuffle_buffer", "learning_rate", "max_length", "seed", "early_stopping_patience",
        "early_stopping_min_delta", "early_stopping_min_validations")}
    signature.update(train_sha256=file_digest(args.train), validation_sha256=file_digest(args.validation))
    if recovery and recovery["signature"] != signature:
        raise ValueError("Recovery data or training settings changed; use a new checkpoint directory")
    print(json.dumps({"dataset_counts": counts}), flush=True)
    model = ChoiceModel(args.model, args.device, args.max_length)
    print(json.dumps({"device": str(model.device), "max_length": model.max_length}), flush=True)
    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    output.mkdir(parents=True, exist_ok=True)
    trace_file = None
    if args.convergence_jsonl:
        trace_path = Path(args.convergence_jsonl)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("a" if recovery else "x")

        def write_trace(record):
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()
        if not recovery:
            write_trace({"event": "run", "arguments": vars(args), "dataset_counts": counts,
                         "meaning": "example events are observed before their optimizer update"})
    else:
        write_trace = None
    history = []
    best = float("inf")
    best_weights = None
    validations_without_improvement = 0
    validation_checks = 0
    start_epoch, progress = 0, None
    if recovery:
        model.model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        history, best, best_weights = recovery["history"], recovery["best"], recovery["best_weights"]
        validations_without_improvement = recovery.get("validations_without_improvement", 0)
        validation_checks = recovery.get("validation_checks", 0)
        start_epoch, progress = recovery["epoch"], recovery["progress"]
        restore_rng(recovery["rng"])
        print(json.dumps({"resumed_epoch": start_epoch, "completed_updates": progress.get("updates", 0)}), flush=True)
        del recovery

    def checkpoint(epoch, progress):
        if args.checkpoint_dir:
            save_recovery(args.checkpoint_dir, {
                "version": 1, "signature": signature, "epoch": epoch, "progress": progress,
                "model": model.model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng_state(), "history": history, "best": best, "best_weights": best_weights,
                "validations_without_improvement": validations_without_improvement,
                "validation_checks": validation_checks,
            })

    def assess_validation(epoch, validation):
        """Keep the best trained weights and decide whether to stop this run."""
        nonlocal best, best_weights, validations_without_improvement, validation_checks
        if not epoch:
            return False, False
        validation_checks += 1
        improved = validation["loss"] < best - args.early_stopping_min_delta
        if improved:
            best = validation["loss"]
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in model.model.state_dict().items()}
            # Save immediately: the best value can occur between epoch ends.
            model.save(output)
            validations_without_improvement = 0
        else:
            validations_without_improvement += 1
        should_stop = (args.early_stopping_patience > 0 and
                       validation_checks >= args.early_stopping_min_validations and
                       validations_without_improvement >= args.early_stopping_patience)
        return improved, should_stop

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = {"epoch": epoch}
        started = monotonic()
        stopped_early = False
        if epoch:
            rows = buffered_shuffle(iter_data(args.train), args.shuffle_buffer, args.seed + epoch)
            def on_update(record):
                if write_trace is not None:
                    write_trace(record)
                if args.eval_every and record["update"] % args.eval_every == 0:
                    validation = evaluate(model, iter_data(args.validation), args.batch_size,
                                          args.eval_pair_batch_size)
                    model.model.train()
                    improved, should_stop = assess_validation(epoch, validation)
                    event = {"event": "validation", "epoch": epoch,
                             "update": record["update"], "validation": validation,
                             "is_best": improved,
                             "validation_check": validation_checks,
                             "validations_without_improvement": validations_without_improvement}
                    print(json.dumps({
                        "event": "validation", "epoch": epoch, "update": record["update"],
                        "rows_seen": record["messages_seen"],
                        "validation_loss": validation["loss"],
                        "accuracy": validation["accuracy"], "is_best": improved,
                        "validation_check": validation_checks,
                        "validations_without_improvement": validations_without_improvement}), flush=True)
                    if write_trace is not None:
                        write_trace(event)
                    if should_stop:
                        print(json.dumps({"event": "early_stop", "epoch": epoch,
                                          "update": record["update"],
                                          "patience": args.early_stopping_patience}), flush=True)
                        return False
            metrics.update(train_epoch(model, optimizer, rows, args.batch_size, args.max_pairs,
                                       args.gradient_accumulation_steps, args.log_every,
                                       progress=progress, checkpoint_every=args.checkpoint_every,
                                       checkpoint_callback=lambda p: checkpoint(epoch, p), epoch=epoch,
                                       trace_sink=write_trace, update_callback=on_update))
            progress = None
            stopped_early = metrics.get("stopped_early", False)
        print(json.dumps({"epoch": epoch, "phase": "validation"}), flush=True)
        metrics["validation"] = evaluate(model, iter_data(args.validation), args.batch_size, args.eval_pair_batch_size)
        metrics["epoch_seconds"] = round(monotonic() - started, 2)
        # If the last update already ran validation, avoid counting that exact
        # result twice for early stopping. Epoch-end validation remains in the
        # history so each epoch has one simple summary row.
        duplicate_interval_validation = (epoch and args.eval_every and
                                         metrics.get("updates", 0) % args.eval_every == 0)
        improved = False
        if epoch and not duplicate_interval_validation:
            improved, stopped_after_epoch = assess_validation(epoch, metrics["validation"])
            stopped_early = stopped_early or stopped_after_epoch
        metrics["is_best"] = improved
        metrics["validations_without_improvement"] = validations_without_improvement
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if write_trace is not None:
            write_trace({"event": "validation", "epoch": epoch,
                         "update": metrics.get("updates", 0), "validation": metrics["validation"],
                         "end_of_epoch": True, "is_best": improved,
                         "validations_without_improvement": validations_without_improvement})
        if not stopped_early:
            checkpoint(epoch + 1, {})
        # Preserve metrics after every epoch, including the initial baseline.
        temporary = output / "training_metrics.json.tmp"
        temporary.write_text(json.dumps({"arguments": vars(args), "history": history}, indent=2))
        temporary.replace(output / "training_metrics.json")
        if stopped_early:
            break
    # Restore the best trained model even when a Spot restart began in a new
    # container, where the old /opt/ml/model directory is no longer present.
    if best_weights is not None:
        model.model.load_state_dict(best_weights)
        model.save(output)
        (output / "training_metrics.json").write_text(json.dumps({"arguments": vars(args), "history": history}, indent=2))
    if trace_file is not None:
        trace_file.close()


if __name__ == "__main__":
    main()
