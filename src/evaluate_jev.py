"""Run TypeSafe JEV against a labeled ChoiceFit JSONL file."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model import iter_data


API_URL = "https://api.typesafe.ai/v1/systemone"


def payload_for(row, model):
    """Translate one ChoiceFit row to TypeSafe's documented choice request."""
    return {
        "model": model,
        "state": row["message"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": row.get("question"),
                "criteria": row["choices"],
            }
        },
    }


def request_jev(payload, api_key, timeout, retries):
    body = json.dumps(payload).encode("utf-8")
    request = Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise RuntimeError(f"JEV API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"Could not reach JEV API: {exc.reason}") from exc
        time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def existing_ids(path):
    if not path.exists():
        return set()
    result = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: prediction has no usable id")
            if identifier in result:
                raise ValueError(f"{path}:{line_number}: duplicate id {identifier!r}")
            result.add(identifier)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labeled ChoiceFit JSONL input")
    parser.add_argument("--predictions", required=True, help="New JSONL result file")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--api-key-env", default="TYPESAFE_API_KEY")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent API requests (watch your org rate limit; 429s are retried)")
    parser.add_argument("--limit", type=int, default=0, help="0 means every input row")
    parser.add_argument("--resume", action="store_true", help="Append only rows absent from predictions")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0 or args.limit < 0 or args.workers < 1:
        parser.error("timeout must be positive; retries and limit must be zero or positive; workers at least 1")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        parser.error(f"Set {args.api_key_env} before calling the live JEV API")

    destination = Path(args.predictions)
    done = existing_ids(destination) if args.resume else set()
    if destination.exists() and not args.resume:
        parser.error("Predictions already exist; choose a new path or pass --resume")
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    submitted = skipped = input_tokens = output_tokens = 0
    mode = "a" if args.resume else "x"

    def pending_rows():
        nonlocal skipped
        queued = 0
        for row in iter_data(args.data):
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("JEV evaluation requires every input row to have a nonempty unique id")
            if identifier in seen:
                raise ValueError(f"Input has duplicate id {identifier!r}")
            seen.add(identifier)
            if identifier in done:
                skipped += 1
                continue
            if args.limit and queued >= args.limit:
                return
            queued += 1
            yield row

    def evaluate_row(row):
        response = request_jev(payload_for(row, args.model), api_key, args.timeout, args.retries)
        answer = response["answers"]["decision"]
        choice = answer["choice"]
        if choice not in row["choices"]:
            raise ValueError(f"JEV returned unknown choice {choice!r} for id {row['id']!r}")
        usage = response.get("usage", {})
        return {
            "id": row["id"], "source": row.get("source"), "message": row["message"],
            "question": row.get("question"), "choices": row["choices"],
            "target": row.get("target", row.get("label")), "choice": choice,
            "description": row["choices"][choice],
            "probabilities": answer["probabilities"], "confidence": answer.get("confidence"),
            "api_model": response.get("model"), "usage": usage,
        }

    # Rows are sent in windows of concurrent requests and written in input order.
    window = args.workers * 4
    rows = pending_rows()
    try:
        with destination.open(mode) as sink, ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                batch = list(islice(rows, window))
                if not batch:
                    break
                for record in pool.map(evaluate_row, batch):
                    usage = record["usage"]
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    submitted += 1
                    if submitted % 25 == 0:
                        print(json.dumps({"submitted": submitted, "skipped": skipped,
                                          "input_tokens": input_tokens}), flush=True)
    except BaseException:
        # A failed first call must not leave an empty file that blocks the next attempt.
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
        raise
    print(json.dumps({"predictions": str(destination), "submitted": submitted,
                      "skipped": skipped, "input_tokens": input_tokens,
                      "output_tokens": output_tokens}), flush=True)


if __name__ == "__main__":
    main()
