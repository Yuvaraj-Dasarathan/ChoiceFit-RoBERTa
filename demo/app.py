"""Gradio demo for the ChoiceFit RoBERTa model.

Local:  .venv/bin/python demo/app.py
Space:  uploaded next to model.py; loads the model from the Hugging Face Hub.
"""
import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root locally (src.model)
sys.path.insert(0, str(HERE))         # model.py next to app.py on the Space
try:
    from src.model import DEFAULT_QUESTION, ChoiceModel
except ImportError:
    from model import DEFAULT_QUESTION, ChoiceModel

LOCAL_MODEL = HERE.parent / "artifacts/sagemaker/choicefit-l4-full-b8-001"
MODEL_ID = os.getenv("CHOICEFIT_MODEL", str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "")
MAX_OPTIONS = 8          # training used 2-8 options
MAX_CHARS = 2000
MAX_OPTION_CHARS = 300

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_ID:
            raise gr.Error("Set CHOICEFIT_MODEL to a model folder or Hugging Face repo id.")
        _model = ChoiceModel(MODEL_ID, device="cpu")
    return _model


def predict(message, question, options_text):
    message = (message or "").strip()
    question = (question or "").strip() or DEFAULT_QUESTION
    options = [line.strip() for line in (options_text or "").splitlines() if line.strip()]
    if not message:
        raise gr.Error("Enter a message.")
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise gr.Error(f"Enter 2 to {MAX_OPTIONS} options, one per line (you gave {len(options)}).")
    if len(message) > MAX_CHARS or any(len(o) > MAX_OPTION_CHARS for o in options):
        raise gr.Error(f"Message must be under {MAX_CHARS} characters and each option under {MAX_OPTION_CHARS}.")
    if len(set(options)) != len(options):
        raise gr.Error("Options must be different from each other.")
    choices = {f"option_{i}": text for i, text in enumerate(options)}
    result = get_model().predict(message, choices, question, pair_batch_size=8)
    return {choices[key]: probability for key, probability in result["probabilities"].items()}


EXAMPLES = [
    ["I was charged twice for the same coffee this morning.",
     "Which category best describes what the customer wants?",
     "Report an incorrect or duplicate charge\nReset a forgotten password\nOpen a new bank account"],
    ["The movie was a total waste of two hours, but the soundtrack was lovely.",
     "What is the overall sentiment?",
     "The message is mostly positive\nThe message is mostly negative\n"
     "The message is neither positive nor negative\nThe message contains both positive and negative views"],
    ["My top-up was reverted, why did this happen?",
     "Which category best describes what the customer wants?",
     "Top up was reverted\nCard payment fee charged\nChange PIN\nTransfer fee charged"],
]

DESCRIPTION = """
**ChoiceFit** is a RoBERTa-base model (125M parameters) fine-tuned on about 486k multiple-choice decisions.
Give it a message, a question, and 2-8 options (one per line). It scores each option and shows probabilities.

*Learning project, for research and educational use only, not for commercial use. Do not enter sensitive information.*
"""

with gr.Blocks(title="ChoiceFit RoBERTa demo") as demo:
    gr.Markdown("# ChoiceFit: pick the best option")
    gr.Markdown(DESCRIPTION)
    message = gr.Textbox(label="Message", lines=3)
    question = gr.Textbox(label="Question (optional)", value=DEFAULT_QUESTION)
    options = gr.Textbox(label=f"Options (2-{MAX_OPTIONS}, one per line)", lines=5)
    run = gr.Button("Score options", variant="primary")
    output = gr.Label(label="Probability of each option", num_top_classes=MAX_OPTIONS)
    run.click(predict, [message, question, options], output)
    gr.Examples(EXAMPLES, [message, question, options])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
