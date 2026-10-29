import React, { useEffect, useState } from "react";

import { REMOTE_API_URL } from "./config.js";

const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API = isLocal ? "" : REMOTE_API_URL.replace(/\/$/, "");
const HEADERS = { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" };

const DEFAULT_QUESTION = "Which category best describes what the customer wants?";
const MAX_OPTIONS = 8;

const EXAMPLES = [
  {
    name: "Duplicate charge",
    tag: "Banking",
    message: "I was charged twice for the same coffee this morning.",
    question: DEFAULT_QUESTION,
    options: ["Report an incorrect or duplicate charge", "Reset a forgotten password", "Open a new bank account"],
  },
  {
    name: "Mixed sentiment",
    tag: "Sentiment",
    message: "The movie was a total waste of two hours, but the soundtrack was lovely.",
    question: "What is the overall sentiment?",
    options: [
      "The message is mostly positive",
      "The message is mostly negative",
      "The message is neither positive nor negative",
      "The message contains both positive and negative views",
    ],
  },
  {
    name: "Reverted top-up",
    tag: "Banking",
    message: "My top-up was reverted, why did this happen?",
    question: DEFAULT_QUESTION,
    options: ["Top up was reverted", "Card payment fee charged", "Change PIN", "Transfer fee charged"],
  },
  {
    name: "Capital of France",
    tag: "General knowledge",
    message: "Capital of france is paris",
    question: "What is the capital of France?",
    options: ["New Delhi", "France", "London", "Paris"],
  },
  {
    name: "Software bug",
    tag: "Support",
    message: "The app crashes when I open settings.",
    question: "What kind of problem is the customer reporting?",
    options: ["A software bug", "A billing problem", "A request for a new feature", "A compliment"],
  },
  {
    name: "Reschedule",
    tag: "Email intent",
    message: "Could we move tomorrow's 3pm call to Thursday morning instead?",
    question: "What does the sender want?",
    options: ["Reschedule a meeting", "Cancel the subscription", "Request a refund", "Ask for a product demo"],
  },
  {
    name: "Lost card",
    tag: "Banking",
    message: "I think someone stole my card, I can't find it anywhere",
    question: DEFAULT_QUESTION,
    options: ["Report a lost or stolen card", "Change PIN", "Check exchange rate", "Top up account"],
  },
  {
    name: "Safe command",
    tag: "Safety",
    message: "Command: rm -rf ./node_modules && npm install",
    question: "How risky is this shell command?",
    options: ["Safe, routine cleanup", "Risky, could delete important data", "Unrelated to files"],
  },
  {
    name: "Dangerous command",
    tag: "Safety",
    message: "Command: rm -rf / --no-preserve-root",
    question: "How risky is this shell command?",
    options: ["Safe, routine cleanup", "Risky, could delete important data", "Unrelated to files"],
  },
];

export default function App() {
  const first = EXAMPLES[0];
  const [message, setMessage] = useState(first.message);
  const [question, setQuestion] = useState(first.question);
  const [options, setOptions] = useState(first.options);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState(true);
  const [active, setActive] = useState(0);

  // One health check when the page opens; after that, a failed submit shows the error.
  useEffect(() => {
    fetch(API + "/health", { headers: HEADERS })
      .then((r) => setOnline(r.ok))
      .catch(() => setOnline(false));
  }, []);

  const loadExample = (i) => {
    const ex = EXAMPLES[i];
    setActive(i);
    setMessage(ex.message);
    setQuestion(ex.question);
    setOptions(ex.options);
    setResult(null);
    setError("");
  };

  const setOption = (i, value) => setOptions(options.map((o, j) => (j === i ? value : o)));

  const run = async () => {
    setError("");
    setResult(null);
    const cleaned = options.map((o) => o.trim()).filter(Boolean);
    if (!message.trim()) return setError("Enter a message.");
    if (cleaned.length < 2) return setError("Enter at least 2 options.");
    if (new Set(cleaned).size !== cleaned.length) return setError("Options must be different from each other.");
    setLoading(true);
    try {
      const res = await fetch(API + "/predict", {
        method: "POST",
        headers: HEADERS,
        body: JSON.stringify({ message, question, options: cleaned }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "The server is temporarily down. Please try again after some time.";
        throw new Error(detail);
      }
      setResult(data);
      setOnline(true);
    } catch (e) {
      if (e instanceof TypeError) setOnline(false);
      setError(e instanceof TypeError ? "The server is temporarily down. Please try again after some time." : e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      {!online && (
        <div className="offline">
          The server is temporarily down. Please try again after some time.
        </div>
      )}

      <header>
        <h1>ChoiceFit Playground</h1>
        <p>
          A RoBERTa-base model (125M parameters) fine-tuned on about 486k multiple-choice decisions. Give it a message
          and 2 to {MAX_OPTIONS} options; it scores each option.
        </p>
      </header>

      <div className="examples">
        <div className="examples-title">Try an example (click to load, then edit it)</div>
        <div className="chips">
          {EXAMPLES.map((ex, i) => (
            <button key={ex.name} className={"chip" + (active === i ? " active" : "")} onClick={() => loadExample(i)}>
              <small>{ex.tag}</small>
              {ex.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid">
        <section className="card">
          <div className="row">
            <h2>Input</h2>
            <span className="muted">Edit anything, then score</span>
          </div>

          <label>Message</label>
          <textarea rows={4} maxLength={2000} value={message} onChange={(e) => setMessage(e.target.value)} />

          <label>Question</label>
          <input value={question} maxLength={500} onChange={(e) => setQuestion(e.target.value)} />

          <label>Options</label>
          {options.map((o, i) => (
            <div className="option-row" key={i}>
              <input value={o} maxLength={300} placeholder={"Option " + (i + 1)} onChange={(e) => setOption(i, e.target.value)} />
              {options.length > 2 && (
                <button className="icon" title="Remove" onClick={() => setOptions(options.filter((_, j) => j !== i))}>×</button>
              )}
            </div>
          ))}
          {options.length < MAX_OPTIONS && (
            <button className="link" onClick={() => setOptions([...options, ""])}>+ Add option</button>
          )}

          <button className="primary" onClick={run} disabled={loading}>
            {loading ? "Scoring..." : "Score options"}
          </button>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="card">
          <h2>Result</h2>
          {!result && !loading && <p className="muted">Scores appear here. The first request after the server starts may take a few seconds.</p>}
          {loading && <p className="muted">Running the model...</p>}
          {result && (
            <>
              <div className="best">
                <span className="muted">Best option</span>
                <strong>{result.best_option}</strong>
              </div>
              {result.results.map((r, i) => (
                <div className={"bar-row" + (i === result.best_index ? " top" : "")} key={i}>
                  <div className="bar-label">
                    <span>{r.option}</span>
                    <span>{(r.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="bar"><div className="fill" style={{ width: r.probability * 100 + "%" }} /></div>
                </div>
              ))}
            </>
          )}
        </section>
      </div>

      <footer>
        Learning project, for research and educational use only. Trained on{" "}
        <a href="https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions" target="_blank" rel="noreferrer">
          tasksource-jev-typed-decisions
        </a>
        . Model is overconfident at times and not for real decisions. Do not enter sensitive information.
      </footer>
    </div>
  );
}
