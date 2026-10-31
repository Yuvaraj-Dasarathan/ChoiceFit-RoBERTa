import React, { useEffect, useRef, useState } from "react";

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
    name: "Antibiotics & colds",
    tag: "Health reasoning",
    message: "Antibiotics don't work on colds, and doctors usually advise rest and fluids instead.",
    question: "What is the main reason antibiotics don't work on colds?",
    options: [
      "Colds are caused by viruses, not bacteria",
      "Colds are too mild to treat",
      "Antibiotics expire quickly in the body",
      "Colds are caused by cold weather",
    ],
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

const REPO = "https://github.com/Yuvaraj-Dasarathan/ChoiceFit-RoBERTa";
const RESULTS = "https://github.com/Yuvaraj-Dasarathan/ChoiceFit-RoBERTa/tree/main/results";
const HF = "https://huggingface.co/Yuvaraj-ml/choicefit-roberta";

const TOPICS = [
  ["Reasoning & logic (does one statement follow from another?)", 34],
  ["Science, math & medical questions", 11],
  ["Commonsense & reading comprehension", 8],
  ["Safety, hate speech & prompt-injection checks", 8],
  ["Sentiment, emotion & opinion", 7],
  ["Other mixed tasks", 32],
];
const OPTION_MIX = [["2", 38], ["3", 26], ["4", 23], ["5–8", 12]];

const SAMPLE = `{
  "message":  "A $70 gift is expensive. A $30 gift is cheap.",
  "question": "Does the first statement support, contradict, or say nothing about: 'A $26 gift is cheap.'?",
  "options": [
    "entailment",     <- correct (target 1.0)
    "neutral",        <-          (target 0.0)
    "contradiction"   <-          (target 0.0)
  ]
}`;

const browserInfo = () => ({
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  language: navigator.language,
  screen: `${window.screen.width}x${window.screen.height}`,
  page: location.href,
});

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
  const [edited, setEdited] = useState(false);
  const resultRef = useRef(null);
  const [seconds, setSeconds] = useState(null);

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
    setEdited(false);
  };

  const touch = (fn) => (...args) => {
    setEdited(true);
    fn(...args);
  };

  const resetAll = () => {
    setActive(-1);
    setMessage("");
    setQuestion("");
    setOptions(["", ""]);
    setResult(null);
    setError("");
    setEdited(false);
  };

  const setOption = touch((i, value) => setOptions(options.map((o, j) => (j === i ? value : o))));

  const run = async () => {
    setError("");
    setResult(null);
    const cleaned = options.map((o) => o.trim()).filter(Boolean);
    if (!message.trim()) return setError("Enter a message.");
    if (cleaned.length < 2) return setError("Enter at least 2 options.");
    if (new Set(cleaned).size !== cleaned.length) return setError("Options must be different from each other.");
    setLoading(true);
    setSeconds(null);
    // On phones the result card is below the form, so bring it into view right away
    if (window.innerWidth < 720) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    const started = performance.now();
    try {
      const res = await fetch(API + "/predict", {
        method: "POST",
        headers: HEADERS,
        body: JSON.stringify({ message, question, options: cleaned, client: browserInfo() }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "The server is temporarily down. Please try again after some time.";
        throw new Error(detail);
      }
      setSeconds((performance.now() - started) / 1000);
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
    <>
      {!online && (
        <div className="offline">The server is temporarily down. Please try again after some time.</div>
      )}

      <section className="hero">
        <div className="wrap hero-row">
          <div>
            <h1>ChoiceFit <span>Playground</span></h1>
            <p>Fine-tuned RoBERTa that picks the best answer for a message from 2–8 options.</p>
          </div>
          <div className="links">
            <a className="gh" href={REPO} target="_blank" rel="noreferrer">
              <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.72 1.22 1.88.87 2.34.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
              GitHub
            </a>
            <a className="gh" href={HF} target="_blank" rel="noreferrer">
              <span aria-hidden="true">🤗</span> Model
            </a>
          </div>
        </div>
      </section>

      <div className="wrap">
        <p className="goal">🎯 Goal: Learn how close a small fine-tuned model can get to frontier classification models.</p>
      </div>

      <main className="wrap">
        <div className="steps">
          <div><i>1</i>Pick an example</div>
          <div><i>2</i>Edit anything</div>
          <div><i>3</i>Score options</div>
        </div>

        <h2 className="section-title">Try an example</h2>
        <div className="chips">
          {EXAMPLES.map((ex, i) => (
            <button key={ex.name} className={"chip" + (active === i ? " active" : "")} onClick={() => loadExample(i)}>
              <small>{ex.tag}</small>
              {ex.name}
            </button>
          ))}
        </div>

        <div className="grid">
          <section className="card">
            <h2>Your input</h2>

            <label>Message</label>
            <textarea rows={4} maxLength={2000} value={message} onChange={touch((e) => setMessage(e.target.value))} />

            <label>Question</label>
            <input value={question} maxLength={500} onChange={touch((e) => setQuestion(e.target.value))} />

            <label>Options <span className="count">{options.length}/{MAX_OPTIONS}</span></label>
            {options.map((o, i) => (
              <div className="option-row" key={i}>
                <span className="num">{i + 1}</span>
                <input value={o} maxLength={300} placeholder={"Option " + (i + 1)} onChange={(e) => setOption(i, e.target.value)} />
                {options.length > 2 && (
                  <button className="icon" aria-label="Remove option" onClick={touch(() => setOptions(options.filter((_, j) => j !== i)))}>×</button>
                )}
              </div>
            ))}
            {options.length < MAX_OPTIONS && (
              <button className="link" onClick={touch(() => setOptions([...options, ""]))}>+ Add option</button>
            )}

            <button className="primary" onClick={run} disabled={loading}>
              {loading ? <span className="spinner" /> : null}
              {loading ? "Sending to RoBERTa…" : "Score options"}
            </button>
            <button className="reset" onClick={resetAll} disabled={!edited || loading}>
              Reset
            </button>
            {error && <div className="error">{error}</div>}
          </section>

          <section className="card result" ref={resultRef}>
            <h2>Result</h2>
            {!result && !loading && (
              <div className="placeholder">
                <div className="emoji">🎯</div>
                <p>Scores appear here after you tap <b>Score options</b>.</p>
                <p className="muted">The first request after the server starts can take a few seconds.</p>
              </div>
            )}
            {loading && (
              <div className="pipeline">
                <div className="pipe-title">Running your input through the trained model</div>
                <div className="pipe-flow">
                  <div className="node">Your message<br />+ options</div>
                  <div className="arrow"><i /><i /><i /></div>
                  <div className="node model">ChoiceFit<br />RoBERTa</div>
                  <div className="arrow"><i /><i /><i /></div>
                  <div className="node">Scores</div>
                </div>
                <p className="muted">Each option is paired with your message, and the fine-tuned model scores how well it fits.</p>
              </div>
            )}
            {result && !loading && (
              <>
                <div className="best">
                  <span>Best match</span>
                  <strong>{result.best_option}</strong>
                  <em>{(result.results[result.best_index].probability * 100).toFixed(1)}% confident</em>
                </div>
                <div className="scored-by">⚡ Scored by the fine-tuned ChoiceFit RoBERTa model{seconds != null ? " in " + seconds.toFixed(1) + "s" : ""}</div>
                {result.results.map((r, i) => (
                  <div className={"bar-row" + (i === result.best_index ? " top" : "")} key={i}>
                    <div className="bar-label">
                      <span>{r.option}</span>
                      <b>{(r.probability * 100).toFixed(1)}%</b>
                    </div>
                    <div className="bar"><div className="fill" style={{ width: r.probability * 100 + "%" }} /></div>
                  </div>
                ))}
              </>
            )}
          </section>
        </div>
      </main>

      <section className="train" id="training">
        <div className="wrap">
          <h2>How I trained this RoBERTa model</h2>
          <p className="lead">
            I fine-tuned <b>RoBERTa-base</b> on about <b>half a million</b> (486k) multiple-choice examples, so it learns
            to choose among options it has never seen. It is a general decision model, not tied to one industry.
          </p>

          <div className="tiles">
            <div><b>~½ million</b><span>training examples</span></div>
            <div><b>2–8</b><span>options per example</span></div>
            <div><b>~2.5</b><span>passes over the data (early-stopped)</span></div>
            <div><b>50+ hrs</b><span>total training effort, incl. laptop testing</span></div>
          </div>

          <div className="two">
            <div className="panel">
              <h3>What the training data looked like</h3>
              <p className="muted">One example = a message, a question, and 2–8 options with the correct one marked (simplified from a real training row):</p>
              <pre>{SAMPLE}</pre>
              <p className="muted">
                The model reads the message and question with each option, gives every option a score, and is trained
                to push the correct option's score up.
              </p>
            </div>

            <div className="panel">
              <h3>What topics were covered (approx.)</h3>
              <p className="muted">
                Mostly general reasoning and question answering across many topics. It was not built for any one
                industry, such as banking or insurance, but it can still handle everyday intents like the examples above.
              </p>
              {TOPICS.map(([name, pct]) => (
                <div className="mini" key={name}>
                  <div className="mini-label"><span>{name}</span><b>{pct}%</b></div>
                  <div className="bar"><div className="fill" style={{ width: pct * 2.4 + "%" }} /></div>
                </div>
              ))}
              <h3 className="gap">How many options per example</h3>
              <div className="pills">
                {OPTION_MIX.map(([n, pct]) => (<span key={n}><b>{n}</b> options · {pct}%</span>))}
              </div>
            </div>
          </div>

          <div className="panel">
            <h3>Where and how it was trained</h3>
            <ul>
              <li><b>Hardware:</b> AWS SageMaker <code>ml.g6.2xlarge</code> with one <b>NVIDIA L4</b> GPU (24 GB).</li>
              <li><b>Setup:</b> bf16 precision, 256-token limit, learning rate 2e-5, about 64 messages per update, stopped early once validation loss stopped improving.</li>
              <li><b>Time:</b> 50+ hours overall (approx.), counting cloud training plus testing and evaluation on my laptop.</li>
              <li><b>Result:</b> 66.7% accuracy on 6,597 unseen test examples. It can be overconfident.</li>
            </ul>
            <a className="cta" href={REPO} target="_blank" rel="noreferrer">
              To learn more about how this BERT-based model was trained, check out the GitHub repo →
            </a>
            <a className="cta alt" href={HF} target="_blank" rel="noreferrer">
              🤗 Download the trained model on Hugging Face →
            </a>
          </div>
        <div className="cmp">
          <h3>ChoiceFit vs TypeSafe JEV</h3>
          <p className="muted">
            This training and evaluation approach is inspired by TypeSafe.ai's JEV classification. I ran the same
            labeled examples through my model and the JEV API. JEV scores higher on every set.
          </p>
          <div className="tablewrap">
            <table>
              <thead>
                <tr><th>Test set (examples)</th><th>ChoiceFit</th><th>JEV</th></tr>
              </thead>
              <tbody>
                <tr><td>Mixed held-out test (6,597)</td><td><b>66.7%</b></td><td>78.3%</td></tr>
                <tr><td>Banking intents, 4 options (500)</td><td><b>67.0%</b></td><td>96.2%</td></tr>
                <tr><td>Everyday intents, 4 options (500)</td><td><b>82.8%</b></td><td>99.4%</td></tr>
              </tbody>
            </table>
          </div>
          <p className="verified">
            ✅ Not random or AI-made percentages: these were measured on held-out test data the model never trained on, with the same examples run through both systems.{" "}
            <a href={RESULTS} target="_blank" rel="noreferrer">See the raw reports on GitHub →</a>
          </p>
          <p className="muted small">Accuracy = how often the top-scored option was the correct one. JEV is a separate commercial API and was only used as a reference.</p>
        </div>
      </div>
      </section>

      <footer>
        <div className="wrap">
          A learning project for research and educational use only, not for commercial use. Trained on{" "}
          <a href="https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions" target="_blank" rel="noreferrer">
            tasksource-jev-typed-decisions
          </a>
          . The model can be overconfident and shouldn't be used for real decisions. Inputs and basic browser details (IP, browser, timezone) are logged for testing. Please don't enter sensitive information.
        </div>
      </footer>
    </>
  );
}
