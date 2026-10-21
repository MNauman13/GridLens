"use client";

import { useState } from "react";
import Link from "next/link";

type Fixture = {
  fixture_kind: string; real_demand_data: boolean; trained_model: boolean;
  contract_version: string; issued_at: string; context_start: string;
  context_end_exclusive: string; target_start: string; target_end_exclusive: string;
  input_count: number; gap_intervals: number; output_count: number;
};

const steps = [
  { title: "Read the context", text: "Use seven complete UTC days of historical demand. Every selected version must also meet the evidence-mode availability rule.", region: 0 },
  { title: "Respect the issue-day gap", text: "Issue at 09:30 UTC. Observed demand from the issue day is excluded; the complete day is not yet known.", region: 1 },
  { title: "Predict the target day", text: "Produce 48 half-hour predictions for the following UTC day. Target observations remain hidden from model inputs.", region: 2 },
  { title: "Wait for the actuals", text: "Update calibration only when actuals become available. Revised-data replay simulates release at 09:00 UTC after the target day.", region: 2 },
];

export default function Home() {
  const [issueDate, setIssueDate] = useState("2024-02-10");
  const [result, setResult] = useState<Fixture | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  async function check() {
    setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch(`/api/fixture?issue_date=${encodeURIComponent(issueDate)}`);
      if (!response.ok) throw new Error(response.status === 503 ? "The verification API is unavailable. Start the Python API and try again." : "The issue date could not be verified.");
      setResult(await response.json() as Fixture);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Verification failed.");
    } finally { setLoading(false); }
  }
  return <main>
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><span className="phase">Phase 1 · Foundations</span></header>
    <section className="intro"><p className="eyebrow">ELECTRICITY FORECASTING, MADE AUDITABLE</p>
      <h1>A forecast starts with<br/>what we know.</h1>
      <p className="lede">Before training a Transformer, verify its time boundaries. Inspect the history it can use, the day it must skip, and the day it will predict.</p>
      <p className="notice">Development fixture · Synthetic inputs · No trained forecast</p>
    </section>
    <section className="panel" aria-labelledby="contract-heading">
      <div className="section-heading"><div><p className="eyebrow">01 / CHECK THE CONTRACT</p><h2 id="contract-heading">One issue. Exact boundaries.</h2></div><span className="unit">NESO ND · MW · UTC</span></div>
      <form onSubmit={event => { event.preventDefault(); void check(); }}>
        <label htmlFor="issue-date">Issue date<input id="issue-date" type="date" value={issueDate} onChange={event => {setIssueDate(event.target.value); setResult(null); setError("");}} required/></label>
        <button type="submit" disabled={loading}>{loading ? "Checking…" : "Run contract check"}</button>
      </form>
      {error && <p role="alert" className="error">{error}</p>}
      <div role="status" aria-live="polite">{result ? <>
        <p className="verified">Contract verified · {result.contract_version}</p>
        <dl className="counts"><div><dt>History inputs</dt><dd>{result.input_count}</dd></div><div><dt>Skipped intervals</dt><dd>{result.gap_intervals}</dd></div><div><dt>Target outputs</dt><dd>{result.output_count}</dd></div></dl>
        <dl className="boundaries"><div><dt>Issued at</dt><dd>{result.issued_at}</dd></div><div><dt>Context ends, exclusive</dt><dd>{result.context_end_exclusive}</dd></div><div><dt>Target starts</dt><dd>{result.target_start}</dd></div><div><dt>Target ends, exclusive</dt><dd>{result.target_end_exclusive}</dd></div></dl>
      </> : !loading && <p className="hint">Run the check to validate a synthetic input window through the Python API.</p>}</div>
    </section>
    <section className="panel" aria-labelledby="timeline-heading">
      <p className="eyebrow">02 / FOLLOW THE INFORMATION</p><h2 id="timeline-heading">What can the model see?</h2>
      <div className="timeline" aria-label="Forecast regions, shown schematically rather than to time scale">{["7-day context", "Issue-day gap", "Target day"].map((name, index) => <div key={name} className={index === steps[step].region ? "region active" : "region"}><span>{name}</span><strong>{["336 inputs", "48 excluded", "48 outputs"][index]}</strong></div>)}</div>
      <div className="step-detail" aria-live="polite"><h3>{steps[step].title}</h3><p>{steps[step].text}</p></div>
      <div className="step-controls"><button onClick={() => setStep(value => value - 1)} disabled={step === 0}>Previous</button><span aria-label="Timeline step">{step + 1} / {steps.length}</span><button onClick={() => setStep(value => value + 1)} disabled={step === steps.length - 1}>Next</button></div>
    </section>
    <footer><p>Supported by National Energy SO Open Data</p><p>Independent project · Source quality audit and model training are later phases.</p></footer>
  </main>;
}
