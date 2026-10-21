"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import demo from "../../../data/settlement-demo.json";

type Day = keyof typeof demo;
const explanations: Record<Day, string> = {
  ordinary: "A normal settlement day has 48 half-hours. Leap day changes the number of days in the year, not the number of periods in this day.",
  spring: "The British clock jumps from 00:59 GMT to 02:00 BST. This local day contains 46 periods. UTC still advances in uninterrupted half-hours.",
  autumn: "The British clock repeats the 01:00 hour. This day has 50 periods. The UTC offset distinguishes the two occurrences; UTC intervals remain unique.",
};

export default function SettlementLesson() {
  const [day, setDay] = useState<Day>("spring");
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const selected = demo[day];
  const current = selected.periods[position];
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setPosition(value => (value + 1) % selected.periods.length), 650);
    return () => clearInterval(timer);
  }, [playing, selected.periods.length]);
  return <main>
    <header><Link className="brand" href="/">Grid<span>Lens</span></Link><span className="phase">Phase 2 / Canonical time</span></header>
    <section className="intro"><p className="eyebrow">FOLLOW A HALF-HOUR</p><h1>One clock changes.<br/>UTC keeps moving.</h1><p className="lede">Explore the exact conversion used by the ingestion pipeline. These are teaching timestamps, with no demand values or model predictions.</p></section>
    <section className="panel" aria-labelledby="lesson-title">
      <h2 id="lesson-title">British settlement time to UTC</h2>
      <label htmlFor="settlement-day">Day type</label>
      <select id="settlement-day" value={day} onChange={event => {setPlaying(false); setPosition(0); setDay(event.target.value as Day);}}>
        <option value="ordinary">Ordinary / leap day: 48 periods</option>
        <option value="spring">Spring: 46 periods</option>
        <option value="autumn">Autumn: 50 periods</option>
      </select>
      <p>{explanations[day]}</p>
      <p className="verified" data-testid="period-count">{selected.date} / {selected.periods.length} settlement periods</p>
      <div className="settlement-slots" aria-label="Settlement periods">{selected.periods.map((period, index) => <button key={period.period} aria-label={`Period ${period.period}`} aria-pressed={position === index} onClick={() => {setPlaying(false); setPosition(index);}}>{period.period}</button>)}</div>
      <div className="step-controls"><button onClick={() => setPlaying(value => !value)}>{playing ? "Pause animation" : "Play animation"}</button><label htmlFor="period-slider">Settlement period {current.period}</label></div>
      <input className="period-slider" id="period-slider" type="range" min="1" max={selected.periods.length} value={position + 1} onChange={event => {setPlaying(false); setPosition(Number(event.target.value) - 1);}}/>
      <dl className="settlement-reading" aria-live={playing ? "off" : "polite"}>
        <div><dt>Local start, including UTC offset</dt><dd data-testid="local-start">{current.local_start}</dd></div>
        <div><dt>UTC start</dt><dd data-testid="utc-start">{current.utc_start}</dd></div>
        <div><dt>UTC end, exclusive</dt><dd>{current.utc_end}</dd></div>
      </dl>
      <p className="notice">The offset matters: 01:00+01:00 and 01:00+00:00 are different instants.</p>
    </section>
    <section className="panel"><h2>Why the model still predicts 48 values</h2><p>NESO settlement days use the British local clock. GridLens converts their periods to UTC first, then builds forecasts for complete UTC days. A UTC target day always has 48 half-hour intervals, including on British clock-change dates.</p><p>The original date and period stay attached to each observation so the conversion can be inspected.</p><p><a href="https://www.elexon.co.uk/bsc/settlement/">Elexon settlement conversion table</a></p></section>
    <footer><Link href="/">Back to the forecast contract</Link><p>Supported by National Energy SO Open Data</p></footer>
  </main>;
}
