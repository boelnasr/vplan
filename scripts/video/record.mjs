#!/usr/bin/env node
// Scripted, repeatable takes of the vplan demo video (W4.3). One browser context, so one .webm, per chapter.
//
// Every take starts from a reset state (a fresh page load) and logs its events (chapter, caption, box, click, cut)
// with t in seconds from the start of that take's video. The takes are recorded clean: nothing is drawn into
// the page. burn.mjs adds captions, highlight boxes and click rings afterwards from the logs.
//
// Takes wait on UI state (an element visible, a text present), never on fixed sleeps alone; the fixed holds are
// reading time after the state is reached. A take that cannot find what its caption names throws instead of logging
// a caption that does not match the screen: re-record it with --chapter.
//
//   node record.mjs [--url URL] [--chapter NAME[,NAME...]] [--cosmos] [--out DIR] [--headed] [--list]
//
// Without --chapter it records every chapter except the optional cosmos chapter, which needs the UI started with
// --cosmos-url while the others need it without; --cosmos records that chapter alone.
//
// The UI must already run (see README.md). The replay chapters need it started with --replay on the c3-test run and
// --seed-dir on its seed set.

import { chromium } from 'playwright';
import { existsSync, readFileSync } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const VIEWPORT = { width: 1920, height: 1080 };
const DEFAULT_URL = 'http://172.18.80.1:7861/?__theme=light';
const WINDOWS_FONTS = '/mnt/c/Windows/Fonts';

// ---- the chapters, in video order -------------------------------------------------------------------------------

const CHAPTERS = [
  { name: 'title', run: titleTake },
  { name: 'architecture', run: architectureTake },
  { name: 'live', run: liveTake },
  { name: 'repair', run: repairTake },
  { name: 'refusal', run: refusalTake },
  // Optional: needs the UI started with --cosmos-url (record it alone with --cosmos); every other take needs it off
  { name: 'cosmos', run: cosmosTake, optional: true },
  { name: 'results', run: resultsTake },
  { name: 'closing', run: closingTake },
];

// ---- cards (local HTML, system fonts, no network) -----------------------------------------------------------------

const FONT = `"Segoe UI", system-ui, -apple-system, ui-sans-serif, Roboto, "Helvetica Neue", Arial, sans-serif`;
const CARD_CSS = `
:root{--bg:#f8fafc;--ink:#0f172a;--sub:#475569;--muted:#64748b;--blue:#2563eb;--line:#e2e8f0;--card:#fff;
--green:#1a7f37;--green-bg:#dafbe1;--amber:#9a6700}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font-family:${FONT};display:flex;align-items:center;justify-content:center}
.wrap{width:1440px}
.kicker{font-size:28px;font-weight:600;color:var(--blue);letter-spacing:.02em;margin:0 0 18px}
h1{font-size:96px;line-height:1.05;margin:0;font-weight:700;letter-spacing:-.015em}
h1 .brand{color:var(--blue)}
h2{font-size:72px;line-height:1.1;margin:0;font-weight:700;letter-spacing:-.01em}
p.lead{font-size:38px;line-height:1.45;color:var(--sub);margin:30px 0 0;max-width:1300px}
ol.flow{display:flex;gap:18px;align-items:center;list-style:none;padding:0;margin:48px 0 0;font-size:34px;font-weight:600}
ol.flow li{display:inline-flex;align-items:center;gap:12px}
ol.flow li+li::before{content:"\\2192";color:var(--muted);font-weight:400;margin-right:6px}
.n{display:inline-flex;align-items:center;justify-content:center;width:1.5em;height:1.5em;border-radius:50%;
background:var(--blue);color:#fff;font-size:.72em;font-weight:700}
.foot{margin-top:64px;font-size:26px;color:var(--muted);border-top:1px solid var(--line);padding-top:22px}
.tiles{display:grid;gap:22px;margin-top:34px}
.t3{grid-template-columns:repeat(3,1fr)}
.tile{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 30px;
box-shadow:0 1px 2px rgba(15,23,42,.06)}
.big{font-size:64px;font-weight:700;line-height:1.05;letter-spacing:-.01em}
.big.blue{color:var(--blue)}.big.green{color:var(--green)}
.lbl{font-size:26px;line-height:1.4;color:var(--sub);margin-top:10px}
.ci{font-size:22px;color:var(--muted);margin-top:8px}
.hero{display:flex;align-items:center;gap:40px}
.hero .tile{flex:1}
.arrow{font-size:72px;color:var(--muted)}
.reveal{opacity:0;animation:in .6s ease forwards}
body:has(.diagram){align-items:flex-start;padding-top:48px}
.wrap:has(.diagram){width:1300px}
.diagram{margin-top:22px;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#fff}
.diagram svg{display:block}
.rows{display:grid;gap:22px;margin-top:30px}
.row{display:grid;grid-template-columns:330px 1fr 1fr 1fr;gap:22px;align-items:stretch}
.row .who{display:flex;flex-direction:column;justify-content:center}
.who .m{font-size:34px;font-weight:700;line-height:1.15}
.who .s{font-size:22px;color:var(--muted);margin-top:8px;line-height:1.35}
.row .big{font-size:52px}
.big .blue{color:var(--blue)}
.row .lbl{font-size:23px}
body::after{content:"";position:fixed;right:0;bottom:0;width:4px;height:4px;background:#f8fafc;animation:hb .25s steps(2) infinite}
@keyframes hb{50%{background:#f7f9fb}}
@keyframes in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
`;

// The corner heartbeat (body::after, one shade of grey, invisible in the video) keeps the browser sending frames
// while a card is still: Playwright's recorder starts the video at its first frame, so a card that is never
// repainted would be missing from the start of the take, and every later event would be early by its length.
function cardPage(body) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>vplan</title><style>${CARD_CSS}</style>` +
    `</head><body><div class="wrap">${body}</div></body></html>`;
}

const FLOW = `<ol class="flow"><li><span class="n">1</span>Plan</li><li><span class="n">2</span>Verify</li>` +
  `<li><span class="n">3</span>Execute</li><li><span class="n">4</span>Judge</li></ol>`;

// The architecture card shows the SVG of a diagram in docs/diagrams, cropped by viewBox (the system overview without
// its legend strip: the captions say what each part does).
function diagramSvg(file, viewBox) {
  const html = readFileSync(path.join(REPO, 'docs', 'diagrams', file), 'utf8');
  const svg = html.slice(html.indexOf('<svg'), html.indexOf('</svg>') + '</svg>'.length);
  return svg.replace(/viewBox="[^"]*"/, `viewBox="${viewBox}" width="100%"`);
}

const CARDS = {
  title: cardPage(`
    <h1><span class="brand">vplan</span> LLMs plan, physics verifies</h1>
    <p class="lead">NVIDIA Nemotron proposes a pick-and-place plan for a robot arm. Kinematic and collision checks verify
    it before anything moves, and an independent judge scores the result in physics simulation.</p>
    ${FLOW}
    <div class="foot">Nebius × NVIDIA Global AI Hackathon · Physical AI track · simulation only</div>`),
  architecture: cardPage(`
    <div class="kicker">How it works</div><h2 style="font-size:52px">The model proposes, physics decides</h2>
    <div class="diagram">${diagramSvg('system-overview.html', '30 30 900 470')}</div>`),
  live: cardPage(`
    <div class="kicker">Part 1 of 3</div><h2>A live run in the browser</h2>
    <p class="lead">Plan, verify, execute and judge one task. The planner here is the rule-based baseline, so no API
    key is needed.</p>`),
  repair: cardPage(`
    <div class="kicker">Part 2 of 3</div><h2>Nemotron repairs its own plan</h2>
    <p class="lead">A recorded NVIDIA Nemotron attempt from the held-out evaluation, replayed without keys. The checks
    reject two plans; the third passes.</p>`),
  refusal: cardPage(`
    <div class="kicker">Part 3 of 3</div><h2>Knowing when to refuse</h2>
    <p class="lead">Also recorded: a task the arm cannot do, from the same held-out evaluation.</p>`),
  cosmos: cardPage(`
    <div class="kicker">Optional</div><h2>A learned visual check</h2>
    <p class="lead">NVIDIA Cosmos3-Edge, served locally, looks at rendered key frames after the checks pass. It is
    advisory, never verification.</p>`),
  // Lightning (D70) is the headline; Ultra (D82) ran on code changed after the freeze, and the richer findings (D85)
  // were merged after the evaluation, so both are shown beside it
  results: cardPage(`
    <h2 class="reveal" style="font-size:60px">Results on 300 held-out tasks</h2>
    <div class="lead reveal" style="font-size:30px;margin-top:14px;animation-delay:.2s">Share of solvable tasks done
    correctly. NVIDIA Nemotron on Nebius Token Factory.</div>
    <div class="rows">
      <div class="row reveal" style="grid-template-columns:300px 1.45fr 1fr 1fr;animation-delay:.8s">
        <div class="who"><div class="m">Nemotron 3.5 Lightning</div><div class="s">the headline: prompt and code
          frozen before the held-out set</div></div>
        <div class="tile"><div class="big">24.7% → <span class="blue">65.0%</span></div>
          <div class="lbl">first plan verified → up to 3 repairs from the findings</div>
          <div class="ci">95% CI 20.1–29.8% → 59.4–70.2%</div></div>
        <div class="tile"><div class="big blue">+29.3 pts</div><div class="lbl">over retrying as often without
          feedback</div><div class="ci">95% CI 23–36</div></div>
        <div class="tile"><div class="big green">0 of 215</div><div class="lbl">verified runs unsafe. Without the
          checks: 169 of 346.</div></div>
      </div>
      <div class="row reveal" style="grid-template-columns:300px 1.45fr 1fr 1fr;animation-delay:5.4s">
        <div class="who"><div class="m">Nemotron 3 Ultra</div><div class="s">run later, on code changed after the
          freeze</div></div>
        <div class="tile"><div class="big">75.7% → <span class="blue">92.7%</span></div>
          <div class="lbl">first plan verified → up to 3 repairs from the findings</div>
          <div class="ci">95% CI 70.5–80.2% → 89.1–95.1%</div></div>
        <div class="tile"><div class="big">−2.0 pts</div><div class="lbl">over retrying: the stronger model gains
          nothing from feedback</div><div class="ci">95% CI −5.7 to +1.7</div></div>
        <div class="tile"><div class="big green">0 of 298</div><div class="lbl">verified runs unsafe. Without the
          checks: 42 of 304.</div></div>
      </div>
      <div class="row reveal" style="grid-template-columns:300px 1.45fr 1fr 1fr;animation-delay:10s">
        <div class="who"><div class="m">Richer findings</div><div class="s">merged after the evaluation: each
          collision says what hit what, and how far</div></div>
        <div class="tile"><div class="big blue">76.0%</div>
          <div class="lbl">Lightning with repair, up from 65.0% on the same tasks</div>
          <div class="ci">paired +11.0 pts, 95% CI +4.7 to +17.3</div></div>
        <div class="tile"><div class="big blue">96.7%</div><div class="lbl">Ultra with repair, was 92.7%</div>
          <div class="ci">+4.0 pts, mostly from first plans, not repairs</div></div>
        <div class="tile"><div class="big green">0 of 559</div><div class="lbl">verified runs unsafe, both
          models</div></div>
      </div>
    </div>
    <div class="foot reveal" style="margin-top:30px;animation-delay:13.6s">Held-out set: 300 solvable and 100
    impossible generated tasks. Later runs are reported beside the headline, not instead of it.</div>`),
  closing: cardPage(`
    <h1 style="font-size:84px"><span class="brand">vplan</span> LLMs plan, physics verifies</h1>
    <p class="lead" style="font-size:35px;max-width:1440px">Planner: NVIDIA Nemotron 3.5 Lightning and 3 Ultra, served by Nebius Token Factory.<br>Checks: ManipulaPy kinematics and
    PyBullet collision checks, before anything moves.<br>Judge: independent scoring in PyBullet physics.</p>
    <div class="foot">Open source under AGPL-3.0 · Every recorded attempt replays without keys</div>`),
};

// ---- the execution GIF ------------------------------------------------------------------------------------------
// The UI shows the execution as a looping GIF. Chromium starts a GIF at its first paint in the viewport and keeps its
// clock running while it is scrolled away, so its phase is (now - first seen) mod loop. This init script only reads
// the page: it notes when each execution image first intersects the viewport.
const GIF_WATCH = `(() => {
  const seen = new WeakMap();
  window.__vpGifSeen = null;
  setInterval(() => {
    for (const img of document.querySelectorAll('.vp-exec img')) {
      if (seen.has(img) || !img.offsetParent || !img.complete) continue;
      const r = img.getBoundingClientRect();
      if (r.width > 0 && r.bottom > 0 && r.top < innerHeight) {
        const t = Date.now();
        seen.set(img, t);
        window.__vpGifSeen = t;
      }
    }
  }, 40);
})();`;

// Loop length of a GIF in seconds: the sum of its frame delays (Graphic Control Extensions, in 1/100 s).
function gifLoopSeconds(buf) {
  let i = 13;
  if (buf[10] & 0x80) i += 3 * (1 << ((buf[10] & 7) + 1));
  let delay = 0;
  const skipSubBlocks = () => {
    while (buf[i] !== 0) i += buf[i] + 1;
    i += 1;
  };
  while (i < buf.length) {
    const b = buf[i++];
    if (b === 0x3b) break;
    if (b === 0x21) {
      if (buf[i++] === 0xf9) delay += buf[i + 2] | (buf[i + 3] << 8);
      skipSubBlocks();
    } else if (b === 0x2c) {
      const packed = buf[i + 8];
      i += 9;
      if (packed & 0x80) i += 3 * (1 << ((packed & 7) + 1));
      i += 1;
      skipSubBlocks();
    } else throw new Error('the execution image is not a well-formed GIF');
  }
  return delay / 100;
}

// ---- the take: an event log against the video clock ---------------------------------------------------------------

class TakeError extends Error {}

class Take {
  constructor(page, name, url) {
    this.page = page;
    this.name = name;
    this.url = url;
    this.ref = 0; // epoch ms of the video's t = 0 (set in recordChapter)
    this.events = [];
    this.start = null;
    this.caption_ = null; // the open caption event
    this.boxes = []; // the open box events
  }

  now() {
    return (Date.now() - this.ref) / 1000;
  }

  log(type, fields = {}) {
    const event = { t: round(this.now()), type, ...fields };
    this.events.push(event);
    return event;
  }

  // The chapter starts here: burn.mjs drops everything before it (the blank first frames).
  async chapter(text) {
    await this.painted();
    this.start = round(this.now());
    this.log('chapter', { text });
  }

  // Two animation frames after the last change, so what the log says is on screen is painted.
  async painted() {
    await this.page.evaluate(() => document.fonts.ready.then(() => new Promise((ok) =>
      requestAnimationFrame(() => requestAnimationFrame(ok)))));
  }

  async card(name, holdS) {
    await this.page.setContent(CARDS[name], { waitUntil: 'load' });
    await this.chapter(name);
    await this.hold(holdS);
  }

  // A caption stays until the next caption or clear(); it closes the boxes of the previous one.
  async caption(text, { pos = 'bottom' } = {}) {
    this.closeCaption();
    this.unbox();
    await this.painted();
    this.caption_ = this.log('caption', { text, pos });
  }

  closeCaption() {
    if (this.caption_) this.caption_.dur = round(this.now() - this.caption_.t);
    this.caption_ = null;
  }

  // A highlight box around an element (or around several: their union), measured now; the page must not scroll
  // while it is open.
  async box(locator, label = '') {
    const rects = await Promise.all([].concat(locator).map(rectOf));
    const x0 = Math.min(...rects.map((r) => r.x));
    const y0 = Math.min(...rects.map((r) => r.y));
    const x1 = Math.max(...rects.map((r) => r.x + r.width));
    const y1 = Math.max(...rects.map((r) => r.y + r.height));
    const rect = { x: round(x0), y: round(y0), width: round(x1 - x0), height: round(y1 - y0) };
    this.boxes.push(this.log('box', { text: label || this.caption_?.text || '', rect }));
  }

  unbox() {
    for (const b of this.boxes) b.dur = round(this.now() - b.t);
    this.boxes = [];
  }

  clear() {
    this.closeCaption();
    this.unbox();
  }

  // Hover, log, then click 0.3 s later: the ring starts while the clicked element is still on screen (a dropdown
  // option or a thumbnail disappears on the click itself).
  async click(locator, label = '') {
    await locator.scrollIntoViewIfNeeded();
    await locator.hover();
    const rect = await rectOf(locator);
    this.log('click', { text: label, rect });
    await this.page.waitForTimeout(300);
    await locator.click();
    // the recording shows no pointer; parked in the empty margin it leaves no hover style behind
    await this.page.mouse.move(VIEWPORT.width - 24, VIEWPORT.height / 2);
  }

  // Reading time after the state is reached (pacing, not a wait for the UI).
  async hold(seconds) {
    await this.page.waitForTimeout(seconds * 1000);
  }

  // Wait for a UI state. A wait longer than lead + tail is cut in burn.mjs, keeping its first `lead` and last `tail`
  // seconds, so the video shows that the page worked without showing the whole wait.
  async until(what, fn, { lead = 0.6, tail = 0.4 } = {}) {
    const t1 = this.now();
    try {
      await fn();
    } catch (err) {
      throw new TakeError(`${this.name}: timed out waiting for ${what}: ${err.message.split('\n')[0]}`);
    }
    const t2 = this.now();
    if (t2 - t1 > lead + tail + 0.3) this.log('cut', { t: round(t1 + lead), dur: round(t2 - t1 - lead - tail), text: what });
  }

  // Cut everything from t1 to now (a pre-roll between a chapter card and the loaded page).
  cutSince(t1, what, tail = 0.1) {
    const t2 = this.now() - tail;
    if (t2 > t1) this.log('cut', { t: round(t1), dur: round(t2 - t1), text: what });
  }

  // Scroll the page (as a reader would, smoothly) so the element's top sits `top` px below the viewport's top.
  async scrollTo(locator, top = 90) {
    this.unbox();
    const box = await locator.boundingBox();
    if (!box) throw new TakeError(`${this.name}: cannot scroll to an element that is not visible`);
    await this.page.evaluate((dy) => window.scrollBy({ top: dy, behavior: 'smooth' }), box.y - top);
    await this.settle();
  }

  async scrollToY(y, smooth = true) {
    this.unbox();
    await this.page.evaluate(([y, s]) => window.scrollTo({ top: y, behavior: s ? 'smooth' : 'instant' }), [y, smooth]);
    await this.settle();
  }

  // Until scrollY has not changed for 200 ms.
  async settle() {
    await this.page.waitForFunction(() => new Promise((ok) => {
      const y0 = window.scrollY;
      setTimeout(() => ok(window.scrollY === y0), 200);
    }), null, { timeout: 10_000, polling: 50 });
    await this.painted();
  }

  // Throws unless the locator's text matches: a caption must never name what the screen does not show.
  async expectText(locator, pattern, what) {
    try {
      await locator.filter({ hasText: pattern }).first().waitFor({ state: 'visible', timeout: 15_000 });
    } catch {
      const shown = (await locator.first().innerText().catch(() => '(not visible)')).slice(0, 300);
      throw new TakeError(`${this.name}: expected ${what} (${pattern}) but the page shows: ${shown}`);
    }
  }

  async noBusyBanner() {
    const busy = tab(this.page).getByText('The simulator is still running an earlier attempt');
    if (await busy.count() && await busy.first().isVisible()) throw new TakeError(`${this.name}: the simulator was busy; re-record this take`);
  }

  async gif() {
    const src = await ui.gif(this.page).getAttribute('src');
    const loop = gifLoopSeconds(Buffer.from(src.slice(src.indexOf(',') + 1), 'base64'));
    const seen = await this.page.evaluate(() => window.__vpGifSeen);
    return { loop, phase: seen == null ? null : ((Date.now() - seen) / 1000) % loop };
  }

  // Before scrolling to the execution: if its GIF already plays, wait until it is about to start over (cut from the
  // video; the page is still meanwhile), so the Execute and Judge captions get one whole loop from its first frame.
  async alignGif(lead = 0.6) {
    const { loop, phase } = await this.gif();
    if (phase === null || phase < 0.15) return;
    const wait = loop - phase - lead;
    if (wait <= 0) return;
    const t1 = this.now();
    await this.page.waitForTimeout(wait * 1000);
    this.log('cut', { t: round(t1), dur: round(this.now() - t1), text: 'until the execution starts over' });
  }

  // Seconds until the GIF starts over, less a margin: the last caption ends on its final, held frame.
  async gifLeft(margin = 0.5) {
    const { loop, phase } = await this.gif();
    return loop - (phase ?? 0) - margin;
  }

  finish() {
    this.clear();
    this.end = round(this.now());
  }
}

// Duration of a WebM file in seconds, from its Segment Info: Duration (0x4489, a float in TimecodeScale units) and
// TimecodeScale (0x2AD7B1, nanoseconds per unit, 1,000,000 by default). null when the header does not say.
function webmDurationSeconds(buf) {
  const head = buf.subarray(0, Math.min(buf.length, 1 << 16));
  let scale = 1e6;
  const ts = head.indexOf(Buffer.from([0x2a, 0xd7, 0xb1]));
  if (ts >= 0) {
    const n = head[ts + 3] & 0x0f; // size byte 0x8n: n bytes follow
    let v = 0;
    for (let k = 0; k < n; k++) v = v * 256 + head[ts + 4 + k];
    if (v > 0) scale = v;
  }
  for (let i = head.indexOf(Buffer.from([0x44, 0x89])); i >= 0; i = head.indexOf(Buffer.from([0x44, 0x89]), i + 1)) {
    if (head[i + 2] === 0x88) return (head.readDoubleBE(i + 3) * scale) / 1e9;
    if (head[i + 2] === 0x84) return (head.readFloatBE(i + 3) * scale) / 1e9;
  }
  return null;
}

class LostFramesError extends Error {}

function round(x) {
  return Math.round(x * 1000) / 1000;
}

async function rectOf(locator) {
  await locator.waitFor({ state: 'visible', timeout: 15_000 });
  const b = await locator.boundingBox();
  if (!b) throw new TakeError('no bounding box for a highlighted element');
  return { x: round(b.x), y: round(b.y), width: round(b.width), height: round(b.height) };
}

// ---- UI helpers --------------------------------------------------------------------------------------------------

// Gradio keeps a tab it has rendered in the DOM, hidden, so everything below is looked up in the visible tab only.
const tab = (p) => p.locator('[role="tabpanel"]:visible');
const ui = {
  liveTab: (p) => p.getByRole('tab', { name: 'Live' }),
  replayTab: (p) => p.getByRole('tab', { name: 'Replay' }),
  button: (p, name) => tab(p).getByRole('button', { name, exact: true }),
  attempt: (p) => tab(p).getByRole('combobox', { name: 'Attempt' }),
  outcome: (p) => tab(p).locator('.vp-card:has(.vp-outcome)'),
  infoCard: (p) => tab(p).locator('.vp-card:not(:has(.vp-outcome))'),
  scene: (p) => tab(p).locator('.block:has(.image-container)'),
  rounds: (p) => tab(p).locator('.vp-round'),
  step: (p, title) => tab(p).locator('.vp-step', { has: p.locator('h2', { hasText: new RegExp(`^${title}$`) }) }),
  plan: (p) => tab(p).locator('.block:has(.cm-editor)').filter({ hasText: 'Plan (JSON)' }),
  gif: (p) => tab(p).locator('.vp-exec img'),
  judge: (p) => tab(p).locator('.prose:has(.vp-verdict)').filter({ hasText: 'Criterion' }),
};

// Load scene and the run button are enabled, a scene picture is shown and nothing is locked (visible tab only).
async function liveIdle(page) {
  await page.waitForFunction(() => {
    const shown = (e) => e && e.offsetParent !== null;
    const btn = (name) => [...document.querySelectorAll('button')].find((b) => shown(b) && b.textContent.trim() === name);
    const load = btn('Load scene');
    const run = btn('Plan, verify, execute');
    return load && run && !load.disabled && !run.disabled && [...document.querySelectorAll('.vp-card')].some(shown) &&
      [...document.querySelectorAll('.image-container img')].some(shown);
  }, null, { timeout: 120_000, polling: 100 });
}

// The page opens on the Replay tab (the UI runs with --replay) and loads the default live scene in the background.
// Showing the Live tab until that load is done keeps the Replay button from finding the simulator busy.
async function openPage(take) {
  const { page } = take;
  await page.goto(take.url, { waitUntil: 'domcontentloaded' });
  await ui.liveTab(page).click();
  await liveIdle(page);
}

async function openReplay(take) {
  const { page } = take;
  await openPage(take);
  await ui.replayTab(page).click();
  await tab(page).getByText('Opened', { exact: true }).waitFor({ state: 'visible', timeout: 120_000 });
  await tab(page).getByText('(400 recorded attempts)').waitFor({ state: 'visible', timeout: 10_000 });
}

async function chooseAttempt(take, label) {
  const { page } = take;
  await take.click(ui.attempt(page), 'Attempt list');
  const option = page.getByRole('option', { name: label, exact: true });
  await option.waitFor({ state: 'visible', timeout: 10_000 });
  await take.hold(0.6);
  await take.click(option, label);
  await page.waitForFunction((v) => document.querySelector('[role="combobox"][aria-label="Attempt"]')?.value === v,
    label, { timeout: 10_000 });
  await take.hold(0.5);
}

async function replay(take, label, expectHeadline) {
  const { page } = take;
  await chooseAttempt(take, label);
  await take.click(ui.button(page, 'Replay'), 'Replay');
  await take.until('the replayed attempt', async () => {
    await ui.outcome(page).filter({ hasText: expectHeadline }).waitFor({ state: 'visible', timeout: 120_000 });
  });
  await take.noBusyBanner();
  await take.painted();
}

// ---- the takes ---------------------------------------------------------------------------------------------------

async function titleTake(take) {
  await take.card('title', 4.5);
}

// The architecture, one part at a time, on the system-overview diagram. A node is the nearest node-sized rect
// (at least 144 units wide) before its label; a zone is the nearest 120-unit-high band before its label.
async function architectureTake(take) {
  const { page } = take;
  await take.card('architecture', 0.8);
  const before = (label, rect) => page.locator(`xpath=//*[local-name()="text" and normalize-space(.)="${label}"]` +
    `/preceding-sibling::*[local-name()="rect" and ${rect}][1]`);
  const node = (label) => before(label, '@width >= 144');
  const zone = (label) => before(label, '@height = 120');

  await take.caption('A harness generates solvable and provably impossible tasks, runs them and builds the report.');
  await take.box(zone('EVALUATION HARNESS'));
  await take.hold(3.5);

  await take.caption('NVIDIA Nemotron on Nebius Token Factory answers each task with a typed plan, or a refusal.');
  await take.box(node('Nemotron planner'));
  await take.hold(4);

  await take.caption('The plan becomes a joint trajectory. Eight kinematic and collision checks test it in a preview simulation.');
  await take.box([node('Motion layer'), node('Verifier')]);
  await take.hold(5);

  await take.caption('A rejected plan goes back with findings: what failed, by how much, and what to change. Up to 3 repairs.');
  // two boxes, not their union, which would cut through the Cosmos node between them
  await take.box(node('Verifier'));
  await take.box(node('Nemotron planner'));
  await take.hold(5);

  await take.caption('Only an approved plan runs in PyBullet physics. An independent judge, sharing no code with the checks, scores it.');
  await take.box([node('Executor'), node('Judge')]);
  await take.hold(5);

  await take.caption('Optional: NVIDIA Cosmos3-Edge looks at rendered frames after the checks pass. It can only block.');
  await take.box(node('Cosmos visual gate'));
  await take.hold(4);
}

async function resultsTake(take) {
  await take.card('results', 17);
}

async function closingTake(take) {
  await take.card('closing', 4.5);
}

async function liveTake(take) {
  const { page } = take;
  await take.card('live', 3);
  const t0 = take.now();
  await openPage(take);
  await take.scrollToY(140, false);
  take.cutSince(t0, 'page load');

  const wall = tab(page).getByLabel('Basic scene with wall');
  await take.caption('A simulated Franka Panda arm, two cubes and a bin. The task is typed in plain English.');
  await take.box(tab(page).locator('.block.padded').filter({ has: page.getByRole('textbox', { name: /^Instruction/ }) }));
  await take.hold(4);

  await take.caption('Put a wall between the cubes and the bin. The planner is the rule-based baseline: no LLM, no API key.');
  await take.click(wall, 'Basic scene with wall');
  await take.until('the scene with the wall', async () => {
    await ui.infoCard(page).filter({ hasText: '1 wall' }).waitFor({ state: 'visible', timeout: 60_000 });
    await liveIdle(page);
  });
  await take.painted();
  await take.box(ui.scene(page));
  await take.hold(2.5);
  await take.box(tab(page).locator('.block.padded').filter({ hasText: 'Rule-based baseline (no LLM)' }));
  await take.hold(2.5);

  await take.caption('One click runs the whole loop: plan, verify, execute, judge.');
  const run = ui.button(page, 'Plan, verify, execute');
  await take.click(run, 'Plan, verify, execute');
  const tRun = take.now();
  await tab(page).locator('.cm-content').filter({ hasText: 'pick_place' }).waitFor({ state: 'visible', timeout: 30_000 });
  await take.hold(1);
  await take.scrollTo(ui.step(page, 'Plan'), 70);

  await take.caption('1 Plan: a typed plan with the grasp, waypoints over the wall and the drop-off point.');
  await take.expectText(ui.plan(page), /transport_via/, 'the plan JSON');
  await take.expectText(tab(page).locator('.vp-banner'), /Cosmos visual check off/,
    'the Cosmos-off banner (start the UI without --cosmos-url for this take)');
  await take.box(ui.plan(page));
  await take.hold(4);

  await take.until('the checks and the execution', async () => {
    await ui.rounds(page).first().filter({ hasText: 'Executed' }).waitFor({ state: 'visible', timeout: 60_000 });
  });
  await take.caption('2 Verify: eight kinematic and collision checks run in a preview simulation. All pass, so the plan is approved.');
  await take.expectText(ui.rounds(page).first(), /Release in bin/, 'the passed checks');
  await take.box(ui.rounds(page).first());
  await take.hold(4.5);
  take.clear();

  await take.until('the recorded execution', async () => {
    await ui.gif(page).waitFor({ state: 'visible', timeout: 60_000 });
    await ui.outcome(page).filter({ hasText: 'Success' }).waitFor({ state: 'visible', timeout: 10_000 });
  }, { lead: 0.3, tail: 0.2 });
  log(`live: the run took ${(take.now() - tRun).toFixed(1)} s from the click to the finished panels`);
  await take.alignGif();
  await take.scrollTo(ui.step(page, 'Execute'), 20);
  const left = await take.gifLeft();
  await take.caption('3 Execute: only the approved plan runs, in PyBullet physics: over the wall and into the bin.');
  await take.box(ui.gif(page));
  await take.hold(left * 0.48);
  await take.caption('4 Judge: an independent judge scores the result: cube in the bin, no unsafe contact, all limits kept.');
  await take.expectText(ui.judge(page), /No unsafe contact/, 'the judge criteria');
  await take.box(ui.judge(page));
  await take.hold(left * 0.52);
}

async function repairTake(take) {
  const { page } = take;
  await take.card('repair', 3.5);
  const t0 = take.now();
  await openReplay(take);
  take.cutSince(t0, 'page load');

  await take.caption('A recorded attempt by NVIDIA Nemotron 3.5 Lightning on a held-out task. Replay needs no API key, GPU or network.');
  await take.box([tab(page).locator('.block.padded').filter({ has: page.getByRole('textbox', { name: /^Run folder or recording/ }) }),
    tab(page).locator('.block.padded').filter({ has: page.getByRole('textbox', { name: /^Seed-set folder/ }) })]);
  await take.hold(4);
  await take.caption('Pick the attempt and replay it.');
  await replay(take, 'Task 1 (test set) · Executed · success', 'Success');

  await take.scrollTo(ui.outcome(page), 90);
  await take.caption('Task: pick the object next to the red cylinder and place it in the bin. Nemotron needed three plans.');
  await take.expectText(ui.outcome(page), /3 planner calls/, 'three planner calls');
  await take.box(ui.outcome(page));
  await take.hold(4.5);

  const rounds = ui.rounds(page);
  await take.scrollTo(rounds.nth(0), 80);
  await take.caption("Round 1: Nemotron's first plan sets the drop-off point 8\u00a0cm too high. The checks reject it before anything moves.");
  await take.expectText(rounds.nth(0), /measured 0\.08 m/, 'the drop-off height finding');
  await take.box(rounds.nth(0));
  await take.hold(5.5);

  await take.caption('Round 2: told what failed, it fixes the height. Now the carried cylinder passes 0.95\u00a0cm from the wall; the minimum is 1\u00a0cm.');
  await take.expectText(rounds.nth(1), /clearance 0\.95 cm \(minimum 1\.00 cm\)/, 'the clearance finding');
  await take.box(rounds.nth(1));
  await take.hold(6);

  // the magenta close-up of the failed parts (in the D78 cut) is left out, to keep the video under 3 minutes with
  // the architecture chapter
  await take.scrollTo(ui.plan(page), 70);
  await take.caption('Round 3: Nemotron adds high waypoints (left) to carry the cylinder over the wall. All eight checks pass.');
  await take.expectText(rounds.nth(2), /Executed/, 'round 3 executed');
  await take.expectText(ui.plan(page), /\[0\.45, 0\.3, 0\.3\]/, 'the repaired waypoints');
  await take.box(ui.plan(page));
  await take.box(rounds.nth(2));
  await take.hold(5.5);

  await take.alignGif();
  await take.scrollTo(ui.step(page, 'Execute'), 20);
  const left = await take.gifLeft();
  await take.caption('3 Execute: the approved plan runs in PyBullet physics, played back from the recording.');
  await take.box(ui.gif(page));
  await take.hold(left * 0.48);
  await take.caption('4 Judge: the independent judge confirms it: the cylinder rests in the bin and nothing unsafe happened.');
  await take.expectText(ui.judge(page), /Object resting in the bin/, 'the judge criteria');
  await take.box(ui.judge(page));
  await take.hold(left * 0.52);
}

async function refusalTake(take) {
  const { page } = take;
  await take.card('refusal', 3);
  const t0 = take.now();
  await openReplay(take);
  take.cutSince(t0, 'page load');

  await take.caption('Another recorded Nemotron attempt from the held-out set.');
  await replay(take, 'Task 3 (test set) · Refused · success', 'Correct refusal');

  await take.scrollToY(10_000);
  await take.caption('The yellow cylinder is out of the arm\'s reach. The planner is not told this.');
  await take.expectText(ui.outcome(page), /impossible: object out of reach/, 'the hidden ground truth');
  await take.box(ui.outcome(page));
  await take.hold(4);

  await take.caption('Nemotron refuses instead of planning, and says why: the cylinder is about 1.2\u00a0m away, beyond the 0.95\u00a0m reach.');
  await take.expectText(ui.rounds(page).first(), /approximately 1\.20 m/, 'the refusal explanation');
  await take.box(ui.rounds(page).first());
  await take.hold(5.5);

  await take.caption('The judge accepts the refusal: refusing is correct only when the task is impossible.', { pos: 'top' });
  await take.expectText(ui.judge(page), /Refusal matches the task/, 'the refusal criterion');
  await take.box(ui.judge(page));
  await take.hold(4);
}

// The Cosmos gate's verdicts vary between runs (D73: false rejections, and replies the strict parser cannot read), so
// this take reads the round's verdict from the page and takes the caption written for it.
async function cosmosTake(take) {
  const { page } = take;
  await take.card('cosmos', 2.5);
  const t0 = take.now();
  await openPage(take);
  // One round only: the chapter shows one plan and what Cosmos makes of it (the slider stays visible at 0)
  const rounds = tab(page).locator('.block').filter({ hasText: 'Repair rounds' });
  await rounds.locator('input[type="number"]').fill('0');
  await rounds.locator('input[type="number"]').press('Enter');
  await page.waitForFunction(() => [...document.querySelectorAll('input[type="range"]')]
    .some((r) => r.offsetParent && r.value === '0'), null, { timeout: 10_000 });
  await take.scrollToY(0, false);
  take.cutSince(t0, 'page load');

  await take.caption('Now with the Cosmos visual check on: the same rule-based baseline, on the basic scene without ' +
    'the wall.', { pos: 'top' });
  await take.hold(0.6);
  await take.click(ui.button(page, 'Plan, verify, execute'), 'Plan, verify, execute');
  await take.hold(1.2);
  await take.until('the Cosmos verdicts', async () => {
    await ui.outcome(page).waitFor({ state: 'visible', timeout: 120_000 });
    await liveIdle(page);
  });
  await take.painted();
  const banner = tab(page).locator('.vp-banner');
  await take.expectText(banner, /Cosmos visual check on: a learned, advisory check.*0\.61.*0\.52/,
    'the Cosmos-on banner with the shadow-study numbers (start the UI with --cosmos-url for this take)');
  const round = ui.rounds(page).first();
  const visual = await round.evaluate((r) => {
    const badge = r.querySelector('.vp-visual .vp-badge');
    const rows = [...r.querySelectorAll('table.vp-checks')].filter((t) => /Moment/i.test(t.innerText))
      .flatMap((t) => [...t.querySelectorAll('tr')].slice(1).map((tr) => [...tr.cells].map((c) => c.innerText.trim())));
    return { status: badge ? badge.innerText.replace(/[^A-Za-z]/g, '') : null, rows };
  });
  log(`cosmos: the gate said ${visual.status}; ${visual.rows.map((r) => `${r[0]}: ${r[2]} (${r[3]})`).join('; ')}`);
  const stage = { pregrasp: 'pre-grasp', grasp: 'grasp', prerelease: 'pre-release', pre_release: 'pre-release' };
  let verdict;
  if (visual.status === 'Fail') {
    const failed = visual.rows.find((r) => /Fail/.test(r[2]));
    const where = stage[failed?.[0]] ?? failed?.[0];
    verdict = `The eight checks pass, but Cosmos rejects the plan at the ${where} frame, so nothing moves.`;
  } else if (visual.status === 'Unsure') {
    verdict = 'The eight checks pass, but Cosmos gives no usable answer, even with a third view, so nothing moves.';
  } else if (visual.status === 'Pass') {
    verdict = 'The eight checks pass, and Cosmos agrees at every key frame, so the plan runs.';
  } else {
    throw new TakeError(`cosmos: the gate reported ${visual.status ?? 'nothing'}; is the Cosmos server answering?`);
  }
  await take.expectText(round, /All 8 checks ran/, 'all eight checks passed');

  await take.caption('It is advisory, not verification: in our shadow study its balanced accuracy was 0.61 on real ' +
    'attempts and 0.52 on injected errors.', { pos: 'top' });
  await take.box(banner);
  await take.hold(4);

  await take.scrollTo(ui.step(page, 'Verify'), 20);
  await take.caption(verdict);
  await take.box(round);
  await take.hold(4);

  const gallery = tab(page).locator('.block:has(.grid-wrap)').filter({ hasText: 'What Cosmos saw' });
  await take.caption(visual.status === 'Pass'
    ? 'These are the frames it saw. Even when it agrees, it only advises: the checks decide.'
    : 'These are the frames it saw. The plan passed every check, yet it was stopped: this is why Cosmos stays advisory.',
  { pos: 'top' });
  await take.box(gallery);
  await take.hold(4);
}

// ---- recording ---------------------------------------------------------------------------------------------------

function log(message) {
  process.stdout.write(`${message}\n`);
}

async function recordChapter(browser, chapter, opts) {
  const tmp = path.join(opts.out, `.tmp-${chapter.name}`);
  await fs.rm(tmp, { recursive: true, force: true });
  const context = await browser.newContext({
    viewport: VIEWPORT, deviceScaleFactor: 1, colorScheme: 'light', reducedMotion: 'no-preference',
    recordVideo: { dir: tmp, size: VIEWPORT },
  });
  await context.addInitScript(GIF_WATCH);
  const before = Date.now();
  const page = await context.newPage();
  const after = Date.now();
  const take = new Take(page, chapter.name, opts.url);
  // Playwright's recorder stamps frames from its creation, inside newPage; the midpoint is within ~25 ms of it, and
  // burn.mjs corrects the rest from the video's creation_time.
  take.ref = (before + after) / 2;
  page.setDefaultTimeout(30_000);
  let failure = null;
  try {
    await chapter.run(take);
    take.finish();
  } catch (err) {
    failure = err;
  }
  await context.close();
  const video = page.video();
  const webm = path.join(opts.out, `${chapter.name}.webm`);
  const eventsPath = path.join(opts.out, `${chapter.name}.events.json`);
  if (failure) {
    await fs.rm(tmp, { recursive: true, force: true });
    throw failure;
  }
  await video.saveAs(webm);
  await fs.rm(tmp, { recursive: true, force: true });
  // The video starts at the first frame the browser sent; if that came late, the video is shorter than the log and
  // every event would be early by the difference. Such a take has drifted: record it again.
  const duration = webmDurationSeconds(await fs.readFile(webm));
  if (duration !== null && duration < take.end - 0.6) {
    throw new LostFramesError(`${chapter.name}: the video is ${duration.toFixed(2)} s but the take ran ` +
      `${take.end.toFixed(2)} s (frames missing at its start)`);
  }
  const doc = {
    chapter: chapter.name, video: path.basename(webm), viewport: VIEWPORT, url: opts.url,
    recorded_at: new Date(take.ref).toISOString(), ref_epoch_ms: take.ref, start: take.start, end: take.end,
    playwright: opts.playwrightVersion, events: take.events,
  };
  await fs.writeFile(eventsPath, `${JSON.stringify(doc, null, 2)}\n`);
  const shown = take.end - take.start - take.events.filter((e) => e.type === 'cut').reduce((s, e) => s + e.dur, 0);
  log(`${chapter.name}: ${path.relative(REPO, webm)} (${take.end.toFixed(1)} s recorded, about ${shown.toFixed(1)} s ` +
    `after cuts, ${take.events.length} events)`);
}

function parseArgs(argv) {
  const opts = { url: DEFAULT_URL, chapters: null, out: path.join(REPO, 'runs', 'video'), headed: false, list: false,
    cosmos: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === '--url') opts.url = value();
    else if (a === '--chapter') opts.chapters = [...(opts.chapters ?? []), ...value().split(',')];
    else if (a === '--out') opts.out = path.resolve(value());
    else if (a === '--headed') opts.headed = true;
    else if (a === '--cosmos') opts.cosmos = true;
    else if (a === '--list') opts.list = true;
    else if (a === '-h' || a === '--help') {
      log('usage: node record.mjs [--url URL] [--chapter NAME[,NAME...]] [--cosmos] [--out DIR] [--headed] [--list]');
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.list) {
    log(CHAPTERS.map((c) => c.name + (c.optional ? ' (optional: --cosmos)' : '')).join('\n'));
    return;
  }
  const names = CHAPTERS.map((c) => c.name);
  const unknown = (opts.chapters ?? []).filter((n) => !names.includes(n));
  if (unknown.length) throw new Error(`unknown chapter ${unknown.join(', ')} (chapters: ${names.join(', ')})`);
  const wanted = [...(opts.chapters ?? []), ...(opts.cosmos ? ['cosmos'] : [])];
  const chosen = wanted.length ? CHAPTERS.filter((c) => wanted.includes(c.name)) : CHAPTERS.filter((c) => !c.optional);
  await fs.mkdir(opts.out, { recursive: true });
  // burn.mjs joins the chapters in this order (the full list, also when only some chapters are re-recorded); an
  // optional chapter joins only once it is recorded
  const order = { chapters: names, optional: CHAPTERS.filter((c) => c.optional).map((c) => c.name) };
  await fs.writeFile(path.join(opts.out, 'order.json'), `${JSON.stringify(order)}\n`);
  opts.playwrightVersion = JSON.parse(await fs.readFile(path.join(HERE, 'node_modules', 'playwright', 'package.json'), 'utf8')).version;

  // The UI asks for Segoe UI and Consolas; on WSL, fonts.conf points this Chromium (only) at the Windows copies, so
  // the recording looks as the UI does on Windows. Elsewhere the page falls back to the platform fonts.
  const env = { ...process.env };
  if (existsSync(WINDOWS_FONTS) && !process.env.FONTCONFIG_FILE) env.FONTCONFIG_FILE = path.join(HERE, 'fonts.conf');
  const browser = await chromium.launch({ headless: !opts.headed, env });
  try {
    for (const chapter of chosen) {
      for (let attempt = 1; ; attempt++) {
        try {
          await recordChapter(browser, chapter, opts);
          break;
        } catch (err) {
          if (!(err instanceof LostFramesError) || attempt >= 3) throw err;
          log(`${err.message}; recording it again (attempt ${attempt + 1} of 3)`);
        }
      }
    }
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stderr.write(`record.mjs: ${err instanceof TakeError || err instanceof LostFramesError ? err.message : err.stack}\n`);
  process.exit(1);
});
