#!/usr/bin/env node
// Burns the overlays into the clean takes from record.mjs and joins them into the demo video (W4.3).
//
// From each <chapter>.events.json it maps the take's clock onto the cut video (the chapter start and the cuts), then
// writes:
//   ass/<chapter>.ass        captions, highlight boxes and click rings for that chapter (burned in with libass)
//   chapters/<chapter>.mp4   the chapter: H.264, yuv420p, 1920x1080, 30 fps
//   vplan-demo.mp4           all chapters joined, in order.json's order (an optional chapter only once recorded)
//   captions.srt             the captions on the joined timeline (for the upload)
//   vplan-demo.ass           every overlay on the joined timeline
//   frames/                  one PNG per event from vplan-demo.mp4, and index.tsv (time, chapter, type, text)
//
//   node burn.mjs [--out DIR] [--ffmpeg PATH] [--ffprobe PATH] [--chapter NAME[,NAME...]] [--no-cosmos] [--no-frames]
//
// ffmpeg: --ffmpeg, else $FFMPEG, else `ffmpeg` on PATH. It runs with DIR as its working directory and only relative
// paths, so a Windows ffmpeg.exe called from WSL works without path conversion (DIR must be on a Windows drive then).

import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const W = 1920;
const H = 1080;
const FPS = 30;
const FADE_S = 0.3; // fade in and out of every chapter, through the UI's background colour
const BG = '0xf8fafc'; // slate 50, the UI's page background

// Overlay style: captions white on slate 900; boxes orange 600 (it stands out from the UI's blue, green and red
// badges, and from the magenta of the highlighted parts), and click rings in the same orange with a white halo, so
// they also show on the UI's blue buttons.
const ASS_ORANGE = '&H0C58EA&'; // #EA580C as &HBBGGRR&
const ASS_WHITE = '&HFFFFFF&';
const BOX_PAD = 8;
const BOX_RADIUS = 12;

function parseArgs(argv) {
  const opts = { out: path.join(REPO, 'runs', 'video'), ffmpeg: process.env.FFMPEG || 'ffmpeg', ffprobe: null,
    chapters: null, frames: true, cosmos: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === '--out') opts.out = path.resolve(value());
    else if (a === '--ffmpeg') opts.ffmpeg = value();
    else if (a === '--ffprobe') opts.ffprobe = value();
    else if (a === '--chapter') opts.chapters = [...(opts.chapters ?? []), ...value().split(',')];
    else if (a === '--no-frames') opts.frames = false;
    else if (a === '--no-cosmos') opts.cosmos = false;
    else if (a === '-h' || a === '--help') {
      console.log('usage: node burn.mjs [--out DIR] [--ffmpeg PATH] [--ffprobe PATH] [--chapter NAME[,NAME...]] [--no-cosmos] [--no-frames]');
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  // ffprobe sits next to ffmpeg (ffmpeg.exe -> ffprobe.exe), or on PATH
  opts.ffprobe ??= opts.ffmpeg.includes('/') || opts.ffmpeg.includes('\\')
    ? path.join(path.dirname(opts.ffmpeg), path.basename(opts.ffmpeg).replace(/^ffmpeg/i, 'ffprobe'))
    : 'ffprobe';
  return opts;
}

function run(cmd, args, cwd) {
  const res = spawnSync(cmd, args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (res.error) throw new Error(`${cmd}: ${res.error.message}`);
  if (res.status !== 0) throw new Error(`${path.basename(cmd)} ${args.join(' ').slice(0, 400)}\n${res.stderr}`);
  return res.stdout;
}

const r3 = (x) => Math.round(x * 1000) / 1000;

// ---- the timeline of one chapter ---------------------------------------------------------------------------------

// Kept spans of the take [s, e), in take time: [start, end] minus the cuts. map(t) gives the time in the burned
// chapter.
function timeline(doc) {
  const cuts = doc.events.filter((e) => e.type === 'cut').map((e) => [e.t, e.t + e.dur]).sort((a, b) => a[0] - b[0]);
  let spans = [[doc.start, doc.end]];
  for (const [c0, c1] of cuts) {
    spans = spans.flatMap(([s, e]) => (c1 <= s || c0 >= e ? [[s, e]] : [[s, c0], [c1, e]].filter(([a, b]) => b - a > 1e-3)));
  }
  const segs = spans.map(([s, e]) => ({ s, e }));
  let acc = 0;
  for (const g of segs) {
    g.out = acc;
    acc += g.e - g.s;
  }
  const duration = acc;
  const map = (t) => {
    if (t <= segs[0].s) return 0;
    for (const g of segs) {
      if (t < g.e) return t < g.s ? g.out : g.out + (t - g.s); // inside a cut: where the next span starts
    }
    return duration;
  };
  return { segs, duration, map };
}

// The burned events of one chapter on its own timeline: {type, t, end, text, rect, pos}
function overlays(doc, tl) {
  const out = [];
  for (const e of doc.events) {
    if (!['caption', 'box', 'click', 'chapter'].includes(e.type)) continue;
    const t = tl.map(e.t);
    const dur = e.type === 'click' ? 0.6 : e.type === 'chapter' ? 0 : e.dur ?? 0;
    const end = e.type === 'click' ? t + dur : tl.map(e.t + dur);
    if (e.type !== 'chapter' && end - t < 0.05) continue; // wholly inside a cut
    out.push({ type: e.type, t: r3(t), end: r3(end), text: e.text ?? '', rect: e.rect ?? null, pos: e.pos ?? 'bottom' });
  }
  return out;
}

// ---- ASS ---------------------------------------------------------------------------------------------------------

function assTime(s) {
  const cs = Math.max(0, Math.round(s * 100));
  const h = Math.floor(cs / 360000);
  const m = Math.floor(cs / 6000) % 60;
  const sec = Math.floor(cs / 100) % 60;
  const p = (n) => String(n).padStart(2, '0');
  return `${h}:${p(m)}:${p(sec)}.${p(cs % 100)}`;
}

function assHeader() {
  // BorderStyle 3: a box behind each caption line in OutlineColour (slate 900), Outline = padding. Opaque, because
  // the boxes of two wrapped lines overlap and a translucent overlap would show as a darker band
  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${W}
PlayResY: ${H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Segoe UI Semibold,38,&H00FFFFFF,&H00FFFFFF,&H002A170F,&H002A170F,0,0,0,0,100,100,0,0,3,16,0,2,280,280,44,1
Style: CaptionTop,Segoe UI Semibold,38,&H00FFFFFF,&H00FFFFFF,&H002A170F,&H002A170F,0,0,0,0,100,100,0,0,3,16,0,8,280,280,44,1
Style: Shape,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
}

// A no-break space (between a number and its unit) becomes ASS's hard space \h, so a line never breaks inside "8 cm"
const assText = (s) => s.replace(/\\/g, '/').replace(/[{}]/g, (c) => (c === '{' ? '(' : ')')).replace(/\n/g, '\\N')
  .replace(/\u00a0/g, '\\h');

function roundedRect(w, h, r) {
  const f = (n) => Math.round(n * 10) / 10;
  [w, h, r] = [f(w), f(h), Math.min(r, w / 2, h / 2)];
  return `m ${r} 0 l ${f(w - r)} 0 b ${w} 0 ${w} 0 ${w} ${r} l ${w} ${f(h - r)} b ${w} ${h} ${w} ${h} ${f(w - r)} ${h} ` +
    `l ${r} ${h} b 0 ${h} 0 ${h} 0 ${f(h - r)} l 0 ${r} b 0 0 0 0 ${r} 0`;
}

function circle(rad) {
  const k = Math.round(rad * 0.5523 * 10) / 10;
  const d = 2 * rad;
  return `m ${rad} 0 b ${rad + k} 0 ${d} ${rad - k} ${d} ${rad} b ${d} ${rad + k} ${rad + k} ${d} ${rad} ${d} ` +
    `b ${rad - k} ${d} 0 ${rad + k} 0 ${rad} b 0 ${rad - k} ${rad - k} 0 ${rad} 0`;
}

function assLines(items, offset = 0) {
  const lines = [];
  const dlg = (layer, t0, t1, style, text) =>
    lines.push(`Dialogue: ${layer},${assTime(t0 + offset)},${assTime(t1 + offset)},${style},,0,0,0,,${text}`);
  for (const o of items) {
    if (o.type === 'caption') {
      dlg(5, o.t, o.end, o.pos === 'top' ? 'CaptionTop' : 'Caption', `{\\fad(120,120)}${assText(o.text)}`);
    } else if (o.type === 'box' && o.rect) {
      const x0 = Math.max(4, o.rect.x - BOX_PAD);
      const y0 = Math.max(4, o.rect.y - BOX_PAD);
      const x1 = Math.min(W - 4, o.rect.x + o.rect.width + BOX_PAD);
      const y1 = Math.min(H - 4, o.rect.y + o.rect.height + BOX_PAD);
      const shape = roundedRect(x1 - x0, y1 - y0, BOX_RADIUS);
      const pos = `\\an7\\pos(${Math.round(x0)},${Math.round(y0)})\\p1\\shad0\\1a&HFF&`;
      // a soft white halo under the orange line keeps it readable on dark renders as well as on white panels
      dlg(1, o.t, o.end, 'Shape', `{${pos}\\bord9\\3c${ASS_WHITE}\\3a&H60&\\blur2\\fad(160,160)}${shape}`);
      dlg(2, o.t, o.end, 'Shape', `{${pos}\\bord5\\3c${ASS_ORANGE}\\3a&H00&\\fad(160,160)}${shape}`);
    } else if (o.type === 'click' && o.rect) {
      const cx = Math.round(o.rect.x + o.rect.width / 2);
      const cy = Math.round(o.rect.y + o.rect.height / 2);
      const ring = `\\an5\\pos(${cx},${cy})\\p1\\shad0\\1a&HFF&\\fscx60\\fscy60`;
      dlg(3, o.t, o.t + 0.6, 'Shape',
        `{${ring}\\bord8\\3c${ASS_WHITE}\\3a&H40&\\t(0,550,\\fscx260\\fscy260\\3a&HFF&)}${circle(18)}`);
      dlg(4, o.t, o.t + 0.6, 'Shape',
        `{${ring}\\bord4\\3c${ASS_ORANGE}\\t(0,550,\\fscx260\\fscy260\\3a&HFF&)}${circle(18)}`);
      dlg(4, o.t, o.t + 0.35, 'Shape',
        `{\\an5\\pos(${cx},${cy})\\p1\\shad0\\bord2\\3c${ASS_WHITE}\\1c${ASS_ORANGE}\\1a&H20&\\fad(0,200)}${circle(9)}`);
    }
  }
  return lines;
}

// ---- SRT ---------------------------------------------------------------------------------------------------------

function srtTime(s) {
  const ms = Math.max(0, Math.round(s * 1000));
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
}

// ---- main --------------------------------------------------------------------------------------------------------

function probeCreation(opts, cwd, file) {
  try {
    const out = run(opts.ffprobe, ['-v', 'error', '-show_entries', 'format_tags=creation_time', '-of', 'default=nw=1:nk=1', file], cwd).trim();
    const ms = Date.parse(out);
    return Number.isFinite(ms) ? ms : null;
  } catch {
    return null;
  }
}

function probeDuration(opts, cwd, file) {
  return Number(run(opts.ffprobe, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', file], cwd).trim());
}

async function burnChapter(opts, name, doc, tl) {
  const cwd = opts.out;
  const items = overlays(doc, tl);
  await fs.writeFile(path.join(cwd, 'ass', `${name}.ass`), assHeader() + assLines(items).join('\n') + '\n');
  const parts = tl.segs.map((g, i) =>
    `[0:v]trim=start=${r3(g.s)}:end=${r3(g.e)},setpts=PTS-STARTPTS[s${i}]`);
  const joined = tl.segs.length > 1
    ? `${tl.segs.map((_, i) => `[s${i}]`).join('')}concat=n=${tl.segs.length}:v=1:a=0[cat]`
    : null;
  const src = joined ? '[cat]' : '[s0]';
  const d = tl.duration;
  // The takes are JPEG screencast frames encoded with BT.601; the output is tagged and converted to BT.709 (HD)
  const chain = `${src}fps=${FPS},scale=${W}:${H}:in_color_matrix=bt601:out_color_matrix=bt709:out_range=tv,` +
    `subtitles=filename=ass/${name}.ass,` +
    `fade=t=in:st=0:d=${FADE_S}:color=${BG},fade=t=out:st=${r3(Math.max(0, d - FADE_S))}:d=${FADE_S}:color=${BG},` +
    `format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv[v]`;
  const graph = [...parts, ...(joined ? [joined] : []), chain].join(';');
  run(opts.ffmpeg, ['-y', '-hide_banner', '-loglevel', 'error', '-i', doc.video, '-filter_complex', graph, '-map', '[v]',
    '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', String(FPS),
    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
    '-movflags', '+faststart', `chapters/${name}.mp4`], cwd);
  return items;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const cwd = opts.out;
  // record.mjs writes {chapters, optional}. The optional cosmos chapter joins when it has been recorded, unless
  // --no-cosmos leaves it out.
  const saved = JSON.parse(await fs.readFile(path.join(cwd, 'order.json'), 'utf8'));
  const optional = saved.optional ?? [];
  const order = (saved.chapters ?? saved).filter((name) => {
    if (!optional.includes(name)) return true;
    if (name === 'cosmos' && !opts.cosmos) return false;
    const recorded = existsSync(path.join(cwd, `${name}.events.json`));
    if (!recorded) console.log(`${name}: optional and not recorded, left out`);
    return recorded;
  });
  const unknown = (opts.chapters ?? []).filter((n) => !order.includes(n));
  if (unknown.length) throw new Error(`unknown chapter ${unknown.join(', ')} (chapters: ${order.join(', ')})`);
  for (const dir of ['ass', 'chapters']) await fs.mkdir(path.join(cwd, dir), { recursive: true });
  run(opts.ffmpeg, ['-hide_banner', '-version'], cwd); // fails early when ffmpeg is missing

  const chapters = [];
  for (const name of order) {
    const eventsFile = path.join(cwd, `${name}.events.json`);
    if (!existsSync(eventsFile)) throw new Error(`${eventsFile} is missing: record it with node record.mjs --chapter ${name}`);
    const doc = JSON.parse(await fs.readFile(eventsFile, 'utf8'));
    // Event times count from the midpoint of newPage; the video counts from its recorder's creation_time
    const created = probeCreation(opts, cwd, doc.video);
    const shift = created ? (doc.ref_epoch_ms - created) / 1000 : 0;
    if (Math.abs(shift) < 2) {
      doc.start = r3(doc.start + shift);
      doc.end = r3(doc.end + shift);
      for (const e of doc.events) e.t = r3(e.t + shift);
    }
    // record.mjs re-records such takes; this catches an older one (see webmDurationSeconds there)
    const recorded = probeDuration(opts, cwd, doc.video);
    if (recorded < doc.end - 0.6) {
      throw new Error(`${doc.video} is ${recorded.toFixed(2)} s but its take ran ${doc.end.toFixed(2)} s: frames are ` +
        `missing at its start, so its events would be early. Re-record it: node record.mjs --chapter ${name}`);
    }
    const tl = timeline(doc);
    const redo = !opts.chapters || opts.chapters.includes(name) || !existsSync(path.join(cwd, 'chapters', `${name}.mp4`));
    let items;
    if (redo) {
      items = await burnChapter(opts, name, doc, tl);
      console.log(`${name}: chapters/${name}.mp4 (${tl.duration.toFixed(2)} s, ${tl.segs.length} spans, shift ${shift.toFixed(3)} s)`);
    } else {
      items = overlays(doc, tl);
    }
    chapters.push({ name, items, duration: probeDuration(opts, cwd, `chapters/${name}.mp4`) });
  }

  await fs.writeFile(path.join(cwd, 'chapters', 'list.txt'), chapters.map((c) => `file '${c.name}.mp4'`).join('\n') + '\n');
  run(opts.ffmpeg, ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'chapters/list.txt',
    '-c', 'copy', '-movflags', '+faststart', 'vplan-demo.mp4'], cwd);
  const total = probeDuration(opts, cwd, 'vplan-demo.mp4');

  // The joined timeline: SRT (captions only), one ASS with every overlay, and the event index
  let offset = 0;
  const srt = [];
  const ass = [];
  const global = [];
  for (const c of chapters) {
    ass.push(...assLines(c.items, offset));
    for (const o of c.items) {
      global.push({ ...o, chapter: c.name, gt: r3(o.t + offset), gend: r3(o.end + offset) });
      if (o.type === 'caption') {
        srt.push(`${srt.length + 1}\n${srtTime(o.t + offset)} --> ${srtTime(o.end + offset)}\n${o.text.replace(/\u00a0/g, ' ')}\n`);
      }
    }
    offset += c.duration;
  }
  await fs.writeFile(path.join(cwd, 'captions.srt'), srt.join('\n'));
  await fs.writeFile(path.join(cwd, 'vplan-demo.ass'), assHeader() + ass.join('\n') + '\n');

  if (opts.frames) await extractFrames(opts, global);
  console.log(`vplan-demo.mp4: ${total.toFixed(2)} s (${chapters.map((c) => `${c.name} ${c.duration.toFixed(1)}`).join(', ')})`);
  console.log(`captions.srt: ${srt.length} captions; vplan-demo.ass; ${opts.frames ? `frames/: ${global.length} frames` : 'no frames'}`);
}

// One frame per event, where its overlay is fully drawn: a caption 0.8 s in (or halfway, if shorter), a box after its
// fade-in, a click while its ring is visible, a chapter 1 s in.
async function extractFrames(opts, global) {
  const cwd = opts.out;
  const dir = path.join(cwd, 'frames');
  await fs.rm(dir, { recursive: true, force: true });
  await fs.mkdir(dir, { recursive: true });
  const at = (o) => {
    if (o.type === 'caption') return o.gt + Math.min(0.8, (o.gend - o.gt) / 2);
    if (o.type === 'box') return o.gt + Math.min(0.5, (o.gend - o.gt) / 2);
    if (o.type === 'click') return o.gt + 0.12;
    return o.gt + 1.0;
  };
  const rows = global.map((o, i) => ({ ...o, i, n: Math.round(at(o) * FPS) })).sort((a, b) => a.n - b.n || a.i - b.i);
  const ns = [...new Set(rows.map((r) => r.n))];
  const expr = ns.map((n) => `eq(n,${n})`).join('+');
  run(opts.ffmpeg, ['-y', '-hide_banner', '-loglevel', 'error', '-i', 'vplan-demo.mp4', '-vf', `select='${expr}'`,
    '-fps_mode', 'passthrough', 'frames/raw-%04d.png'], cwd);
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40);
  const index = ['file\ttime_s\tchapter\ttype\ttext'];
  for (const [k, r] of rows.entries()) {
    const raw = path.join(dir, `raw-${String(ns.indexOf(r.n) + 1).padStart(4, '0')}.png`);
    const name = `${String(k + 1).padStart(3, '0')}-${r.chapter}-${r.type}-${slug(r.text) || 'event'}.png`;
    await fs.copyFile(raw, path.join(dir, name));
    index.push(`${name}\t${(r.n / FPS).toFixed(2)}\t${r.chapter}\t${r.type}\t${r.text}`);
  }
  for (const f of await fs.readdir(dir)) if (f.startsWith('raw-')) await fs.rm(path.join(dir, f));
  await fs.writeFile(path.join(dir, 'index.tsv'), index.join('\n') + '\n');
}

main().catch((err) => {
  console.error(`burn.mjs: ${err.message}`);
  process.exit(1);
});
