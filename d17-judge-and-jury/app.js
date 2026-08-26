/**
 * UI: build an evaluation protocol, run it over 400 pairwise items, and see
 * both the numbers you'd report and the leaderboard those numbers produce —
 * which is where judge bias actually does its damage.
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const pct = v => (v * 100).toFixed(1) + '%';

  const PRESETS = {
    cheap: { panel: ['judge-C1'], swap: false, aggregate: 'majority', excludeSelf: false, lengthNormalize: false },
    mono: { panel: ['judge-A1', 'judge-A2'], swap: false, aggregate: 'majority', excludeSelf: false, lengthNormalize: false },
    full: {
      panel: ['judge-A1', 'judge-B1', 'judge-C1', 'judge-H'],
      swap: true, aggregate: 'confidence', excludeSelf: true, lengthNormalize: true,
    },
  };

  let last = null;

  /* ---------- build the judge checkboxes ---------- */

  function renderJudges() {
    const box = $('judges');
    box.innerHTML = '';
    for (const j of JURY.JUDGES) {
      const label = document.createElement('label');
      label.className = 'jcard';
      const bias = [
        j.position > 0.14 ? 'position' : null,
        j.verbosity > 0.25 ? 'verbose-loving' : null,
        j.selfPref > 0.19 ? 'self-preferring' : null,
        j.noise > 0.14 ? 'noisy' : null,
      ].filter(Boolean);
      label.innerHTML =
        `<input type="checkbox" data-j="${j.id}">` +
        `<span class="jcard__body">` +
        `<b>${j.id}</b> <span class="jcard__fam">family ${j.family}</span>` +
        `<span class="jcard__bias">${bias.length ? bias.join(' · ') : 'well-behaved'} · ${j.cost} tok</span>` +
        `</span>`;
      box.appendChild(label);
    }
    for (const cb of box.querySelectorAll('input')) cb.addEventListener('change', run);
  }

  function readCfg() {
    const ids = [...document.querySelectorAll('#judges input:checked')].map(c => c.dataset.j);
    return {
      panel: ids.map(id => JURY.JUDGES.find(j => j.id === id)),
      swap: $('c-swap').checked,
      aggregate: $('c-agg').checked ? 'confidence' : 'majority',
      excludeSelf: $('c-self').checked,
      lengthNormalize: $('c-len').checked,
    };
  }

  function writeCfg(p) {
    for (const cb of document.querySelectorAll('#judges input')) {
      cb.checked = p.panel.includes(cb.dataset.j);
    }
    $('c-swap').checked = p.swap;
    $('c-agg').checked = p.aggregate === 'confidence';
    $('c-self').checked = p.excludeSelf;
    $('c-len').checked = p.lengthNormalize;
    run();
  }

  /* ---------- run + render ---------- */

  function run() {
    const cfg = readCfg();
    if (cfg.panel.length === 0) {
      $('status').textContent = 'pick at least one judge — an eval with no judges agrees with nothing';
      $('status').className = 'status status--warn';
      return;
    }
    const r = JURY.runEval(cfg, { n: 400, seed: 11 });
    last = r;

    $('m-agree').textContent = pct(r.acc);
    $('m-kappa').textContent = r.kappa.toFixed(2);
    $('m-close').textContent = pct(r.closeAcc);
    $('m-cost').textContent = Math.round(r.costPerItem).toLocaleString();

    $('m-close').className = 'metric__num' + (r.closeAcc < 0.65 ? ' metric__num--bad' : '');
    $('m-kappa').className = 'metric__num' + (r.kappa < 0.6 ? ' metric__num--bad' : '');

    drawAttribution(r.attribution);
    drawLeaderboard(r);

    const trust = r.kappa >= 0.6 && r.topCorrect && r.spearman >= 0.9;
    $('status').textContent = trust
      ? `✓ this protocol is trustworthy enough to gate a launch (κ ${r.kappa.toFixed(2)}, ranking intact)`
      : !r.topCorrect
        ? `✗ this eval crowns the wrong model — it says ${r.judged[0].id}, the truth says ${r.truth[0].id}`
        : `✗ κ ${r.kappa.toFixed(2)} is too low to gate a launch — the ranking survived here, but on luck`;
    $('status').className = 'status ' + (trust ? 'status--ok' : 'status--bad');
  }

  function drawAttribution(a) {
    const box = $('attr');
    box.innerHTML = '';
    const rows = [
      ['position bias', a.position, 'prefers whichever answer came first'],
      ['verbosity bias', a.verbosity, 'prefers the longer answer'],
      ['self-preference', a.selfPref, 'prefers its own model family'],
    ];
    const max = Math.max(0.02, ...rows.map(r => Math.abs(r[1])));
    for (const [name, v, note] of rows) {
      const row = document.createElement('div');
      row.className = 'attr__row';
      const w = (Math.abs(v) / max) * 100;
      row.innerHTML =
        `<span class="attr__name">${name}</span>` +
        `<span class="attr__track"><span class="attr__fill${v > 0 ? '' : ' attr__fill--neg'}" style="width:${w}%"></span></span>` +
        `<span class="attr__val">${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}pp</span>` +
        `<span class="attr__note">${note}</span>`;
      box.appendChild(row);
    }
  }

  function drawLeaderboard(r) {
    const box = $('board');
    box.innerHTML = '';
    const trueRank = {};
    r.truth.forEach((x, i) => { trueRank[x.id] = i + 1; });

    r.judged.forEach((x, i) => {
      const moved = trueRank[x.id] - (i + 1);
      const row = document.createElement('div');
      row.className = 'brow' + (moved !== 0 ? ' brow--moved' : '');
      row.innerHTML =
        `<span class="brow__pos">${i + 1}</span>` +
        `<span class="brow__id">${x.id}</span>` +
        `<span class="brow__bar"><span style="width:${(x.v * 100).toFixed(0)}%"></span></span>` +
        `<span class="brow__v">${pct(x.v)} win</span>` +
        `<span class="brow__delta">${moved === 0 ? '·' : moved > 0 ? `▲ ${moved} vs truth` : `▼ ${-moved} vs truth`}</span>`;
      box.appendChild(row);
    });

    $('board-note').textContent = r.topCorrect
      ? `ranking correlation with ground truth: ρ = ${r.spearman.toFixed(2)}`
      : `⚠ your eval would ship ${r.judged[0].id}; ground truth prefers ${r.truth[0].id} (ρ = ${r.spearman.toFixed(2)})`;
    $('board-note').className = 'hint' + (r.topCorrect ? '' : ' hint--bad');
    $('truth-note').textContent = 'ground truth: ' + r.truth.map(t => t.id).join(' > ');
  }

  /* ---------- wiring ---------- */

  for (const id of ['c-swap', 'c-agg', 'c-self', 'c-len']) {
    $(id).addEventListener('change', run);
  }
  $('p-cheap').addEventListener('click', () => writeCfg(PRESETS.cheap));
  $('p-mono').addEventListener('click', () => writeCfg(PRESETS.mono));
  $('p-full').addEventListener('click', () => writeCfg(PRESETS.full));

  renderJudges();
  writeCfg(PRESETS.cheap);

  window.JURY_APP = { run, writeCfg, PRESETS, last: () => last, readCfg };
})();
