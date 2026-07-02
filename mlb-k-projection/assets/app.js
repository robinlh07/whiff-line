// assets/app.js — shared across all pages

// Header shadow on scroll + mobile nav toggle
document.addEventListener('DOMContentLoaded', () => {
  const header = document.getElementById('siteHeader');
  if (header) {
    window.addEventListener('scroll', () => {
      header.classList.toggle('scrolled', window.scrollY > 10);
    });
  }
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.style.display === 'flex';
      links.style.display = open ? 'none' : 'flex';
      links.style.cssText += 'position:absolute;top:60px;right:28px;background:#151d13;flex-direction:column;padding:16px 20px;border:1px solid rgba(244,241,230,0.28);gap:14px;';
    });
  }
});

// Formats a timestamp like "Jul 1, 2026, 6:12 PM ET" (best-effort, uses browser locale)
function formatStamp(iso) {
  if (!iso) return 'unknown';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

function pct(x) {
  if (x === null || x === undefined || isNaN(x)) return '—';
  return (x * 100).toFixed(1) + '%';
}

function signedPct(x) {
  if (x === null || x === undefined || isNaN(x)) return '—';
  const v = (x * 100).toFixed(1);
  return (x >= 0 ? '+' : '') + v + '%';
}

// ---- Today's board (index.html) ----
async function loadProjections() {
  const body = document.getElementById('projTableBody');
  const stamp = document.getElementById('projStamp');
  if (!body) return;
  try {
    const res = await fetch('data/projections.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('no data yet');
    const data = await res.json();
    if (stamp) stamp.textContent = 'Last refreshed ' + formatStamp(data.generated_at);
    const picks = (data.picks || []).slice().sort((a, b) => (b.edge ?? -1) - (a.edge ?? -1));
    if (picks.length === 0) {
      body.innerHTML = '<tr><td colspan="8" class="mono">No games with a positive edge right now. Check back after the next refresh.</td></tr>';
      return;
    }
    body.innerHTML = picks.map(p => `
      <tr>
        <td class="pitcher">${p.pitcher}</td>
        <td>${p.home_away === 'home' ? 'vs' : '@'} ${p.opponent}</td>
        <td>${p.line ?? '—'}</td>
        <td>${p.proj_p50 !== undefined ? p.proj_p50.toFixed(1) : '—'}</td>
        <td>${pct(p.prob_over)}</td>
        <td class="${p.edge >= 0 ? 'edge-pos' : 'edge-neg'}">${signedPct(p.edge)}</td>
        <td>${p.lineup_confidence !== undefined ? p.lineup_confidence.toFixed(1) : '—'}</td>
        <td>${p.side || (p.prob_over >= 0.5 ? 'OVER' : 'UNDER')}</td>
      </tr>
    `).join('');
  } catch (e) {
    body.innerHTML = '<tr><td colspan="8" class="mono">No projections published yet — this fills in once the prediction workflow runs.</td></tr>';
    if (stamp) stamp.textContent = 'Not yet refreshed';
  }
}

// ---- Performance log (performance.html) ----
async function loadPerformance() {
  const body = document.getElementById('perfTableBody');
  const stamp = document.getElementById('perfStamp');
  const summary = document.getElementById('perfSummary');
  if (!body) return;
  try {
    const res = await fetch('data/performance.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('no data yet');
    const data = await res.json();
    if (stamp) stamp.textContent = 'Last reconciled ' + formatStamp(data.generated_at);
    const log = (data.log || []).slice().reverse();

    if (summary && log.length) {
      const graded = log.filter(r => r.grade === 'hit' || r.grade === 'miss');
      const hits = graded.filter(r => r.grade === 'hit').length;
      const rate = graded.length ? (hits / graded.length) : 0;
      summary.innerHTML = `
        <div class="stat"><div class="num mono">${graded.length}</div><div class="label">Settled picks</div></div>
        <div class="stat"><div class="num mono">${pct(rate)}</div><div class="label">Hit rate</div></div>
        <div class="stat"><div class="num mono">${log.length}</div><div class="label">Total logged</div></div>
      `;
    }

    if (log.length === 0) {
      body.innerHTML = '<tr><td colspan="8" class="mono">No results reconciled yet.</td></tr>';
      return;
    }
    body.innerHTML = log.map(r => `
      <tr>
        <td>${r.date}</td>
        <td class="pitcher">${r.pitcher}</td>
        <td>${r.opponent}</td>
        <td>${r.line ?? '—'}</td>
        <td>${r.proj_p50 !== undefined ? r.proj_p50.toFixed(1) : '—'}</td>
        <td class="${r.edge >= 0 ? 'edge-pos' : 'edge-neg'}">${signedPct(r.edge)}</td>
        <td>${r.actual_k ?? '—'}</td>
        <td><span class="badge ${r.grade}">${r.grade_label || r.grade}</span></td>
      </tr>
    `).join('');
  } catch (e) {
    body.innerHTML = '<tr><td colspan="8" class="mono">No performance log yet — this fills in after the first reconciliation run.</td></tr>';
    if (stamp) stamp.textContent = 'Not yet reconciled';
  }
}
