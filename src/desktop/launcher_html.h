#pragma once
// The launcher Zorlo Desktop shows before the model is loaded: choose a model, download progress, loading,
// errors. It talks to the app with chrome.webview.postMessage; the app forwards zorlo start --gui's JSON lines.
// (Each raw string piece stays under MSVC's 16 KB literal limit.)

static const char * LAUNCHER_HTML_1 = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Zorlo</title>
<style>
:root{--bg:#F5F3EE;--surface:#FDFCFA;--s2:#F1EEE8;--s3:#E8E4DC;--ink:#151412;--ink2:#3B3833;--muted:#6B675F;--faint:#9C978E;
--line:rgba(21,20,18,.08);--line2:rgba(21,20,18,.14);--sig:#E5480F;--sigi:#B23C0C;--sigs:rgba(229,72,15,.10);--ok:#2F7A45;--err:#B4291D;
--sh1:0 1px 1px rgba(28,24,18,.04),0 1px 3px rgba(28,24,18,.05);--sh2:0 1px 2px rgba(28,24,18,.05),0 10px 28px -10px rgba(28,24,18,.16);
--eo:cubic-bezier(.23,1,.32,1)}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased;user-select:none}
body{display:flex;align-items:center;justify-content:center;padding:32px}
main{width:min(560px,100%)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:22px;letter-spacing:-.04em;margin-bottom:26px}
.mark{width:18px;height:18px;border-radius:4px;background:var(--ink);position:relative}
.mark::after{content:"";position:absolute;right:-4px;bottom:-4px;width:9px;height:9px;background:var(--sig);border-radius:2px;box-shadow:0 0 0 2px var(--bg)}
h1{font-size:26px;font-weight:500;letter-spacing:-.03em;line-height:1.15;margin-bottom:6px}
.sub{color:var(--muted);margin-bottom:22px}
.view{display:none}.view.on{display:block;animation:in .22s var(--eo)}
@keyframes in{from{opacity:0;transform:translateY(4px)}}
.models{display:grid;gap:10px}
.model{display:grid;grid-template-columns:1fr auto;gap:4px 16px;align-items:center;text-align:left;padding:14px 16px;border:0;border-radius:14px;background:var(--surface);box-shadow:var(--sh1),0 0 0 1px var(--line);font:inherit;color:inherit;cursor:pointer;transition:box-shadow .2s ease,transform .16s var(--eo)}
.model:active{transform:scale(.985)}
.model.def{box-shadow:var(--sh2),0 0 0 1.5px var(--ink)}
.model b{font-weight:600;font-size:15px}
.model p{grid-column:1;color:var(--muted);font-size:13px}
.model .meta{grid-column:1;font:12px/1.4 "Cascadia Mono",Consolas,monospace;color:var(--faint)}
.model .go{grid-row:1/5;grid-column:2;font-size:13px;font-weight:500;padding:7px 12px;border-radius:9px;background:var(--ink);color:var(--surface)}
.model .go.dl{background:var(--sig);color:#fff}
.tag{font:500 10.5px/1 "Cascadia Mono",Consolas,monospace;text-transform:uppercase;letter-spacing:.05em;padding:3px 6px;border-radius:5px;margin-left:8px;vertical-align:2px}
.tag.ok{color:var(--ok);background:#E5F0E7}.tag.rec{color:var(--sigi);background:var(--sigs)}
.newer{margin-top:14px;font-size:12.5px;color:var(--faint)}
.model .have{grid-column:1;font-size:12.5px;color:var(--sigi);text-decoration:underline;text-underline-offset:3px;width:max-content;margin-top:4px}
#locate-msg{margin-top:12px}#locate-msg:empty{display:none}
.bar{height:8px;border-radius:999px;background:var(--s3);overflow:hidden;margin:18px 0 10px}
.bar i{display:block;height:100%;width:0;border-radius:999px;background:var(--sig);transition:width .3s ease}
.bar.busy i{width:35%;animation:slide 1.2s cubic-bezier(.77,0,.175,1) infinite}
@keyframes slide{from{transform:translateX(-100%)}to{transform:translateX(290%)}}
.nums{display:flex;justify-content:space-between;gap:12px;font:12.5px/1.4 "Cascadia Mono",Consolas,monospace;color:var(--muted)}
.hint{color:var(--muted);font-size:13px;margin-top:18px;line-height:1.55}
.spin{width:18px;height:18px;border-radius:50%;border:2px solid var(--sigs);border-top-color:var(--sig);animation:r .7s linear infinite;display:inline-block;vertical-align:-3px;margin-right:10px}
@keyframes r{to{transform:rotate(360deg)}}
.row{display:flex;gap:8px;margin-top:20px;flex-wrap:wrap}
button.btn{height:36px;padding:0 15px;border:0;border-radius:9px;background:var(--ink);color:var(--surface);font:500 13.5px/1 inherit;font-family:inherit;cursor:pointer;transition:background-color .15s ease,transform .16s var(--eo)}
button.btn:active{transform:scale(.97)}
button.ghost{background:var(--surface);color:var(--ink);box-shadow:var(--sh1),0 0 0 1px var(--line2)}
.err{color:var(--err);background:#FBEAE7;border-radius:12px;padding:12px 14px;font-size:14px;line-height:1.5}
.notes{margin-top:16px;display:grid;gap:6px}
.notes p{font-size:12.5px;color:var(--muted);background:var(--surface);border-radius:9px;padding:8px 11px;box-shadow:0 0 0 1px var(--line)}
@media (hover:hover){.model:hover{box-shadow:var(--sh2),0 0 0 1px var(--line2)}.model.def:hover{box-shadow:var(--sh2),0 0 0 1.5px var(--ink)}button.btn:hover{background:#000}button.ghost:hover{background:var(--s2)}}
@media (prefers-reduced-motion:reduce){.view.on{animation:none}.bar.busy i{animation:none;width:100%;opacity:.4}}
</style></head><body><main>
<div class="brand"><span class="mark"></span>zorlo</div>

<section class="view on" id="v-wait"><h1><span class="spin"></span>Getting ready</h1><p class="sub">Checking which AI models you have.</p></section>

<section class="view" id="v-choose"><h1 id="choose-title">Choose an AI model</h1>
<p class="sub" id="choose-sub">It runs on your own PC. You can switch later from the Zorlo icon next to the clock.</p>
<div class="models" id="models"></div><p class="err" id="locate-msg"></p><p class="newer" id="newer"></p></section>

<section class="view" id="v-download"><h1 id="dl-title">Downloading</h1>
<p class="sub">Only the first time. If it stops, open Zorlo again and it continues where it left off.</p>
<div class="bar" id="dl-bar"><i id="dl-fill"></i></div>
<div class="nums"><span id="dl-left">Starting…</span><span id="dl-rate"></span></div>
<p class="hint">You can close this window: the download carries on in the background (Zorlo icon next to the clock).</p>
<div class="row"><button class="btn ghost" data-msg="quit">Cancel and quit</button></div></section>

<section class="view" id="v-load"><h1><span class="spin"></span><span id="load-title">Starting</span></h1>
<p class="sub" id="load-sub">Loading the model into memory. This takes about a minute.</p>
<div class="bar busy"><i></i></div></section>

<section class="view" id="v-error"><h1>Something went wrong</h1><p class="err" id="err-text"></p>
<div class="row"><button class="btn" data-msg="retry">Try again</button><button class="btn ghost" data-msg="menu">Choose another model</button></div></section>

<div class="notes" id="notes"></div>
</main>
)HTML";

static const char * LAUNCHER_HTML_2 = R"HTML(<script>
(() => {
  const $ = (s) => document.querySelector(s);
  const send = (o) => window.chrome && chrome.webview && chrome.webview.postMessage(o);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const gb = (b) => (b / 1e9).toFixed(b >= 1e10 ? 1 : 2) + ' GB';
  const show = (id) => document.querySelectorAll('.view').forEach((v) => v.classList.toggle('on', v.id === id));
  let names = {}, current = '';
  document.addEventListener('click', (e) => { const b = e.target.closest('[data-msg]'); if (b) send({ type: b.dataset.msg }); });
  function models(m) {
    names = {}; m.items.forEach((x) => (names[x.id] = x.name));
    const any = m.items.some((x) => x.installed);
    $('#choose-title').textContent = any ? 'Which model?' : 'Choose an AI model to download';
    $('#models').innerHTML = m.items.map((x) => `<button class="model${x.default ? ' def' : ''}" data-id="${esc(x.id)}">` +
      `<b>${esc(x.name)}${x.ready ? '<span class="tag ok">ready</span>' : x.recommended ? '<span class="tag rec">recommended</span>' : ''}</b>` +
      (x.about ? `<p>${esc(x.about)}</p>` : '') +
      `<span class="meta">${x.ready ? 'downloaded' : (x.installed ? 'download not finished · ' : '') + x.gb.toFixed(1) + ' GB download'}${x.ram ? ' · needs ' + x.ram + ' GB memory' : ''}</span>` +
      `<span class="go${x.ready ? '' : ' dl'}">${x.ready ? 'Start' : 'Download'}</span>` +
      (x.ready ? '' : `<span class="have" data-locate="${esc(x.id)}" role="button" tabindex="0">Already have this model? Choose its folder</span>`) + `</button>`).join('');
    document.querySelectorAll('.model').forEach((b) => (b.onclick = (e) => {
      const loc = e.target.closest('[data-locate]');
      if (loc) { e.stopPropagation(); $('#locate-msg').textContent = ''; send({ type: 'locate', id: loc.dataset.locate }); return; }
      current = b.dataset.id; send({ type: 'choose', id: current }); show('v-wait');
    }));
    $('#newer').textContent = m.too_new && m.too_new.length ? 'Needs a newer Zorlo: ' + m.too_new.join(', ') + '. Download the latest version from the website.' : '';
    show('v-choose');
  }
  let t0 = 0, d0 = 0;
  function progress(p) {
    show('v-download');
    const total = p.all_total || p.total, done = p.all_total ? p.all_done : p.done;
    $('#dl-fill').style.width = (total ? Math.min(100, done / total * 100) : 0).toFixed(1) + '%';
    if (!t0) { t0 = performance.now(); d0 = done; }
    const secs = (performance.now() - t0) / 1000, rate = secs > 3 ? (done - d0) / secs : p.rate;
    $('#dl-left').textContent = `${gb(done)} of ${gb(total)}`;
    if (rate > 0) { const left = (total - done) / rate, h = Math.floor(left / 3600), m = Math.round(left % 3600 / 60);
      $('#dl-rate').textContent = `${(rate / 1e6).toFixed(1)} MB/s · about ${h ? h + ' h ' : ''}${m} min left`; }
  }
  function onMsg(m) {
    switch (m.type) {
      case 'models': models(m); break;
      case 'auto': current = m.id; show('v-wait'); break;
      case 'status': current = m.id || current; if (m.downloading) { $('#dl-title').textContent = m.text; show('v-download'); } else { $('#load-title').textContent = m.text; $('#load-sub').textContent = 'This only takes a moment.'; show('v-load'); } break;
      case 'download': show('v-download'); break;
      case 'progress': progress(m); break;
      case 'starting': $('#load-title').textContent = 'Starting ' + (m.name || names[m.id] || 'Zorlo'); $('#load-sub').textContent = 'Loading the model into memory. This takes about a minute.'; show('v-load'); break;
      case 'located': if (m.ok) { current = m.id; show('v-wait'); } else $('#locate-msg').textContent = m.text; break;
      case 'note': $('#notes').insertAdjacentHTML('beforeend', `<p>${esc(m.text)}</p>`); break;
      case 'error': $('#err-text').textContent = m.text; show('v-error'); break;
      case 'exited': if (!document.querySelector('#v-error.on')) { $('#err-text').textContent = 'Zorlo stopped unexpectedly. Details are in the log: ' + (m.log || ''); show('v-error'); } break;
    }
  }
  if (window.chrome && chrome.webview) { chrome.webview.addEventListener('message', (e) => onMsg(e.data)); send({ type: 'hello' }); }
})();
</script></body></html>
)HTML";
