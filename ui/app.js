// Zorlo agent app.
// The loop runs here: ask the model (with tools) -> run each tool call through the server (/agent/tool applies the
// user's permissions) -> give the results back -> repeat until it answers. todo_write is handled in the page.
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const TOKEN = $('meta[name="zorlo-token"]').content;
  const CHARS_PER_TOKEN = 3.6, MAX_TOOL_CHARS = 8000, TEXT_ATTACH_MAX = 200000;
  // which model the agent talks to: Zorlo's own (12K context), or another provider from Settings → Model
  const remote = () => { const p = (state.settings && state.settings.provider) || {}; return !!p.kind && p.kind !== 'zorlo'; };
  const ctxTokens = () => remote() ? Math.max(4096, +state.settings.provider.context || 128000) : 12288;
  const ctxChars = () => Math.round(ctxTokens() * CHARS_PER_TOKEN * 0.68);   // room left for the reply
  const READ_ONLY = new Set(['list_dir', 'read_file', 'read_document', 'search_files', 'web_search', 'fetch_url', 'load_skill', 'preview_url', 'todo_write', 'find_connectors', 'find_skills', 'connector_tools']);
  const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|jsonl|xml|ya?ml|toml|ini|cfg|conf|log|html?|css|scss|less|js|mjs|cjs|jsx|ts|tsx|vue|svelte|py|rb|php|java|kt|go|rs|c|h|cc|cpp|hpp|cs|swift|m|sh|bash|zsh|ps1|bat|cmd|sql|r|lua|pl|dart|scala|gradle|env|gitignore|dockerfile)$/i;

  async function api(path, body, raw) {
    const r = await fetch(path, { method: body === undefined ? 'GET' : 'POST',
      headers: raw ? { 'X-Zorlo-Token': TOKEN } : { 'Content-Type': 'application/json', 'X-Zorlo-Token': TOKEN },
      body: body === undefined ? undefined : raw ? body : JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || 'HTTP ' + r.status);
    return j;
  }
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const uid = () => Math.random().toString(36).slice(2, 10);
  const kb = (n) => n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(1) + ' KB' : (n / 1048576).toFixed(1) + ' MB';

  // ---------- state ----------
  let state = { settings: {}, tools: [], skills: [], connectors: [], instructions: null };
  let chats = [], current = null, attachments = [], fileCache = { at: 0, list: [] };
  try { chats = JSON.parse(localStorage.getItem('zorlo.chats') || '[]'); } catch { chats = []; }
  // chats are saved by Zorlo (%APPDATA%\Zorlo\chats), so Zorlo Desktop and every browser show the same list;
  // this browser keeps a copy too, for when Zorlo can't be reached
  const dirty = new Set();
  let flushTimer = null;
  async function flushChats() {
    flushTimer = null;
    for (const id of [...dirty]) {
      dirty.delete(id);
      const c = chats.find((x) => x.id === id);
      if (!c || !c.messages.length) continue;
      try { await api('/agent/chat?kind=agent', c); } catch { dirty.add(id); }
    }
  }
  async function syncChats() {   // merge what Zorlo has with this window's chats: the newer copy of each wins
    let server;
    try { server = (await api('/agent/chats?kind=agent')).chats; } catch { return; }
    const byId = new Map(chats.map((c) => [c.id, c]));
    let changed = false;
    for (const s of server) {
      const mine = byId.get(s.id);
      if (!mine) { chats.push(s); changed = true; }
      else if ((s.updated || 0) > (mine.updated || 0) && !running(mine)) { Object.assign(mine, s); changed = true; }
    }
    const onServer = new Set(server.map((s) => s.id));
    for (const c of chats) if (c.messages.length && !onServer.has(c.id)) dirty.add(c.id);   // older browser chats move over once
    if (dirty.size) flushChats();
    chats.sort((a, b) => (b.updated || 0) - (a.updated || 0));
    for (const c of chats) { c.todos = c.todos || []; c.changes = c.changes || []; c.mode = c.mode || 'agent'; if (!c.folder) c.folder = state.settings.workspace; }
    if (changed) { if (!current) current = chats[0] || null; renderAll(); }
  }
  addEventListener('focus', () => syncChats());
  function save(chat) {
    const c = chat || current;
    if (c) { c.updated = Date.now(); dirty.add(c.id); }
    if (!flushTimer) flushTimer = setTimeout(flushChats, 500);
    for (let keep = 60; keep > 0; keep -= 10) {
      try { localStorage.setItem('zorlo.chats', JSON.stringify(chats.slice(0, keep))); return; } catch {}
    }
  }
  const log = $('#log'), input = $('#input'), statusEl = $('#status'), app = $('#app');
  // each chat runs on its own: switch to another chat (or folder) while one is still working
  const runs = new Map();   // chat id -> { ctrl, live, liveEl, ask, status }
  const runOf = (c) => (c ? runs.get(c.id) : null);
  const running = (c) => !!runOf(c);
  let idleStatus = { text: 'Loading…', busy: false };
  function paintStatus() {
    const r = runOf(current), s = r ? r.status : idleStatus;
    statusEl.textContent = s.text; statusEl.classList.toggle('busy', !!s.busy);
    $('#send').hidden = !!r; $('#stop').hidden = !r;
  }
  function setStatus(t, isBusy, chat) {
    const r = runOf(chat || current);
    if (r) r.status = { text: t, busy: isBusy };
    else idleStatus = { text: t, busy: isBusy };
    if (!chat || chat === current) paintStatus();
  }
  const baseName = (p) => (p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p;

  // ---------- markdown + highlighting (everything escaped first) ----------
  const KW = /\b(function|return|if|else|elif|for|while|do|break|continue|const|let|var|class|def|import|from|export|default|new|try|catch|except|finally|raise|throw|async|await|yield|in|of|is|not|and|or|None|null|undefined|true|false|True|False|this|self|public|private|static|void|int|float|string|bool|struct|enum|fn|pub|use|match|switch|case|lambda|with|as|echo|then|fi|elseif|foreach|param)\b/g;
  function highlight(code, lang) {
    let s = esc(code);
    if (/^(html|xml|svg|vue)$/i.test(lang)) return s.replace(/(&lt;\/?)([\w-]+)/g, '$1<span class="tk-t">$2</span>').replace(/(&quot;[^&]*?&quot;)/g, '<span class="tk-s">$1</span>');
    const keep = [];
    const stash = (cls) => (m) => { keep.push(`<span class="${cls}">${m}</span>`); return `\u0001${keep.length - 1}\u0001`; };
    s = s.replace(/(\/\*[\s\S]*?\*\/|\/\/[^\n]*|(^|\s)#[^\n]*)/g, stash('tk-c'))
      .replace(/(&quot;(?:[^&\\]|\\.|&(?!quot;))*?&quot;|&#39;(?:[^&\\]|\\.|&(?!#39;))*?&#39;|`[^`]*`)/g, stash('tk-s'))
      .replace(/\b\d+(\.\d+)?\b/g, stash('tk-n'))
      .replace(KW, stash('tk-k'));
    return s.replace(/\u0001(\d+)\u0001/g, (_, i) => keep[+i]);
  }
  function md(src) {
    const blocks = [];
    let s = String(src || '').replace(/```([\w+-]*)[^\n]*\n?([\s\S]*?)(```|$)/g, (_, lang, code) => { blocks.push({ lang, code: code.replace(/\n$/, '') }); return `\u0000${blocks.length - 1}\u0000`; });
    s = esc(s);
    const inline = (t) => t
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<i>$2</i>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    const lines = s.split('\n'), out = [];
    let list = null;
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (let i = 0; i < lines.length; ++i) {
      const line = lines[i];
      let m;
      if (/^\u0000\d+\u0000$/.test(line.trim())) { closeList(); out.push(line.trim()); continue; }
      if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || '')) {     // table
        closeList();
        const cells = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim()));
        let t = '<table><tr>' + cells(line).map((c) => `<th>${c}</th>`).join('') + '</tr>';
        i += 2;
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) t += '<tr>' + cells(lines[i++]).map((c) => `<td>${c}</td>`).join('') + '</tr>';
        --i;
        out.push(t + '</table>');
        continue;
      }
      if ((m = line.match(/^(#{1,4})\s+(.*)/))) { closeList(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); continue; }
      if ((m = line.match(/^&gt;\s?(.*)/))) { closeList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
      if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; } out.push(`<li>${inline(m[1])}</li>`); continue; }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; } out.push(`<li>${inline(m[1])}</li>`); continue; }
      closeList();
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    closeList();
    return out.join('').replace(/\u0000(\d+)\u0000/g, (_, i) => {
      const b = blocks[+i];
      return `<pre>${b.lang ? `<span class="lang">${esc(b.lang)}</span>` : ''}<button class="copy" type="button">Copy</button><code>${highlight(b.code, b.lang)}</code></pre>`;
    });
  }

  // ---------- chats ----------
  // a chat belongs to one project folder; the folder follows the chat you open
  function newChat(folder) {
    folder = folder || (current && current.folder) || state.settings.workspace || '';
    if (current && !current.messages.length && !running(current) && current.folder === folder) { input.focus(); return; }  // reuse an empty one
    if (current && !current.messages.length && !running(current)) chats = chats.filter((x) => x !== current);
    current = { id: uid(), title: 'New chat', mode: 'agent', folder, messages: [], todos: [], changes: [], preview: '', allowAll: false, updated: Date.now() };
    chats.unshift(current);
    attachments = [];
    useFolder(folder);
    renderAll(); input.focus();
  }
  function openChat(c) {
    current = c; attachments = [];
    useFolder(c.folder);
    renderAll(); $('#side').classList.remove('open');
  }
  // the server's project folder (files panel, @ mentions, preview, uploads) follows the open chat
  async function useFolder(folder) {
    if (!folder) return;
    $('#folder-path').textContent = folder;
    $('#folder').title = 'Project folder of this chat: ' + folder + '. Click to start a chat in another folder.';
    if (state.settings.workspace === folder) return;
    state.settings.workspace = folder;
    fileCache.at = 0;
    try { await api('/agent/settings', { workspace: folder }); const s = await api('/agent/state'); state.instructions = s.instructions; } catch (e) { setStatus('folder: ' + e.message, false); }
  }
  function renderSide() {
    const q = $('#chat-search').value.trim().toLowerCase();
    const nav = $('#chats');
    nav.innerHTML = '';
    const groups = new Map();   // folder -> chats, the most recently used folder first
    for (const c of chats) {
      if (q && !c.title.toLowerCase().includes(q) && !(c.folder || '').toLowerCase().includes(q) && !c.messages.some((m) => m.role === 'user' && (m.text || '').toLowerCase().includes(q))) continue;
      const k = c.folder || '';
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(c);
    }
    if (current && !groups.has(current.folder || '') && !q) groups.set(current.folder || '', [current]);
    for (const [folder, list] of groups) {
      const g = document.createElement('div');
      g.className = 'group' + (current && current.folder === folder ? ' here' : '');
      g.innerHTML = `<div class="group__head" title="${esc(folder)}">${ICON.folder}<span>${esc(baseName(folder) || 'No folder')}</span>` +
        `<button class="group__new" type="button" title="New chat in ${esc(folder)}" aria-label="New chat in this folder">${ICON.add}</button></div>`;
      $('.group__new', g).onclick = () => newChat(folder);
      for (const c of list) {
        const r = runOf(c);
        const b = document.createElement('button');
        b.className = 'chat-item' + (c === current ? ' on' : '');
        b.type = 'button';
        b.innerHTML = `<span>${esc(c.title)}</span>${r ? `<i class="live${r.ask ? ' ask' : ''}" title="${r.ask ? 'Waiting for your OK' : 'Working'}"></i>` : ''}<i class="del" title="Delete chat">&#10005;</i>`;
        b.onclick = (e) => {
          if (e.target.classList.contains('del')) {
            if (running(c)) { setStatus('stop that chat first', false); return; }
            if (!confirm('Delete this chat? Files the agent made stay in your project folder.')) return;
            chats = chats.filter((x) => x !== c);
            dirty.delete(c.id);
            api('/agent/chat_delete?kind=agent', { id: c.id }).catch(() => {});
            save();
            if (c !== current) return renderSide();
            const next = chats.find((x) => x.folder === c.folder) || chats[0];
            return next ? openChat(next) : (current = null, newChat(c.folder));
          }
          openChat(c);
        };
        g.appendChild(b);
      }
      nav.appendChild(g);
    }
  }

  // ---------- messages ----------
  const svg = (p) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;
  const ICON = {
    folder: svg('<path d="M3.5 7.5a2 2 0 0 1 2-2h3.8l2 2h7.2a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>'),
    file: svg('<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4"/>'),
    plus: svg('<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M12 11v6M9 14h6"/>'),
    edit: svg('<path d="M4.5 19.5l1-4 10-10 3 3-10 10z"/><path d="M13.5 7.5l3 3"/>'),
    term: svg('<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M7.5 9.5l3 2.5-3 2.5M12.5 15h4"/>'),
    search: svg('<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>'),
    globe: svg('<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.4 2.5 3.5 5.3 3.5 8.5s-1.1 6-3.5 8.5c-2.4-2.5-3.5-5.3-3.5-8.5s1.1-6 3.5-8.5z"/>'),
    eye: svg('<path d="M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>'),
    book: svg('<path d="M4.5 5.5a2 2 0 0 1 2-2h12v15h-12a2 2 0 0 0-2 2z"/><path d="M4.5 20.5a2 2 0 0 1 2-2h12v2"/>'),
    list: svg('<path d="M9 6.5h10.5M9 12h10.5M9 17.5h10.5"/><circle cx="5" cy="6.5" r="1"/><circle cx="5" cy="12" r="1"/><circle cx="5" cy="17.5" r="1"/>'),
    plug: svg('<path d="M9 3.5v4M15 3.5v4M6.5 7.5h11v3a5.5 5.5 0 0 1-11 0zM12 16v4.5"/>'),
    image: svg('<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="M20.5 16l-5-5-9 8.5"/>'),
    box: svg('<path d="M3.5 7.5l8.5-4 8.5 4v9l-8.5 4-8.5-4z"/><path d="M3.5 7.5l8.5 4 8.5-4M12 11.5v9"/>'),
    x: svg('<path d="M7 7l10 10M17 7L7 17"/>'),
    add: svg('<path d="M12 5.5v13M5.5 12h13"/>'),
    skill: svg('<path d="M12 3.5l2.4 5 5.6.6-4.2 3.8 1.2 5.5L12 15.6l-5 2.8 1.2-5.5L4 9.1l5.6-.6z"/>'),
  };
  const SETUP_TOOLS = new Set(['add_connector', 'remove_connector', 'install_skill']);
  const TOOL_ICON = { list_dir: ICON.folder, read_file: ICON.file, read_document: ICON.file, search_files: ICON.search, write_file: ICON.plus, edit_file: ICON.edit, run_command: ICON.term, web_search: ICON.globe, fetch_url: ICON.globe, preview_url: ICON.eye, load_skill: ICON.book, todo_write: ICON.list,
    find_connectors: ICON.plug, add_connector: ICON.plug, use_connector: ICON.plug, connector_tools: ICON.plug, remove_connector: ICON.plug, find_skills: ICON.skill, install_skill: ICON.skill };
  const parseArgs = (call) => { try { return JSON.parse(call.function.arguments || '{}'); } catch { return {}; } };
  function diffHtml(oldT, newT) {
    const o = String(oldT || '').split('\n'), n = String(newT || '').split('\n');
    return '<div class="diff">' + o.map((l) => `<div class="del">- ${esc(l)}</div>`).join('') + n.map((l) => `<div class="add">+ ${esc(l)}</div>`).join('') + '</div>';
  }
  function toolCard(call, res) {
    const el = document.createElement('div');
    const name = call.function.name, args = parseArgs(call);
    const st = !res ? 'run' : res.undone ? 'ok undone' : res.status === 'ok' ? 'ok' : (res.status === 'blocked' || res.status === 'declined') ? 'blocked' : 'err';
    const label = !res ? 'running' : res.undone ? 'undone' : { ok: 'done', blocked: 'blocked', declined: 'skipped', error: 'error' }[res.status] || res.status;
    el.className = 'tool ' + st;
    el.dataset.call = call.id;
    let sum = res && res.summary ? res.summary : (args.command || args.path || args.query || args.url || '');
    if (name === 'todo_write' && Array.isArray(args.todos)) sum = `${args.todos.filter((t) => t.status === 'completed').length}/${args.todos.length} tasks done`;
    el.innerHTML = `<div class="tool__head"><span class="tool__icon">${TOOL_ICON[name] || ICON.plug}</span><span class="tool__name">${esc(name.replace(/^mcp__/, '').replace(/__/g, ' · '))}</span>` +
      `<span class="tool__sum">${esc(sum)}</span><span class="tool__state"><i class="dot"></i>${esc(label)}</span></div>`;
    const r = res && res.result;
    const links = [];
    if (r && r.preview) links.push(`<button class="btn btn--ghost btn--sm" data-preview="${esc(r.preview)}" type="button">Show preview</button><a class="btn btn--ghost btn--sm" href="${esc(r.preview)}" target="_blank" rel="noopener">Open in browser &#8599;</a>`);
    if (r && r.url && name === 'preview_url') links.push(`<button class="btn btn--ghost btn--sm" data-preview="${esc(r.url)}" type="button">Show preview</button><a class="btn btn--ghost btn--sm" href="${esc(r.url)}" target="_blank" rel="noopener">Open in browser &#8599;</a>`);
    if (r && r.undo_id && !res.undone) links.push(`<button class="btn btn--ghost btn--sm" data-undo="${esc(r.undo_id)}" type="button">Undo</button>`);
    if (links.length) el.insertAdjacentHTML('beforeend', `<div class="tool__links">${links.join('')}</div>`);
    if (name === 'edit_file' && args.old_text !== undefined) el.insertAdjacentHTML('beforeend', `<details${res ? '' : ' open'}><summary>Change</summary>${diffHtml(args.old_text, args.new_text)}</details>`);
    if (name === 'write_file' && args.content) el.insertAdjacentHTML('beforeend', `<details><summary>File content (${args.content.split('\n').length} lines)</summary><pre>${highlight(args.content.slice(0, 30000), (args.path || '').split('.').pop())}</pre></details>`);
    if (name === 'todo_write' && Array.isArray(args.todos)) el.insertAdjacentHTML('beforeend', `<div class="todo-mini">${args.todos.map((t) => `${t.status === 'completed' ? '&#9745;' : t.status === 'in_progress' ? '&#9654;' : '&#9744;'} ${esc(t.content)}`).join('<br>')}</div>`);
    else if (r !== undefined && name !== 'todo_write') {
      const text = typeof r === 'string' ? r : name === 'run_command' && r ? `exit code ${r.exit_code}${r.timed_out ? ' (stopped at the time limit)' : ''}\n\n${r.output || ''}` : JSON.stringify(r, null, 2);
      el.insertAdjacentHTML('beforeend', `<details${name === 'run_command' && res.status === 'ok' ? ' open' : ''}><summary>${name === 'run_command' ? 'Output' : 'Details'}</summary><pre>${esc(text.slice(0, 20000))}</pre></details>`);
    }
    return el;
  }
  function userText(m) {
    let t = m.text || '';
    for (const a of m.attachments || []) {
      if (a.kind === 'text') t += `\n\n<file name="${a.name}">\n${a.text}\n</file>`;
      else {
        t += `\n\n[Attached file saved in the project folder: ${a.path} (${kb(a.size)}${a.info ? ', ' + a.info : ''})${a.image ? '. It is a picture: you cannot see it; any words in it are below' : ''}${a.note ? '. ' + a.note : ''}]`;
        if (a.text) t += `\n<file name="${a.name}">\n${a.text}${a.cut ? `\n[... only the first 40000 of ${a.cut} characters; read_document with offset reads the rest]` : ''}\n</file>`;
      }
    }
    return t;
  }
  function msgEl(m, idx, live) {
    const d = document.createElement('div');
    if (m.role === 'user') {
      if (m.hidden) { d.className = 'note'; d.textContent = 'Zorlo asked the model to fix its last tool call (it was not valid JSON).'; return d; }
      d.className = 'msg msg--user';
      d.dataset.idx = idx;
      const atts = m.attachments || [];
      d.innerHTML = (atts.length ? `<div class="atts">${atts.map((a) => attHtml(a)).join('')}</div>` : '') +
        (m.text ? `<div class="bubble">${esc(m.text)}</div>` : '') +
        `<div class="actions"><button type="button" data-act="edit">Edit</button><button type="button" data-act="copy">Copy</button></div>`;
      return d;
    }
    if (m.role === 'assistant') {
      d.className = 'msg msg--ai';
      d.dataset.idx = idx;
      if (m.reasoning) d.insertAdjacentHTML('beforeend', `<details class="think"${live ? ' open' : ''}><summary>Thinking</summary><div>${esc(m.reasoning)}</div></details>`);
      if (m.content || live) d.insertAdjacentHTML('beforeend', `<div class="text${live ? ' cursor' : ''}">${md(m.content)}</div>`);
      if (!live && (m.content || m.stats)) {
        d.insertAdjacentHTML('beforeend', `<div class="actions">${m.content ? '<button type="button" data-act="copy">Copy</button>' : ''}<button type="button" data-act="regen">Regenerate</button>` +
          (m.stats ? `<span class="stats">${m.stats.tps} tok/s · ${m.stats.secs} s</span>` : '') + '</div>');
        if (m.plan) d.insertAdjacentHTML('beforeend', '<div class="plan-go"><button class="btn btn--sig btn--sm" data-act="run-plan" type="button">Run this plan</button><button class="btn btn--ghost btn--sm" data-act="revise" type="button">Change the plan</button></div>');
      }
      return d;
    }
    if (m.role === 'error') { d.className = 'err-box'; d.textContent = m.content; return d; }
    if (m.role === 'note') { d.className = 'note'; d.textContent = m.content; return d; }
    return null;
  }
  // keep: stay where the user scrolled to (a background update), unless they were at the bottom
  function renderLog(keep) {
    const stick = !keep || atBottom(), top = log.scrollTop;
    log.innerHTML = '';
    const r = runOf(current);
    if (!current || (!current.messages.length && !r)) {
      log.appendChild($('#empty-tpl').content.cloneNode(true));
      $$('[data-idea]', log).forEach((b) => (b.onclick = () => { input.value = b.dataset.idea; autosize(); input.focus(); }));
      return;
    }
    const results = {};
    current.messages.forEach((m) => { if (m.role === 'tool') results[m.tool_call_id] = m.ui; });
    current.messages.forEach((m, i) => {
      const el = msgEl(m, i);
      if (el) log.appendChild(el);
      if (m.role === 'assistant' && m.tool_calls) for (const c of m.tool_calls) log.appendChild(toolCard(c, results[c.id] || (r ? null : { status: 'error', result: 'not run' })));
    });
    // only the last answer can be regenerated
    const ai = $$('.msg--ai', log);
    ai.forEach((e, i) => { if (i !== ai.length - 1 || r) $$('[data-act="regen"],.plan-go', e).forEach((b) => b.remove()); });
    if (r && r.live) { r.liveEl = msgEl(r.live, -1, true); log.appendChild(r.liveEl); }
    if (r && r.ask) attachAsk(r);
    log.scrollTop = stick ? log.scrollHeight : top;
  }
  function renderAll() { renderSide(); renderLog(); renderMode(); renderPanel(); renderCtx(); renderAttachments(); paintStatus(); }
  const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;

  log.addEventListener('click', async (e) => {
    const t = e.target;
    if (t.closest('.copy')) { const b = t.closest('.copy'); navigator.clipboard.writeText(b.parentElement.querySelector('code').textContent).then(() => { b.textContent = 'Copied'; setTimeout(() => (b.textContent = 'Copy'), 1200); }); return; }
    if (t.dataset.preview) { showPreview(t.dataset.preview, true); return; }
    if (t.dataset.undo) { await undo(t.dataset.undo); return; }
    const act = t.dataset.act;
    if (!act || running(current)) return;
    const box = t.closest('[data-idx]'), idx = box ? +box.dataset.idx : -1, m = current.messages[idx];
    if (act === 'copy') { navigator.clipboard.writeText(m.role === 'user' ? m.text : m.content); t.textContent = 'Copied'; setTimeout(() => (t.textContent = 'Copy'), 1200); }
    if (act === 'edit') editMessage(box, idx);
    if (act === 'regen') regenerate();
    if (act === 'run-plan') { setMode('agent'); send('Go ahead and carry out this plan.'); }
    if (act === 'revise') { input.value = 'Change the plan: '; autosize(); input.focus(); }
  });
  function editMessage(box, idx) {
    const m = current.messages[idx];
    box.innerHTML = `<div class="edit-box"><textarea>${esc(m.text)}</textarea><div class="row"><button class="btn btn--ghost btn--sm" type="button" data-e="cancel">Cancel</button><button class="btn btn--sm" type="button" data-e="save">Save and resend</button></div></div>`;
    const ta = $('textarea', box);
    ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
    $('[data-e="cancel"]', box).onclick = renderLog;
    $('[data-e="save"]', box).onclick = () => {
      const text = ta.value.trim();
      if (!text) return;
      const att = m.attachments || [];
      current.messages = current.messages.slice(0, idx);   // everything after the edited message is replaced
      send(text, att);
    };
    ta.onkeydown = (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('[data-e="save"]', box).click(); if (e.key === 'Escape') renderLog(); };
  }
  function regenerate() {
    let i = current.messages.length - 1;
    while (i >= 0 && !(current.messages[i].role === 'user' && !current.messages[i].hidden)) --i;
    if (i < 0) return;
    current.messages = current.messages.slice(0, i + 1);
    run(current);
  }

  // ---------- prompt ----------
  const TODO_TOOL = { type: 'function', function: { name: 'todo_write', description: 'Keep a short to-do list for tasks with 3 or more steps. Send the whole list each time; exactly one item in_progress while working.',
    parameters: { type: 'object', properties: { todos: { type: 'array', items: { type: 'object', properties: { content: { type: 'string' }, status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] } }, required: ['content', 'status'] } } }, required: ['todos'] } } };
  function toolsFor(mode) {
    if (mode === 'chat') return undefined;
    const all = [...state.tools, TODO_TOOL];
    return mode === 'plan' ? all.filter((t) => READ_ONLY.has(t.function.name)) : all;
  }
  function systemPrompt(chat, mode) {
    let p = `You are Zorlo, an AI assistant that runs entirely on the user's own Windows PC. Today is ${new Date().toDateString()}.`;
    if (mode === 'chat') return p + ' Answer clearly and concisely. Use Markdown.';
    p += `\nProject folder: ${chat.folder || state.settings.workspace}`;
    if (mode === 'plan')
      p += '\nPLAN MODE: investigate with the read-only tools (list_dir, read_file, search_files, web_search, fetch_url), then reply with a short numbered plan: what you will change, in which files, and how you will check it works. Do not change anything yet; the user will press "Run this plan".';
    else
      p += '\nUse your tools to do the work instead of explaining how. Work step by step:' +
        '\n- Look before you change: list_dir (depth 2 for an overview), search_files, read_file. Small changes: edit_file. New files: write_file.' +
        '\n- For tasks with 3+ steps, keep a to-do list with todo_write and update it as you go.' +
        '\n- run_command uses Windows PowerShell in the project folder. Don\'t start servers or programs that keep running.' +
        '\n- Websites: plain HTML/CSS/JS files; the user sees them live in the Preview panel. Keep each file reasonably short; write CSS and JS in separate files.' +
        '\n- Tool arguments must be valid JSON: escape quotes and newlines inside strings.' +
        '\n- If a tool fails, read the error and fix the cause. If the user declines a tool, don\'t retry it.' +
        '\n- PDF, Word, Excel, PowerPoint, pictures (words in them, by OCR) and video: read_document. You cannot see pictures; say so if the user asks what one shows. To create Word, Excel, PowerPoint or PDF files, install the docx / xlsx / pptx / pdf skill (find_skills) and follow it.' +
        '\n- Need a tool you don\'t have (GitHub, Notion, a real browser, a database...)? find_connectors, then add_connector. For tasks a skill covers (Word, PDF, PowerPoint, design, testing...), find_skills and install_skill. Ask the user for any key or token; never make one up.' +
        '\n- Finish with one or two sentences saying what you did and where.';
    const run = runOf(chat), ins = run && run.instructions !== undefined ? run.instructions : state.instructions;
    if (ins) p += `\n\nProject instructions from ${ins.file} (follow them):\n${ins.text}`;
    const conns = (state.connectors || []).filter((c) => c.ok && c.tool_names && c.tool_names.length);
    if (conns.length) p += '\n\nConnectors and their tools. These are not your tools directly: call one with use_connector {"connector": "...", "tool": "...", "arguments": {...}}; connector_tools shows the arguments a tool takes.\n' + conns.map((c) => `- ${c.name}: ${c.tool_names.join(', ')}`).join('\n');
    if (state.skills.length) p += '\n\nSkills (load one with load_skill when a task matches):\n' + state.skills.map((k) => `- ${k.name}: ${k.description}`).join('\n');
    return p;
  }
  function wireMessages(chat, mode) {
    const msgs = chat.messages.filter((m) => ['user', 'assistant', 'tool'].includes(m.role)).map((m) => {
      if (m.role === 'tool') return { role: 'tool', tool_call_id: m.tool_call_id, content: m.content };
      if (m.role === 'assistant') { const o = { role: 'assistant', content: m.content || '' }; if (m.tool_calls) o.tool_calls = m.tool_calls; return o; }
      return { role: 'user', content: userText(m) };
    });
    const size = () => msgs.reduce((n, m) => n + (m.content || '').length + JSON.stringify(m.tool_calls || '').length, 0);
    for (let i = 0; i < msgs.length - 4 && size() > ctxChars(); ++i)
      if (msgs[i].role === 'tool' && msgs[i].content.length > 400) msgs[i].content = msgs[i].content.slice(0, 300) + '\n[... shortened to save space]';
    while (size() > ctxChars() && msgs.length > 2) {   // drop the oldest whole turn
      let j = 1;
      while (j < msgs.length - 1 && msgs[j].role !== 'user') ++j;
      if (j >= msgs.length - 1) break;
      msgs.splice(0, j);
    }
    return [{ role: 'system', content: systemPrompt(chat, mode) }, ...msgs];
  }
  function renderCtx() {
    if (!current) return;
    const mode = current.mode || 'agent';
    const chars = JSON.stringify(wireMessages(current, mode)).length + (mode === 'chat' ? 0 : JSON.stringify(toolsFor(mode) || []).length);
    const pct = Math.min(100, Math.round(chars / CHARS_PER_TOKEN / ctxTokens() * 100));
    $('#ctx').style.setProperty('--w', pct + '%');
    $('#ctx-bar').style.setProperty('--w', pct + '%');
    $('#ctx-text').textContent = `${pct}% memory`;
    $('#ctx').classList.toggle('warn', pct > 80);
  }

  // ---------- one model call (streamed) ----------
  async function callModel(chat, mode) {
    const run = runOf(chat);
    const body = { model: 'zorlo', stream: true, messages: wireMessages(chat, mode), mode: $('#speed').value, chat_template_kwargs: { enable_thinking: $('#thinking').checked } };
    const tools = toolsFor(mode);
    if (tools) body.tools = tools;
    // another provider: through Zorlo, which adds the API key (the page never has it)
    const r = await fetch(remote() ? '/agent/llm' : '/v1/chat/completions', { method: 'POST', signal: run.ctrl.signal,
      headers: { 'Content-Type': 'application/json', 'X-Zorlo-Token': TOKEN }, body: JSON.stringify(body) });
    if (!r.ok) { const t = await r.text(); let e = t; try { e = JSON.parse(t).error.message; } catch {} throw new Error(e || 'HTTP ' + r.status); }
    const msg = { role: 'assistant', content: '', reasoning: '', tool_calls: null };
    const calls = [], reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '', tokens = 0, first = 0;
    const t0 = performance.now();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, k).trim();
        buf = buf.slice(k + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        let j;
        try { j = JSON.parse(data); } catch { continue; }
        if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
        const d = j.choices && j.choices[0] && j.choices[0].delta;
        if (!d) continue;
        if (!first) first = performance.now();
        ++tokens;
        if (d.reasoning_content) msg.reasoning += d.reasoning_content;
        if (d.content) msg.content += d.content;
        if (d.tool_calls) for (const tc of d.tool_calls) {
          const c = calls[tc.index || 0] || (calls[tc.index || 0] = { id: '', type: 'function', function: { name: '', arguments: '' } });
          if (tc.id) c.id = tc.id;
          if (tc.function && tc.function.name) c.function.name += tc.function.name;
          if (tc.function && tc.function.arguments) c.function.arguments += tc.function.arguments;
        }
        run.live = { ...msg, content: msg.content.replace(/<tool_call>[\s\S]*$/, '') };
        if (chat === current) {
          const stick = atBottom(), el = msgEl(run.live, -1, true);
          if (run.liveEl && run.liveEl.isConnected) run.liveEl.replaceWith(el); else log.appendChild(el);
          run.liveEl = el;
          if (stick) log.scrollTop = log.scrollHeight;
        }
        const secs = (performance.now() - first) / 1000;
        setStatus(calls.length || /<tool_call>/.test(msg.content) ? `writing a ${calls.map((c) => c.function.name).filter(Boolean).join(', ') || 'tool'} call… ${tokens} tokens` :
          msg.content ? `answering · ${secs > 0 ? (tokens / secs).toFixed(1) : '…'} tok/s` : msg.reasoning ? 'thinking…' : 'reading…', true, chat);
      }
    }
    if (calls.length) msg.tool_calls = calls.map((c) => ({ ...c, id: c.id || 'call_' + uid() }));
    const secs = (performance.now() - t0) / 1000;
    msg.stats = { tps: first ? (tokens / Math.max(0.1, (performance.now() - first) / 1000)).toFixed(1) : '0', secs: Math.round(secs) };
    run.live = null;
    if (run.liveEl) { run.liveEl.remove(); run.liveEl = null; }
    return msg;
  }

  // a tool call the server couldn't parse comes back as text: repair it if possible
  function recoverToolCall(msg) {
    const m = (msg.content || '').match(/<tool_call>\s*([\s\S]*?)\s*(<\/tool_call>|$)/);
    if (!m) return false;
    try {
      const o = JSON.parse(m[1]);
      if (!o.name) return false;
      msg.tool_calls = [{ id: 'call_' + uid(), type: 'function', function: { name: o.name, arguments: JSON.stringify(o.arguments || {}) } }];
      msg.content = msg.content.replace(m[0], '').trim();
      return true;
    } catch { return false; }
  }

  // ---------- tools ----------
  // the model called a connector's tool by its bare name: send it through use_connector
  function routeConnectorCall(call) {
    const n = call.function.name;
    if (n === 'todo_write' || state.tools.some((t) => t.function.name === n)) return;
    const owner = (state.connectors || []).find((c) => c.ok && (c.tool_names || []).includes(n.includes('__') ? n.split('__').pop() : n));
    if (!owner) return;
    call.function = { name: 'use_connector', arguments: JSON.stringify({ connector: owner.name, tool: n.includes('__') ? n.split('__').pop() : n, arguments: parseArgs(call) }) };
  }
  async function runTool(chat, call) {
    routeConnectorCall(call);
    const name = call.function.name, args = call.function.arguments || '{}';
    if (name === 'todo_write') {
      const a = parseArgs(call);
      if (!Array.isArray(a.todos)) return { status: 'error', result: 'error: todos must be a list' };
      chat.todos = a.todos.map((t) => ({ content: String(t.content || ''), status: ['pending', 'in_progress', 'completed'].includes(t.status) ? t.status : 'pending' }));
      if (chat === current) { renderPanel(); if (!app.classList.contains('with-panel') && chat.todos.length) openPanel('tasks'); }
      return { status: 'ok', result: `Updated the to-do list (${chat.todos.filter((t) => t.status === 'completed').length}/${chat.todos.length} done).` };
    }
    if ((chat.mode || 'agent') === 'plan' && !READ_ONLY.has(name)) return { status: 'blocked', result: 'Plan mode: no changes allowed. Put this step in the plan instead.' };
    const send = (approved) => api('/agent/tool', { name, arguments: args, approved, folder: chat.folder || '' });
    let res = await send(chat.allowAll);
    if (res.status === 'needs_approval') {
      const choice = await askUser(chat, call, res);
      if (choice === 'skip') return { status: 'declined', summary: res.summary, result: 'The user declined this. Don\'t retry it; ask what they want instead or choose another way.' };
      if (choice === 'all') chat.allowAll = true;
      setStatus(SETUP_TOOLS.has(name) ? 'setting it up (the first time can take a minute)…' : 'running ' + name + '…', true, chat);
      res = await send(true);
    }
    if (res.status === 'ok' && res.result) recordChange(chat, name, parseArgs(call), res.result);
    if (res.status === 'ok' && SETUP_TOOLS.has(name)) await loadState().catch(() => {});   // new connector tools or skills from the next step on
    return res;
  }
  // a tool waits for the user's OK: kept in the chat's run, so it survives switching chats
  function askUser(chat, call, res) {
    return new Promise((resolve) => {
      const run = runOf(chat);
      run.ask = { callId: call.id, res, done: (c) => { run.ask = null; setStatus('running…', true, chat); if (chat === current) renderLog(true); renderSide(); resolve(c); } };
      setStatus('waiting for your OK…', false, chat);
      renderSide();
      if (chat === current) renderLog(true);
    });
  }
  function attachAsk(run) {
    const card = $(`.tool[data-call="${CSS.escape(run.ask.callId)}"]`, log);
    if (!card) return;
    const res = run.ask.res;
    const box = document.createElement('div');
    box.className = 'ask';
    const what = { command: 'Run this command?', write: 'Change this file?', mcp: 'Use this connector?', web: 'Use the internet?', read: 'Read this?', setup: 'Change Zorlo\'s connectors or skills?' }[res.group] || 'Allow this?';
    box.innerHTML = `<p><b>${esc(what)}</b>${res.outside_project ? ' <span class="warn">This is outside your project folder.</span>' : ''}</p><pre>${esc(res.summary || '')}</pre>` +
      '<button class="btn btn--sig btn--sm" data-c="once" type="button">Allow</button><button class="btn btn--ghost btn--sm" data-c="all" type="button">Allow all in this chat</button><button class="btn btn--ghost btn--sm" data-c="skip" type="button">Skip</button>';
    card.appendChild(box);
    card.classList.remove('run');
    $('.tool__state', card).textContent = 'waiting for you';
    $$('button', box).forEach((b) => (b.onclick = () => run.ask && run.ask.done(b.dataset.c)));
    setTimeout(() => { const b = $('[data-c="once"]', box); if (b && document.activeElement !== input) b.focus({ preventScroll: true }); });
  }
  function relPath(p) {
    const ws = (state.settings.workspace || '').replace(/[\\/]+$/, '');
    return p && p.toLowerCase().startsWith(ws.toLowerCase()) ? p.slice(ws.length + 1).replace(/\\/g, '/') : p;
  }
  function recordChange(chat, name, args, r) {
    if ((name === 'write_file' || name === 'edit_file') && r.undo_id) {
      chat.changes.push({ id: uid(), path: relPath(r.path), kind: name === 'edit_file' ? 'edited' : r.created ? 'created' : 'rewrote', undo_id: r.undo_id,
        old_text: args.old_text, new_text: args.new_text, lines: args.content ? args.content.split('\n').length : 0, at: Date.now(), undone: false });
      fileCache.at = 0;
    }
    if (chat !== current) { if (r.preview && !chat.preview) chat.preview = r.preview; return; }
    if (r.preview) showPreview(r.preview, !current.preview);
    else if (r.changed_web_file || (name === 'write_file' && /\.(css|js)$/i.test(args.path || ''))) reloadPreview();
    renderPanel();
  }
  async function undo(undoId) {
    try {
      await api('/agent/undo', { undo_id: undoId });
      const c = current.changes.find((x) => x.undo_id === undoId);
      if (c) c.undone = true;
      for (const m of current.messages) if (m.role === 'tool' && m.ui && m.ui.result && m.ui.result.undo_id === undoId) m.ui.undone = true;
      save(); renderLog(); renderPanel(); reloadPreview(); fileCache.at = 0;
      setStatus('undone', false);
    } catch (e) { setStatus('undo failed: ' + e.message, false); }
  }

  // ---------- the loop ----------
  async function send(text, att) {
    if (running(current)) return;
    text = (text || '').trim();
    att = att || attachments;
    if (text.startsWith('/') && !att.length && await command(text)) return;
    if (!text && !att.length) return;
    if (!current) newChat();
    // @mentions: attach those project files
    const mentioned = [...text.matchAll(/(^|\s)@([^\s]+)/g)].map((m) => m[2]);
    if (mentioned.length) {
      const known = new Set((await projectFiles()).map((f) => f[0]));
      for (const p of mentioned) if (known.has(p) && !att.some((a) => a.name === p)) {
        try { const f = await api('/agent/file?path=' + encodeURIComponent(p)); att = [...att, { kind: 'text', name: p, text: f.content.slice(0, 60000), size: f.content.length }]; } catch {}
      }
    }
    current.messages.push({ role: 'user', text, attachments: att });
    if (current.title === 'New chat') current.title = (text || att[0].name).slice(0, 48);
    attachments = [];
    save(); renderSide(); renderLog(); renderAttachments();
    await run(current);
  }
  async function run(chat) {
    if (running(chat)) return;
    const R = { ctrl: new AbortController(), live: null, liveEl: null, ask: null, status: { text: 'reading…', busy: true },
                instructions: chat.folder === state.settings.workspace ? state.instructions : null };
    runs.set(chat.id, R);
    const here = () => chat === current;
    const paint = () => { if (here()) { renderLog(true); renderCtx(); } };
    renderSide(); if (here()) paintStatus();
    const mode = chat.mode || 'agent', maxSteps = Math.max(1, +state.settings.max_steps || 25);
    let repairs = 0;
    try {
      for (let step = 0; step < maxSteps; ++step) {
        setStatus('reading…', true, chat);
        const msg = await callModel(chat, mode);
        if (!msg.tool_calls && /<tool_call>/.test(msg.content) && !recoverToolCall(msg)) {
          if (repairs++ < 2) {   // ask the model to send it again, valid this time
            chat.messages.push({ ...msg, content: msg.content.replace(/<tool_call>[\s\S]*$/, '').trim() });
            chat.messages.push({ role: 'user', hidden: true, text: 'Your last tool call could not be read: its JSON was not valid. Send the tool call again with valid JSON (escape every " and newline inside strings). For a long file, write a shorter first version and add to it with edit_file.' });
            paint();
            continue;
          }
        }
        if (mode === 'plan' && !msg.tool_calls) msg.plan = true;
        chat.messages.push(msg);
        save(chat); paint();
        if (!msg.tool_calls) break;
        for (const call of msg.tool_calls) {
          setStatus('running ' + call.function.name + '…', true, chat);
          let res;
          try { res = await runTool(chat, call); } catch (e) { res = { status: 'error', result: 'error: ' + e.message }; }
          let content = typeof res.result === 'string' ? res.result : JSON.stringify(res.result);
          if (content.length > MAX_TOOL_CHARS) content = content.slice(0, MAX_TOOL_CHARS / 2) + '\n[... cut ...]\n' + content.slice(-MAX_TOOL_CHARS / 2);
          chat.messages.push({ role: 'tool', tool_call_id: call.id, content, ui: res });
          save(chat); paint();
          if (R.ctrl.signal.aborted) throw new DOMException('stopped', 'AbortError');
        }
        if (step === maxSteps - 1) chat.messages.push({ role: 'note', content: `Paused after ${maxSteps} steps (Settings → Agent). Say "continue" to go on.` });
      }
    } catch (e) {
      if (e.name !== 'AbortError') chat.messages.push({ role: 'error', content: 'Something went wrong: ' + e.message });
      else chat.messages.push({ role: 'note', content: 'Stopped.' });
    } finally {
      runs.delete(chat.id);
      save(chat);
      if (here()) renderAll(); else renderSide();
      if (here()) setStatus('ready', false);
    }
  }
  function stopRun(chat) {
    const R = runOf(chat);
    if (!R) return;
    if (R.ask) R.ask.done('skip');
    R.ctrl.abort();
  }

  // ---------- slash commands ----------
  const COMMANDS = [
    ['/new', 'Start a new chat'], ['/plan', 'Plan mode: look around and propose a plan'], ['/agent', 'Agent mode: do the work'], ['/chat', 'Chat mode: no tools'],
    ['/undo', 'Undo the agent\'s last file change'], ['/clear', 'Clear this chat'], ['/files', 'Show the project files'], ['/preview', 'Show the website preview'],
    ['/export', 'Download this chat as a Markdown file'], ['/settings', 'Open settings'], ['/help', 'What Zorlo can do'],
  ];
  async function command(text) {
    const [cmd, ...rest] = text.split(/\s+/);
    const arg = text.slice(cmd.length).trim();
    switch (cmd.toLowerCase()) {
      case '/new': newChat(); return true;
      case '/plan': case '/agent': case '/chat': setMode(cmd.slice(1)); if (arg) { input.value = ''; send(arg); } else input.value = ''; return true;
      case '/clear': if (current) { current.messages = []; current.todos = []; save(); renderAll(); } input.value = ''; return true;
      case '/undo': { const c = current && [...current.changes].reverse().find((x) => !x.undone); if (c) await undo(c.undo_id); else setStatus('nothing to undo', false); input.value = ''; return true; }
      case '/files': openPanel('files'); input.value = ''; return true;
      case '/preview': openPanel('preview'); input.value = ''; return true;
      case '/settings': input.value = ''; $('#open-settings').click(); return true;
      case '/export': exportChat(); input.value = ''; return true;
      case '/help':
        input.value = '';
        if (!current) newChat();
        current.messages.push({ role: 'note', content: 'Modes: Chat (just talk), Plan (look around, propose a plan, change nothing), Agent (do the work). Type @ to add a project file, / for commands, or drop files to attach them. Every file change can be undone from its card or the Changes tab. Settings decide what the agent may do without asking.' });
        renderLog(); return true;
      default: return false;
    }
  }
  function exportChat() {
    if (!current) return;
    let out = `# ${current.title}\n\n`;
    for (const m of current.messages) {
      if (m.role === 'user' && !m.hidden) out += `**You:** ${m.text}\n\n`;
      if (m.role === 'assistant') { if (m.content) out += `**Zorlo:** ${m.content}\n\n`; for (const c of m.tool_calls || []) out += `> tool \`${c.function.name}\` ${c.function.arguments.slice(0, 300)}\n\n`; }
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([out], { type: 'text/markdown' }));
    a.download = current.title.replace(/[^\w -]+/g, '').slice(0, 40) + '.md';
    a.click();
  }

  // ---------- modes ----------
  function setMode(mode) { if (!current) newChat(); current.mode = mode; save(); renderMode(); renderCtx(); }
  function renderMode() {
    const mode = (current && current.mode) || 'agent';
    $$('#mode-seg button').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));
    input.placeholder = { chat: 'Ask anything…', plan: 'Describe what you want; Zorlo will look around and propose a plan…', agent: 'Ask Zorlo to build, fix or find something…' }[mode];
  }
  $$('#mode-seg button').forEach((b) => (b.onclick = () => { if (!running(current)) setMode(b.dataset.mode); }));

  // ---------- composer: attachments, popups ----------
  const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + 'px'; };
  // pictures and videos show as thumbnails, other files as cards with their type
  const extOf = (n) => ((n || '').split('.').pop() || '').toUpperCase().slice(0, 5);
  function attHtml(a, i) {
    const rm = i === undefined ? '' : `<button class="att__rm" type="button" data-rm="${i}" aria-label="Remove ${esc(a.name)}">${ICON.x}</button>`;
    if (a.thumb) return `<figure class="att-img${a.video ? ' video' : ''}" title="${esc(a.name)}"><img src="${a.thumb}" alt="${esc(a.name)}" data-zoom loading="lazy">${a.video ? '<span class="att-play"></span>' : ''}${rm}</figure>`;
    const info = a.info || kb(a.size);
    return `<span class="att-file" title="${esc(a.note || a.path || a.name)}"><span class="att-type">${esc(a.kind === 'text' ? extOf(a.name) || 'TXT' : extOf(a.name) || 'FILE')}</span>` +
      `<span class="att-meta"><b>${esc(a.name)}</b><small>${esc(a.busy ? 'reading…' : info)}</small></span>${rm}</span>`;
  }
  function renderAttachments() {
    $('#attachments').innerHTML = attachments.map((a, i) => attHtml(a, i)).join('');
    $$('[data-rm]', $('#attachments')).forEach((b) => (b.onclick = () => { attachments.splice(+b.dataset.rm, 1); renderAttachments(); }));
  }
  // a small copy of a picture (or a video's first frame) kept with the chat, so it shows after a reload too
  async function makeThumb(f, max = 640) {
    try {
      let src;
      if (f.type.startsWith('video/')) {
        const v = document.createElement('video');
        v.muted = true; v.preload = 'auto'; v.src = URL.createObjectURL(f);
        await new Promise((ok, no) => { v.onloadeddata = ok; v.onerror = no; setTimeout(no, 8000); });
        v.currentTime = Math.min(1, (v.duration || 2) / 3);
        await new Promise((ok) => { v.onseeked = ok; setTimeout(ok, 3000); });
        src = v;
      } else src = await createImageBitmap(f);
      const w = src.videoWidth || src.width, h = src.videoHeight || src.height, s = Math.min(1, max / Math.max(w, h));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(w * s)); c.height = Math.max(1, Math.round(h * s));
      c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
      if (src.src) URL.revokeObjectURL(src.src);
      return c.toDataURL('image/jpeg', 0.82);
    } catch { return ''; }
  }
  document.addEventListener('click', (e) => {
    const img = e.target.closest('[data-zoom]');
    if (!img) return;
    $('#zoom-img').src = img.src; $('#zoom-name').textContent = img.alt || '';
    $('#zoom').showModal();
  });
  async function addFiles(list) {
    for (const f of list) {
      try {
        if (f.size <= TEXT_ATTACH_MAX && (f.type.startsWith('text/') || TEXT_EXT.test(f.name) || f.type === 'application/json')) {
          attachments.push({ kind: 'text', name: f.name, text: await f.text(), size: f.size });
        } else {
          setStatus('saving ' + f.name + ' in the project…', true);
          const r = await api('/agent/upload?name=' + encodeURIComponent(f.name), f, true);
          const a = { kind: 'file', name: f.name, path: r.path, size: f.size, image: f.type.startsWith('image/'), video: f.type.startsWith('video/') };
          if (a.image || a.video) a.thumb = await makeThumb(f);
          fileCache.at = 0;
          // documents, PDFs, spreadsheets, pictures (OCR): their text goes with the message; the file stays in the project
          setStatus('reading ' + f.name + '…', true);
          try {
            const x = await api('/agent/extract?path=' + encodeURIComponent(r.path), {});
            const t = x.text || '';
            Object.assign(a, { text: t.slice(0, 40000), cut: t.length > 40000 ? t.length : 0, note: x.note || '',
              info: [x.pages && x.pages + (x.kind === 'pptx' ? ' slides' : x.kind === 'xlsx' ? ' sheets' : ' pages'), x.width && x.width + 'x' + x.height, x.duration_s && Math.round(x.duration_s) + ' s'].filter(Boolean).join(', ') });
          } catch (e) { a.note = 'could not read it: ' + e.message; }
          attachments.push(a);
        }
      } catch (e) { setStatus('could not attach ' + f.name + ': ' + e.message, false); continue; }
    }
    setStatus('ready', false);
    renderAttachments(); input.focus();
  }
  $('#attach').onclick = () => $('#file-input').click();
  $('#file-input').onchange = (e) => { addFiles([...e.target.files]); e.target.value = ''; };
  input.addEventListener('paste', (e) => { const fs = [...(e.clipboardData && e.clipboardData.files || [])]; if (fs.length) { e.preventDefault(); addFiles(fs); } });
  let dragDepth = 0;
  addEventListener('dragenter', (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) { ++dragDepth; $('#main').classList.add('dragging'); } });
  addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('#main').classList.remove('dragging'); } });
  addEventListener('dragover', (e) => e.preventDefault());
  addEventListener('drop', (e) => { e.preventDefault(); dragDepth = 0; $('#main').classList.remove('dragging'); if (e.dataTransfer.files.length) addFiles([...e.dataTransfer.files]); });

  async function projectFiles() {
    if (Date.now() - fileCache.at > 15000) { try { fileCache = { at: Date.now(), list: (await api('/agent/files')).files }; } catch { fileCache = { at: Date.now(), list: [] }; } }
    return fileCache.list;
  }
  const popup = $('#popup');
  let popItems = [], popSel = 0, popApply = null;
  function showPopup(items, apply) {
    popItems = items; popSel = 0; popApply = apply;
    if (!items.length) { popup.hidden = true; return; }
    popup.innerHTML = items.map((it, i) => `<button type="button" class="${i === 0 ? 'on' : ''}" data-i="${i}"><b>${esc(it[0])}</b><span>${esc(it[1] || '')}</span></button>`).join('');
    popup.hidden = false;
    $$('button', popup).forEach((b) => (b.onmousedown = (e) => { e.preventDefault(); popApply(popItems[+b.dataset.i]); popup.hidden = true; }));
  }
  async function updatePopup() {
    const v = input.value, caret = input.selectionStart, before = v.slice(0, caret);
    if (/^\/\S*$/.test(v)) return showPopup(COMMANDS.filter(([c]) => c.startsWith(v.toLowerCase())), (it) => { input.value = it[0] + ' '; autosize(); });
    const m = before.match(/(^|\s)@([^\s@]*)$/);
    if (m) {
      const q = m[2].toLowerCase(), files = await projectFiles();
      const hits = files.filter((f) => f[0].toLowerCase().includes(q)).slice(0, 12).map((f) => [f[0], kb(f[1])]);
      return showPopup(hits, (it) => { const start = caret - m[2].length; input.value = v.slice(0, start) + it[0] + ' ' + v.slice(caret); input.setSelectionRange(start + it[0].length + 1, start + it[0].length + 1); autosize(); });
    }
    popup.hidden = true;
  }
  input.addEventListener('input', () => { autosize(); updatePopup(); renderCtxSoon(); });
  input.addEventListener('blur', () => setTimeout(() => (popup.hidden = true), 150));
  input.addEventListener('keydown', (e) => {
    if (!popup.hidden) {
      const btns = $$('button', popup);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); btns[popSel].classList.remove('on'); popSel = (popSel + (e.key === 'ArrowDown' ? 1 : btns.length - 1)) % btns.length; btns[popSel].classList.add('on'); btns[popSel].scrollIntoView({ block: 'nearest' }); return; }
      if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); popApply(popItems[popSel]); popup.hidden = true; return; }
      if (e.key === 'Escape') { popup.hidden = true; return; }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#composer').requestSubmit(); }
  });
  let ctxTimer;
  const renderCtxSoon = () => { clearTimeout(ctxTimer); ctxTimer = setTimeout(renderCtx, 300); };
  $('#composer').addEventListener('submit', (e) => { e.preventDefault(); const t = input.value; input.value = ''; autosize(); popup.hidden = true; send(t); });
  $('#stop').onclick = () => stopRun(current);
  $('#new-chat').onclick = () => newChat();
  $('#new-chat-top').onclick = () => newChat();
  $('#chat-search').oninput = renderSide;
  $('#toggle-side').onclick = () => $('#side').classList.toggle('open');
  addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); newChat(); } });

  // ---------- side panel ----------
  let panelTab = 'tasks';
  function openPanel(tab) { panelTab = tab || panelTab; app.classList.add('with-panel'); $('#panel').hidden = false; renderPanel(); if (panelTab === 'files') renderFiles(true); }
  function closePanel() { app.classList.remove('with-panel'); $('#panel').hidden = true; }
  $('#toggle-panel').onclick = () => (app.classList.contains('with-panel') ? closePanel() : openPanel());
  $('#close-panel').onclick = closePanel;
  $$('.tabs [data-tab]').forEach((b) => (b.onclick = () => openPanel(b.dataset.tab)));
  function renderPanel() {
    const n = current ? current.changes.filter((c) => !c.undone).length : 0;
    $('#changes-count').textContent = n ? n : '';
    if ($('#panel').hidden) return;
    $$('.tabs [data-tab]').forEach((b) => b.classList.toggle('on', b.dataset.tab === panelTab));
    $$('.tab').forEach((t) => t.classList.toggle('on', t.dataset.pane === panelTab));
    const todos = (current && current.todos) || [];
    $('#tasks').innerHTML = todos.length ? '<ul class="todos">' + todos.map((t) => `<li class="${t.status}"><i></i><span>${esc(t.content)}</span></li>`).join('') + '</ul>'
      : '<p class="hint">For bigger jobs the agent writes its to-do list here and ticks items off as it works.</p>';
    const ch = (current && current.changes) || [];
    $('#changes').innerHTML = ch.length ? [...ch].reverse().map((c) => `<div class="change${c.undone ? ' undone' : ''}"><div class="change__head"><span class="change__kind">${c.undone ? 'undone' : c.kind}</span><b title="${esc(c.path)}">${esc(c.path)}</b>` +
      `<button class="btn btn--ghost btn--sm" data-open="${esc(c.path)}" type="button">Open</button>${c.undone ? '' : `<button class="btn btn--ghost btn--sm" data-undo-c="${esc(c.undo_id)}" type="button">Undo</button>`}</div>` +
      (c.old_text !== undefined ? `<details><summary class="hint" style="padding:0 10px 6px">Show change</summary>${diffHtml(c.old_text, c.new_text)}</details>` : '') + '</div>').join('')
      : '<p class="hint">Nothing changed yet in this chat.</p>';
    $('#undo-all').disabled = !ch.some((c) => !c.undone);
    $$('[data-undo-c]', $('#changes')).forEach((b) => (b.onclick = () => undo(b.dataset.undoC)));
    $$('[data-open]', $('#changes')).forEach((b) => (b.onclick = () => openFile(b.dataset.open)));
    const url = (current && current.preview) || '';
    $('#preview-url').value = url;
    $('#preview-open').href = url || '#';
    $('#preview-empty').hidden = !!url;
    const frame = $('#preview-frame');
    if (panelTab === 'preview' && url && frame.dataset.src !== url) { frame.dataset.src = url; frame.src = url; }
  }
  $('#undo-all').onclick = async () => { for (const c of [...current.changes].reverse()) if (!c.undone) await undo(c.undo_id); };
  async function renderFiles(force) {
    if (force) fileCache.at = 0;
    const list = await projectFiles(), q = $('#file-filter').value.trim().toLowerCase();
    const shown = list.filter((f) => !q || f[0].toLowerCase().includes(q)).slice(0, 500);
    $('#files').innerHTML = shown.length ? shown.map((f) => `<button type="button" data-f="${esc(f[0])}"><span>${esc(f[0])}</span><span class="size">${kb(f[1])}</span></button>`).join('')
      : `<p class="hint">${list.length ? 'No files match.' : 'The project folder is empty.'}</p>`;
    $$('[data-f]', $('#files')).forEach((b) => (b.onclick = () => openFile(b.dataset.f)));
  }
  $('#file-filter').oninput = () => renderFiles(false);
  $('#files-refresh').onclick = () => renderFiles(true);
  async function openFile(path) {
    try {
      const f = await api('/agent/file?path=' + encodeURIComponent(path));
      $('#viewer-path').textContent = path;
      $('#viewer-body').innerHTML = highlight(f.content, path.split('.').pop()).split('\n').map((l) => `<span class="ln">${l || ' '}</span>`).join('');
      $('#viewer-preview').hidden = !/\.html?$/i.test(path);
      $('#viewer').dataset.path = path;
      $('#viewer').showModal();
    } catch (e) { setStatus('cannot open ' + path + ': ' + e.message, false); }
  }
  $('#viewer-close').onclick = () => $('#viewer').close();
  $('#viewer-mention').onclick = () => { const p = $('#viewer').dataset.path; input.value = (input.value ? input.value.trimEnd() + ' ' : '') + '@' + p + ' '; autosize(); $('#viewer').close(); input.focus(); };
  $('#viewer-preview').onclick = () => { $('#viewer').close(); showPreview(`${location.origin}/preview/${$('#viewer').dataset.path.split('/').map(encodeURIComponent).join('/')}`, true); };
  function showPreview(url, open) {
    if (!current) return;
    current.preview = url; save();
    if (open) openPanel('preview'); else renderPanel();
    reloadPreview();
  }
  function reloadPreview() {
    const frame = $('#preview-frame'), url = current && current.preview;
    if (!url || $('#panel').hidden || panelTab !== 'preview') return;
    frame.dataset.src = url;
    frame.src = url + (url.includes('?') ? '&' : '?') + '_=' + Date.now();
  }
  $('#preview-reload').onclick = reloadPreview;
  $('#preview-url').onchange = (e) => showPreview(e.target.value.trim(), true);
  $('#preview-width').onclick = (e) => { const w = $('#frame-wrap').classList.toggle('phone'); e.target.textContent = w ? 'Desktop' : 'Phone'; };

  // ---------- settings ----------
  // model providers: any server with the OpenAI chat API works; these fill in the address
  const PROVIDERS = [
    ['zorlo', 'Zorlo, on this PC', '', '', 12288],
    ['openai', 'OpenAI', 'https://api.openai.com/v1', 'https://platform.openai.com/api-keys', 128000],
    ['anthropic', 'Anthropic', 'https://api.anthropic.com/v1', 'https://console.anthropic.com/settings/keys', 200000],
    ['gemini', 'Google Gemini', 'https://generativelanguage.googleapis.com/v1beta/openai', 'https://aistudio.google.com/apikey', 1000000],
    ['deepseek', 'DeepSeek', 'https://api.deepseek.com/v1', 'https://platform.deepseek.com/api_keys', 64000],
    ['openrouter', 'OpenRouter', 'https://openrouter.ai/api/v1', 'https://openrouter.ai/keys', 128000],
    ['groq', 'Groq', 'https://api.groq.com/openai/v1', 'https://console.groq.com/keys', 128000],
    ['mistral', 'Mistral', 'https://api.mistral.ai/v1', 'https://console.mistral.ai/api-keys', 128000],
    ['xai', 'xAI', 'https://api.x.ai/v1', 'https://console.x.ai', 128000],
    ['together', 'Together AI', 'https://api.together.xyz/v1', 'https://api.together.ai/settings/api-keys', 128000],
    ['ollama', 'Ollama (on this PC)', 'http://localhost:11434/v1', '', 32768],
    ['lmstudio', 'LM Studio (on this PC)', 'http://localhost:1234/v1', '', 32768],
    ['custom', 'Other OpenAI-compatible server', '', '', 128000],
  ];
  const presetOf = (k) => PROVIDERS.find((p) => p[0] === k) || PROVIDERS[0];
  $('#p-kind').innerHTML = PROVIDERS.map((p) => `<option value="${p[0]}">${esc(p[1])}</option>`).join('');
  let pHasKey = false, pSavedBase = '';
  function fillProvider() {
    const p = state.settings.provider || { kind: 'zorlo' };
    $('#p-kind').value = p.kind || 'zorlo';
    $('#p-base').value = p.base_url || presetOf(p.kind)[2];
    $('#p-model').value = p.model || '';
    $('#p-ctx').value = p.context || presetOf(p.kind)[4];
    pHasKey = !!p.has_key; pSavedBase = p.base_url || '';
    $('#p-msg').textContent = '';
    showProvider(false);
  }
  function showProvider(changedKind) {
    const k = $('#p-kind').value, pre = presetOf(k), local = k === 'zorlo';
    $('#p-remote').hidden = local;
    if (changedKind) { $('#p-base').value = pre[2]; $('#p-ctx').value = pre[4]; $('#p-model').value = ''; $('#p-models').innerHTML = ''; }
    $('#p-base').readOnly = !!pre[2] && k !== 'custom';
    const keyless = k === 'ollama' || k === 'lmstudio';
    $('#p-key').placeholder = keyless ? 'not needed' : pHasKey && $('#p-base').value.replace(/\/+$/, '') === pSavedBase ? 'saved (enter a new one to replace it)' : 'paste your API key';
    $('#p-getkey').hidden = !pre[3]; if (pre[3]) $('#p-getkey').href = pre[3];
  }
  $('#p-kind').onchange = () => showProvider(true);
  $('#model-badge').onclick = () => $('#open-settings').click();
  $('#p-base').oninput = () => showProvider(false);
  const providerPayload = () => {
    const k = $('#p-kind').value;
    return { provider: { kind: k, base_url: k === 'zorlo' ? '' : $('#p-base').value.trim(), model: $('#p-model').value.trim(), context: +$('#p-ctx').value || presetOf(k)[4] },
             provider_key: $('#p-key').value.trim() };
  };
  $('#p-load').onclick = async () => {
    $('#p-msg').textContent = 'Asking the provider for its models…';
    try {
      await api('/agent/settings', providerPayload());   // the list needs the address and key
      $('#p-key').value = ''; pHasKey = true; pSavedBase = $('#p-base').value.trim().replace(/\/+$/, '');
      const r = await api('/agent/llm/models');
      if (r.error) { $('#p-msg').textContent = r.error; return; }
      $('#p-models').innerHTML = r.models.sort().map((m) => `<option value="${esc(m)}">`).join('');
      $('#p-msg').textContent = `${r.models.length} models available. Start typing in the Model field to pick one.`;
      await loadState().catch(() => {});
      showProvider(false);
    } catch (e) { $('#p-msg').textContent = e.message; }
  };
  function renderModelBadge() {
    const p = state.settings.provider || {}, r = remote();
    $('#speed').hidden = r; $('#model-badge').hidden = !r; $('#thinking').closest('.switch').hidden = r;
    if (r) { $('#model-badge').textContent = p.model || 'no model set'; $('#model-badge').title = (presetOf(p.kind)[1]) + ' · ' + (p.base_url || '') + ' (Settings → Model)'; }
  }
  const PERM = [['allow', 'Allow'], ['ask', 'Ask me'], ['deny', 'Block']];
  $$('[data-perm]').forEach((sel) => { sel.innerHTML = PERM.map(([v, t]) => `<option value="${v}">${t}</option>`).join(''); });
  async function loadState() {
    const folder = state.settings.workspace;
    state = await api('/agent/state');
    const s = state.settings;
    if (folder && current && current.folder && s.workspace !== current.folder) await useFolder(current.folder);
    for (const c of chats) if (!c.folder) c.folder = s.workspace;   // chats from before folders
    $('#folder-path').textContent = (current && current.folder) || s.workspace;
    $('#speed').value = s.mode || 'exact';
    $('#thinking').checked = !!s.thinking;
    renderModelBadge();
    renderCtx(); renderSide();
    const up = state.update;   // a newer Zorlo is available
    $('#update').hidden = !(up && up.latest);
    if (up && up.latest) {
      $('#update').innerHTML = `<b>Zorlo ${esc(up.latest)} is available</b><span>${esc(up.notes || 'You have ' + state.version + '.')} ${up.url ? 'Download it' : 'Download it from the Zorlo website'}</span>`;
      if (up.url) $('#update').href = up.url; else $('#update').removeAttribute('href');
    }
    // connectors still starting in the background: look again shortly, so their tools are there for the next step
    clearTimeout(loadState.again);
    if ((state.connectors || []).some((c) => c.starting)) loadState.again = setTimeout(() => loadState().catch(() => {}), 3000);
    if ($('#settings').open) renderConnectors();
    if ($('#market').open) renderMarket();
  }
  function fillSettings() {
    const s = state.settings;
    $('#s-workspace').value = (current && current.folder) || s.workspace || '';
    $('#s-instructions').innerHTML = state.instructions ? `Project instructions found: <b>${esc(state.instructions.file)}</b> (the agent follows it).` : 'Tip: put a <code>ZORLO.md</code> (or AGENTS.md / CLAUDE.md) in the project folder with rules the agent should always follow.';
    $('#s-full').checked = !!s.full_permission;
    $$('[data-perm]').forEach((sel) => (sel.value = (s.permissions || {})[sel.dataset.perm] || 'ask'));
    $('#s-brave').value = ''; $('#s-brave').placeholder = s.brave_key_set ? 'saved (enter a new one to replace it)' : 'Brave Search API key (optional)';
    $('#s-skills').value = s.skills_dir || '';
    $('#s-mcp').value = JSON.stringify({ mcpServers: s.mcpServers || {} }, null, 2);
    $('#s-steps').value = s.max_steps || 25;
    $('#s-file').textContent = 'Saved in ' + state.settings_file;
    $('#s-error').textContent = '';
    fillProvider();
    renderConnectors();
  }
  // the connectors and skills you have, each with its own switch / remove
  function renderConnectors() {
    const list = state.connectors || [];
    $('#s-mcp-list').innerHTML = list.length ? list.map((c) => {
      const st = !c.enabled ? ['off', 'Off'] : c.starting ? ['off', 'Starting…'] : c.ok ? ['ok', `Connected · ${c.tools} tool${c.tools === 1 ? '' : 's'}`] : ['bad', 'Not working'];
      return `<div class="conn${c.enabled ? '' : ' conn--off'}" data-name="${esc(c.name)}"><span class="conn__dot ${st[0]}"></span>` +
        `<div class="conn__main"><b>${esc(c.name)}</b><span class="conn__state ${st[0]}">${esc(st[1])}</span><code title="${esc(c.command)}">${esc(c.command)}</code>` +
        (c.enabled && !c.ok && !c.starting && c.error ? `<p class="conn__err">${esc(c.error)}</p>` : '') + '</div>' +
        `<label class="switch" title="${c.enabled ? 'Switch off' : 'Switch on'}"><input type="checkbox" data-toggle ${c.enabled ? 'checked' : ''}><span class="sr">On</span></label>` +
        `<button class="icon-btn icon-btn--show" type="button" data-remove title="Remove" aria-label="Remove ${esc(c.name)}">${ICON.x}</button></div>`;
    }).join('') : '<p class="hint">No connectors yet. Browse the list and add one with a click, or ask in a chat: “add the Playwright browser connector”.</p>';
    $$('#s-mcp-list .conn').forEach((row) => {
      const name = row.dataset.name;
      $('[data-toggle]', row).onchange = async (e) => {
        row.classList.add('busy');
        $('.conn__state', row).textContent = e.target.checked ? 'Starting…' : 'Stopping…';
        try { await api('/agent/connector', { action: e.target.checked ? 'enable' : 'disable', name }); } catch (err) { $('#s-error').textContent = err.message; }
        await loadState().catch(() => {});
      };
      $('[data-remove]', row).onclick = async () => {
        if (!confirm(`Remove the connector “${name}”?`)) return;
        try { await api('/agent/connector', { action: 'remove', name }); } catch (err) { $('#s-error').textContent = err.message; }
        await loadState().catch(() => {});
      };
    });
    const sk = state.skills || [];
    $('#s-skill-list').innerHTML = sk.length ? sk.map((k) => `<div class="conn" data-skill="${esc(k.name)}"><span class="conn__dot skill"></span><div class="conn__main"><b>${esc(k.name)}</b><span class="conn__desc">${esc(k.description)}</span></div>` +
      `<button class="icon-btn icon-btn--show" type="button" data-rm-skill title="Remove" aria-label="Remove ${esc(k.name)}">${ICON.x}</button></div>`).join('')
      : '<p class="hint">No skills yet. Browse skills to install one, or ask in a chat: “install the frontend-design skill”.</p>';
    $$('#s-skill-list [data-rm-skill]').forEach((b) => (b.onclick = async () => {
      const name = b.closest('[data-skill]').dataset.skill;
      if (!confirm(`Remove the skill “${name}”? Its folder is deleted.`)) return;
      try { await api('/agent/skill', { action: 'remove', name }); } catch (err) { $('#s-error').textContent = err.message; }
      await loadState().catch(() => {});
    }));
  }
  $('#open-settings').onclick = async () => { await loadState().catch(() => {}); fillSettings(); $('#settings').showModal(); };
  async function pickFolder() {
    setStatus('choose a folder in the window that opened…', true);
    const r = await api('/agent/pick_folder', {}).catch(() => ({}));
    setStatus(running(current) ? 'working…' : 'ready', running(current));
    return r.path;
  }
  // the folder button: a chat in another folder (this chat keeps its own)
  $('#folder').onclick = async () => {
    const p = await pickFolder();
    if (!p) return;
    if (current && !current.messages.length && !running(current)) { current.folder = p; save(); await useFolder(p); renderSide(); }
    else newChat(p);
    await loadState().catch(() => {});
    setStatus('project folder: ' + p, false);
  };
  $('#s-pick').onclick = async () => { const p = await pickFolder(); if (p) $('#s-workspace').value = p; };
  $('#s-save').onclick = async (e) => {
    e.preventDefault();
    const perms = {};
    $$('[data-perm]').forEach((sel) => (perms[sel.dataset.perm] = sel.value));
    try {
      const ws = $('#s-workspace').value.trim();
      const pp = providerPayload();
      if (pp.provider.kind !== 'zorlo' && !pp.provider.model) { $('#s-error').textContent = 'Model: enter the model name (List models shows what the provider offers).'; return; }
      await api('/agent/settings', { workspace: ws, full_permission: $('#s-full').checked, permissions: perms,
        brave_api_key: $('#s-brave').value.trim(), skills_dir: $('#s-skills').value.trim(), max_steps: +$('#s-steps').value || 25, ...pp });
      $('#p-key').value = '';
      if (current && ws && current.folder !== ws) {
        if (current.messages.length || running(current)) newChat(ws); else { current.folder = ws; save(); }
      }
      await loadState();
      fileCache.at = 0;
      $('#settings').close();
    } catch (err) { $('#s-error').textContent = err.message; }
  };
  $('#s-mcp-apply').onclick = async () => {
    let mcp;
    try { const j = JSON.parse($('#s-mcp').value || '{}'); mcp = j.mcpServers || j; } catch { $('#s-error').textContent = 'Connectors: that is not valid JSON.'; return; }
    $('#s-mcp-apply').disabled = true;
    $('#s-error').textContent = '';
    try { await api('/agent/settings', { mcpServers: mcp }); setStatus('starting connectors…', true); await loadState(); setStatus('ready', false); }
    catch (err) { $('#s-error').textContent = err.message; }
    $('#s-mcp-apply').disabled = false;
  };
  const remember = () => api('/agent/settings', { mode: $('#speed').value, thinking: $('#thinking').checked }).catch(() => {});
  $('#speed').onchange = remember;
  $('#thinking').onchange = remember;

  // ---------- connector and skill markets ----------
  let market = null, skillMarket = null, mTab = 'connectors', mCat = 'All', openForm = '';
  const RUNTIME = { node: 'Node.js', python: 'Python (uv)' };
  async function openMarket(tab) {
    mTab = tab || mTab;
    if (!$('#market').open) $('#market').showModal();
    $('#m-search').value = '';
    renderMarket();
    if (!market) {
      try { market = await api('/agent/market'); } catch (e) { market = { connectors: [], runtimes: {}, error: e.message }; }
      renderMarket();
    }
    if (mTab === 'skills') loadSkillMarket();
  }
  async function loadSkillMarket(force) {
    if (skillMarket && !force) return;
    skillMarket = { loading: true, skills: [] };
    renderMarket();
    try { skillMarket = await api('/agent/skill_market'); } catch (e) { skillMarket = { skills: [], error: e.message }; }
    renderMarket();
  }
  function renderMarket() {
    $$('#market .m-tabs button').forEach((b) => b.classList.toggle('on', b.dataset.mtab === mTab));
    $('#m-search').placeholder = mTab === 'connectors' ? 'Search connectors' : 'Search skills';
    const q = $('#m-search').value.trim().toLowerCase();
    const hit = (o) => !q || JSON.stringify([o.name, o.about, o.description, o.by, o.category, o.id]).toLowerCase().includes(q);
    const body = $('#m-body');
    if (mTab === 'connectors') {
      if (!market) { body.innerHTML = '<p class="hint m-wait"><span class="spin"></span>Loading the list…</p>'; return; }
      const rt = market.runtimes || {};
      const added = new Map((state.connectors || []).map((c) => [c.market || c.name, c]));
      const cats = ['All', ...new Set(market.connectors.map((c) => c.category || 'Other'))];
      const items = market.connectors.filter((c) => (mCat === 'All' || (c.category || 'Other') === mCat) && hit(c));
      let h = '';
      const missing = [rt.node === false && 'node', rt.uv === false && 'uv'].filter(Boolean);
      if (missing.length) h += `<div class="m-note">${missing.includes('node') ? '<p><b>Node.js is not installed.</b> Most connectors need it. <a href="https://nodejs.org/en/download" target="_blank" rel="noopener">Get Node.js (LTS)</a> or <button class="linkish" type="button" data-install="node">ask Zorlo to install it</button>.</p>' : ''}` +
        `${missing.includes('uv') ? '<p><b>uv (Python) is not installed.</b> Connectors marked Python need it. <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank" rel="noopener">Get uv</a> or <button class="linkish" type="button" data-install="uv">ask Zorlo to install it</button>.</p>' : ''}</div>`;
      h += `<div class="m-cats">${cats.map((c) => `<button type="button" class="${c === mCat ? 'on' : ''}" data-cat="${esc(c)}">${esc(c)}</button>`).join('')}</div>`;
      h += '<div class="m-grid">' + items.map((c) => {
        const have = added.get(c.id), needsRt = c.runtime && rt[c.runtime === 'python' ? 'uv' : 'node'] === false;
        const fields = c.fields || [];
        return `<article class="m-card${openForm === c.id ? ' open' : ''}" data-id="${esc(c.id)}"><div class="m-card__top"><b>${esc(c.name)}</b><span class="m-by">${esc(c.by || '')}</span></div>` +
          `<p>${esc(c.about || '')}</p><div class="m-card__foot"><span class="m-tag">${esc(RUNTIME[c.runtime] || c.runtime || '')}</span>${fields.some((f) => f.kind === 'secret') ? '<span class="m-tag">needs a key</span>' : ''}` +
          (have ? `<span class="m-added">${ICON.add}Added${have.ok ? ` · ${have.tools} tools` : ''}</span>` : `<button class="btn btn--sm${needsRt ? ' btn--ghost' : ''}" type="button" data-add ${needsRt ? `title="Needs ${esc(RUNTIME[c.runtime])}"` : ''}>Add</button>`) + '</div>' +
          (openForm === c.id ? `<form class="m-form" data-form>${fields.map((f) => `<label><span>${esc(f.label)}${f.help ? ` <a href="${esc(f.help)}" target="_blank" rel="noopener">get one ↗</a>` : ''}</span>` +
            `<span class="row"><input name="${esc(f.key)}" type="${f.kind === 'secret' ? 'password' : 'text'}" autocomplete="off" spellcheck="false" required>${f.kind === 'folder' ? '<button class="btn btn--ghost btn--sm" type="button" data-pick>Choose…</button>' : ''}</span></label>`).join('')}` +
            `<div class="row"><button class="btn btn--sig btn--sm" type="submit">Connect</button><button class="btn btn--ghost btn--sm" type="button" data-cancel>Cancel</button><span class="m-msg"></span></div></form>` : '') +
          '</article>';
      }).join('') + '</div>';
      if (!items.length) h += '<p class="hint">Nothing matches. Any MCP server can still be added below.</p>';
      h += `<details class="m-custom"><summary>Add another MCP server</summary>
        <p class="hint">A remote server by its address (you may be asked to sign in in your browser), or any program that speaks MCP.</p>
        <form data-remote class="m-form"><label><span>Name</span><input name="name" type="text" placeholder="my-server" required></label><label><span>Server address</span><input name="url" type="text" placeholder="https://example.com/mcp" required></label>
        <div class="row"><button class="btn btn--sm" type="submit">Connect</button><span class="m-msg"></span></div></form>
        <form data-custom class="m-form"><label><span>Name</span><input name="name" type="text" placeholder="my-tool" required></label><label><span>Command</span><input name="command" type="text" placeholder="npx" required></label>
        <label><span>Arguments, one per line</span><textarea name="args" rows="3" placeholder="-y&#10;some-mcp-server"></textarea></label><label><span>Environment, KEY=value per line</span><textarea name="env" rows="2" placeholder="API_KEY=..."></textarea></label>
        <div class="row"><button class="btn btn--sm" type="submit">Connect</button><span class="m-msg"></span></div></form></details>`;
      if (market.error) h = `<p class="error">${esc(market.error)}</p>` + h;
      body.innerHTML = h;
      wireConnectorMarket(body);
    } else {
      const have = new Set((state.skills || []).map((k) => k.name));
      let h = `<form class="m-link" data-skill-link><input name="url" type="text" spellcheck="false" placeholder="Install from a GitHub link: https://github.com/owner/repo/tree/main/my-skill" required><button class="btn btn--sm" type="submit">Install</button><span class="m-msg"></span></form>`;
      if (!skillMarket || skillMarket.loading) h += '<p class="hint m-wait"><span class="spin"></span>Reading the skill list from GitHub…</p>';
      else {
        if (skillMarket.error) h += `<p class="error">${esc(skillMarket.error)} <button class="linkish" type="button" data-retry>Try again</button></p>`;
        const items = skillMarket.skills.filter(hit);
        h += '<div class="m-grid">' + items.map((k) => `<article class="m-card" data-url="${esc(k.url)}"><div class="m-card__top"><b>${esc(k.name)}</b><span class="m-by">${esc(k.by || '')}</span></div>` +
          `<p class="clamp">${esc(k.description || '')}</p><div class="m-card__foot"><span class="m-tag">${esc(k.license || 'skill')}</span><a class="m-src" href="${esc(k.url)}" target="_blank" rel="noopener">source ↗</a>` +
          (have.has(k.name) || have.has(k.folder) ? `<span class="m-added">${ICON.add}Installed</span><button class="btn btn--ghost btn--sm" type="button" data-install-skill>Update</button>` : '<button class="btn btn--sm" type="button" data-install-skill>Install</button>') +
          '</div><span class="m-msg"></span></article>').join('') + '</div>';
        if (!items.length && !skillMarket.error) h += '<p class="hint">Nothing matches. Any skill on GitHub can be installed by its link above.</p>';
      }
      body.innerHTML = h;
      wireSkillMarket(body);
    }
  }
  const busyBtn = (b, t) => { b.disabled = true; b.dataset.label = b.textContent; b.innerHTML = `<span class="spin"></span>${esc(t)}`; };
  const freeBtn = (b) => { b.disabled = false; b.textContent = b.dataset.label || 'Connect'; };
  async function addConnector(payload, btn, msgEl) {
    busyBtn(btn, 'Starting… the first time it downloads');
    msgEl.textContent = ''; msgEl.className = 'm-msg';
    try {
      const r = await api('/agent/connector', { action: 'add', ...payload });
      await loadState().catch(() => {});
      if (r.connected === false) { msgEl.textContent = 'Added, but it did not start: ' + (r.error || ''); msgEl.className = 'm-msg bad'; freeBtn(btn); return false; }
      openForm = '';
      renderMarket();
      setStatus(`connector ${r.name} added${r.tools ? ` · ${r.tools.length} tools` : ''}`, false);
      return true;
    } catch (e) { msgEl.textContent = e.message; msgEl.className = 'm-msg bad'; freeBtn(btn); return false; }
  }
  function wireConnectorMarket(body) {
    $$('[data-cat]', body).forEach((b) => (b.onclick = () => { mCat = b.dataset.cat; renderMarket(); }));
    $$('[data-install]', body).forEach((b) => (b.onclick = () => askToInstall(b.dataset.install)));
    $$('.m-card', body).forEach((card) => {
      const id = card.dataset.id, entry = market.connectors.find((c) => c.id === id);
      const add = $('[data-add]', card);
      if (add) add.onclick = () => {
        if (entry.fields && entry.fields.length) { openForm = id; renderMarket(); const f = $(`.m-card[data-id="${CSS.escape(id)}"] input`, body.ownerDocument); if (f) f.focus(); return; }
        addConnector({ id }, add, $('.m-card__foot', card).appendChild(Object.assign(document.createElement('span'), { className: 'm-msg' })));
      };
      const form = $('[data-form]', card);
      if (form) {
        $$('[data-pick]', form).forEach((b) => (b.onclick = async () => { const p = await pickFolder(); if (p) b.previousElementSibling.value = p; }));
        $('[data-cancel]', form).onclick = () => { openForm = ''; renderMarket(); };
        form.onsubmit = (e) => {
          e.preventDefault();
          const values = Object.fromEntries(new FormData(form).entries());
          addConnector({ id, values }, $('[type=submit]', form), $('.m-msg', form));
        };
      }
    });
    const remote = $('[data-remote]', body);
    remote.onsubmit = (e) => { e.preventDefault(); const v = Object.fromEntries(new FormData(remote).entries()); addConnector({ name: v.name, url: v.url.trim() }, $('[type=submit]', remote), $('.m-msg', remote)); };
    const custom = $('[data-custom]', body);
    custom.onsubmit = (e) => {
      e.preventDefault();
      const v = Object.fromEntries(new FormData(custom).entries());
      const args = v.args.split('\n').map((x) => x.trim()).filter(Boolean);
      const env = {};
      for (const line of v.env.split('\n')) { const k = line.indexOf('='); if (k > 0) env[line.slice(0, k).trim()] = line.slice(k + 1).trim(); }
      addConnector({ name: v.name, command: v.command.trim(), args, env }, $('[type=submit]', custom), $('.m-msg', custom));
    };
  }
  async function installSkill(url, btn, msgEl) {
    busyBtn(btn, 'Installing…');
    msgEl.textContent = ''; msgEl.className = 'm-msg';
    try {
      const r = await api('/agent/skill', { action: 'install', url });
      await loadState().catch(() => {});
      renderMarket();
      setStatus(`skill ${r.installed} installed`, false);
    } catch (e) { msgEl.textContent = e.message; msgEl.className = 'm-msg bad'; freeBtn(btn); }
  }
  function wireSkillMarket(body) {
    const link = $('[data-skill-link]', body);
    link.onsubmit = (e) => { e.preventDefault(); installSkill(link.url.value.trim(), $('[type=submit]', link), $('.m-msg', link)); };
    const retry = $('[data-retry]', body);
    if (retry) retry.onclick = () => loadSkillMarket(true);
    $$('[data-install-skill]', body).forEach((b) => (b.onclick = () => { const card = b.closest('.m-card'); installSkill(card.dataset.url, b, $('.m-msg', card)); }));
  }
  // Node.js / uv missing: the agent installs it (the user approves the command)
  function askToInstall(what) {
    $('#market').close();
    if (!current || current.messages.length || running(current)) newChat();
    setMode('agent');
    send(what === 'node' ? 'Install Node.js LTS on this PC with winget (winget install --id OpenJS.NodeJS.LTS -e --accept-package-agreements --accept-source-agreements), then check it works with: node --version'
      : 'Install uv on this PC with winget (winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements), then check it works with: uv --version');
    market = null;
  }
  $('#m-search').oninput = renderMarket;
  $$('#market .m-tabs button').forEach((b) => (b.onclick = () => { mTab = b.dataset.mtab; openForm = ''; renderMarket(); if (mTab === 'skills') loadSkillMarket(); }));
  $('#m-close').onclick = () => $('#market').close();
  $('#market').addEventListener('close', () => { openForm = ''; if ($('#settings').open) renderConnectors(); });
  $('#s-browse-mcp').onclick = () => openMarket('connectors');
  $('#s-browse-skills').onclick = () => openMarket('skills');

  // ---------- start ----------
  chats.forEach((c) => { c.todos = c.todos || []; c.changes = c.changes || []; c.mode = c.mode || 'agent'; for (const m of c.messages) if (m.role === 'user' && m.text === undefined) m.text = m.content || ''; });
  current = chats[0] || null;
  loadState().then(async () => {
    await syncChats();
    if (!current) current = chats.find((c) => c.messages.length) || null;
    if (!current) newChat(); else { useFolder(current.folder); renderAll(); }
    setStatus('ready', false);
  }).catch((e) => { if (!current) newChat(); else renderAll(); setStatus('agent tools unavailable: ' + e.message, false); });
})();
