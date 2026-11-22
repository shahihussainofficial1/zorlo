#pragma once

// The simple chat page (/simple, and "/" when the agent app's ui folder is missing). Talks to the same server's
// OpenAI API. Chats are kept in this browser (localStorage), with a list to switch between them.
// (Each raw string piece stays under MSVC's 16 KB literal limit; adjacent pieces are joined by the compiler.)

static const char * ZORLO_UI_HTML = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zorlo Chat</title><meta name="zorlo-token" content="{{ZORLO_TOKEN}}">
<style>
:root{--bg:#F5F3EE;--side:#EEEBE4;--panel:#FDFCFA;--s2:#F1EEE8;--s3:#E8E4DC;--text:#151412;--ink2:#3B3833;--muted:#6B675F;--faint:#9C978E;--line:rgba(21,20,18,.08);--line2:rgba(21,20,18,.14);--sig:#E5480F;--sigs:rgba(229,72,15,.10);--err:#B4291D;--sh1:0 1px 1px rgba(28,24,18,.04),0 1px 3px rgba(28,24,18,.05);--sh2:0 1px 2px rgba(28,24,18,.05),0 10px 28px -10px rgba(28,24,18,.16);--sh3:0 2px 6px rgba(28,24,18,.06),0 28px 70px -18px rgba(28,24,18,.32);--eo:cubic-bezier(.23,1,.32,1)}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font:15px/1.6 "Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased;display:grid;grid-template-columns:260px minmax(0,1fr);overflow:hidden}
select,input,textarea,button{font:inherit;color:inherit}
button{cursor:pointer;border:0;background:none}
.side{display:flex;flex-direction:column;background:var(--side);border-right:1px solid var(--line);min-height:0}
.side__top{padding:14px 12px 8px;display:grid;gap:10px}
.brand{display:flex;align-items:center;gap:9px;font-weight:600;font-size:18px;letter-spacing:-.04em;padding:0 4px;color:var(--text);text-decoration:none}
.mark{width:15px;height:15px;border-radius:4px;background:var(--text);position:relative}
.mark::after{content:"";position:absolute;right:-3px;bottom:-3px;width:7px;height:7px;background:var(--sig);border-radius:2px;box-shadow:0 0 0 2px var(--side)}
.new{display:flex;align-items:center;justify-content:space-between;height:34px;padding:0 12px;border-radius:9px;background:var(--panel);box-shadow:var(--sh1),0 0 0 1px var(--line2);font-weight:500;font-size:13.5px;transition:background-color .15s ease,transform .16s var(--eo)}
.new:active{transform:scale(.98)}
kbd{font:11px/1 "Cascadia Mono",Consolas,monospace;padding:2px 5px;border-radius:5px;background:var(--s2);color:var(--muted)}
.search{width:100%;height:32px;padding:0 11px;border:0;border-radius:8px;background:var(--panel);box-shadow:0 0 0 1px var(--line);font-size:13px;outline:none}
.search:focus{box-shadow:0 0 0 1px var(--line2),0 0 0 3px var(--sigs)}
.chats{flex:1;overflow:auto;padding:4px 8px 8px}
.day{font-size:11.5px;font-weight:600;color:var(--faint);padding:10px 10px 4px}
.item{display:flex;align-items:center;gap:6px;width:100%;text-align:left;padding:8px 10px;border-radius:8px;font-size:13.5px;color:var(--ink2);transition:background-color .15s ease}
.item span{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.item.on{background:var(--panel);color:var(--text);font-weight:500;box-shadow:var(--sh1)}
.item i{font-style:normal;opacity:0;color:var(--faint);padding:0 3px;transition:opacity .15s ease}
.item .live{opacity:1;width:7px;height:7px;padding:0;border-radius:50%;background:var(--sig)}
.empty-list{color:var(--faint);font-size:13px;padding:12px 10px}
.side__bottom{padding:8px;border-top:1px solid var(--line)}
.back{display:block;padding:8px 10px;border-radius:8px;font-size:13.5px;color:var(--ink2);text-decoration:none}
.main{display:flex;flex-direction:column;min-width:0;min-height:0}
header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 18px;border-bottom:1px solid var(--line)}
header h1{font-size:14px;font-weight:600;flex:1;min-width:120px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
label{color:var(--muted);font-size:12.5px;display:flex;gap:6px;align-items:center}
select,input[type=number]{height:30px;border:0;border-radius:8px;background:var(--panel);box-shadow:0 0 0 1px var(--line2);padding:0 9px;font-size:13px}
input[type=number]{width:72px}
.menu{display:none;width:34px;height:34px;border-radius:8px;align-items:center;justify-content:center;font-size:18px}
#log{flex:1;overflow-y:auto;padding:24px 18px}
.inner{max-width:780px;margin:0 auto}
.hello{text-align:center;color:var(--muted);padding:12vh 20px 0}
.hello b{display:block;color:var(--text);font-size:26px;font-weight:500;letter-spacing:-.03em;margin-bottom:6px}
.msg{margin:0 0 18px;white-space:pre-wrap;word-wrap:break-word}
.msg.user{margin-left:auto;width:fit-content;max-width:78%;background:var(--s3);padding:10px 15px;border-radius:18px 18px 6px 18px}
.msg .who{font-size:11.5px;color:var(--faint);margin-bottom:3px}.msg.user .who{display:none}
.msg.err .body{color:var(--err)}
.think{color:var(--muted);background:var(--s2);border-radius:8px;padding:8px 12px;margin-bottom:8px;font-size:13.5px}
.foot{display:flex;gap:10px;align-items:center;margin-top:8px;font:11px/1 "Cascadia Mono",Consolas,monospace;color:var(--faint)}
.foot button{font:12px/1 inherit;font-family:"Segoe UI",system-ui,sans-serif;color:var(--muted);padding:4px 6px;border-radius:6px}
form{display:flex;gap:8px;align-items:flex-end;margin:0 auto 14px;max-width:780px;width:calc(100% - 36px);padding:6px;background:var(--panel);border-radius:18px;box-shadow:var(--sh2),0 0 0 1px var(--line);transition:box-shadow .2s ease}
form:focus-within{box-shadow:var(--sh2),0 0 0 1px var(--line2),0 0 0 4px var(--sigs)}
textarea{flex:1;resize:none;height:56px;border:0;background:transparent;outline:none;padding:9px 10px;line-height:1.5}
.btn{border-radius:9px;height:34px;padding:0 14px;background:var(--text);color:var(--panel);font-weight:500;font-size:13.5px;transition:background-color .15s ease,transform .16s var(--eo)}
.btn:active{transform:scale(.97)}.btn:disabled{opacity:.4;cursor:default;transform:none}
.btn.ghost{background:var(--panel);color:var(--text);box-shadow:var(--sh1),0 0 0 1px var(--line2)}
#status{color:var(--muted);font:11.5px/1 "Cascadia Mono",Consolas,monospace}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.msg.user .chips{justify-content:flex-end}
.chip{display:inline-flex;align-items:center;gap:6px;max-width:280px;height:26px;padding:0 9px;border-radius:999px;background:var(--panel);box-shadow:0 0 0 1px var(--line);font:12px/1 "Cascadia Mono",Consolas,monospace;color:var(--ink2)}
.chip span{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.chip small{color:var(--faint)}.chip.bad{color:var(--err)}.chip.busy::before{content:"";width:9px;height:9px;border-radius:50%;border:1.5px solid var(--sigs);border-top-color:var(--sig);animation:spin .7s linear infinite}
.chip button{font-size:13px;line-height:1;color:var(--faint);padding:0 2px}
@keyframes spin{to{transform:rotate(360deg)}}
.tile{position:relative;margin:0;width:64px;height:64px;border-radius:10px;overflow:hidden;background:var(--s2);box-shadow:0 0 0 1px var(--line);flex:none}
.msg .tile{width:auto;max-width:260px;height:180px;border-radius:12px}
.tile img{display:block;width:100%;height:100%;object-fit:cover;cursor:zoom-in}
.tile .x{position:absolute;top:3px;right:3px;width:20px;height:20px;border-radius:50%;background:rgba(21,20,18,.6);color:#fff;font-size:11px}
.tile .play{position:absolute;left:50%;top:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;background:rgba(21,20,18,.55);pointer-events:none}
.tile .play::after{content:"";position:absolute;left:11px;top:8px;border-style:solid;border-width:7px 0 7px 10px;border-color:transparent transparent transparent #fff}
.card{display:flex;align-items:center;gap:9px;min-width:170px;max-width:270px;padding:7px 10px 7px 7px;border-radius:12px;background:var(--panel);box-shadow:var(--sh1),0 0 0 1px var(--line)}
.card .ty{flex:none;width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font:600 10px/1 "Cascadia Mono",Consolas,monospace;color:#fff;background:var(--ink2)}
.card .mt{display:grid;min-width:0;flex:1}.card b{font-size:13px;font-weight:500;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.card small{font-size:11.5px;color:var(--muted)}
.card.bad small{color:var(--err)}.card.busy small::before{content:"";display:inline-block;width:8px;height:8px;margin-right:5px;border-radius:50%;border:1.5px solid var(--sigs);border-top-color:var(--sig);animation:spin .7s linear infinite}
.card .x{color:var(--faint);font-size:12px;padding:0 2px}
.msg.user.files{background:none;padding:0}.msg.user .chips{margin:0 0 6px}
dialog.zoom{margin:auto;border:0;background:transparent;max-width:min(1100px,calc(100vw - 48px));overflow:visible;text-align:center}
dialog.zoom::backdrop{background:rgba(21,20,18,.72)}dialog.zoom img{max-width:100%;max-height:calc(100vh - 110px);border-radius:12px;box-shadow:var(--sh3)}
dialog.zoom p{color:#fff;opacity:.8;font-size:12.5px;margin-top:10px}
.attach{width:34px;height:34px;flex:none;border-radius:9px;display:inline-flex;align-items:center;justify-content:center;color:var(--muted);transition:background-color .15s ease,transform .16s var(--eo)}.attach:active{transform:scale(.94)}
.attach svg{width:19px;height:19px;fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}
.composer{margin:0 auto 14px;max-width:780px;width:calc(100% - 36px)}.composer form{margin:0;width:100%}#pending:empty{display:none}#pending{margin:0 4px 8px}
.drop{display:none;position:fixed;inset:14px;border-radius:20px;background:rgba(245,243,238,.9);box-shadow:inset 0 0 0 2px var(--sig);align-items:center;justify-content:center;font-size:16px;font-weight:500;color:var(--sig);z-index:9;pointer-events:none}body.dragging .drop{display:flex}
.switch{display:flex;align-items:center;gap:6px;cursor:pointer}.switch input{appearance:none;width:26px;height:16px;border-radius:999px;background:var(--s3);position:relative;cursor:pointer;transition:background-color .2s ease}.switch input::after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.2);transition:transform .2s var(--eo)}.switch input:checked{background:var(--text)}.switch input:checked::after{transform:translateX(10px)}
@media (hover:hover) and (pointer:fine){.attach:hover{background:rgba(21,20,18,.05);color:var(--text)}.chip button:hover{color:var(--text)}.item:hover{background:rgba(255,255,255,.55)}.item:hover i{opacity:1}.item i:hover{color:var(--text)}.new:hover,.btn.ghost:hover{background:var(--s2)}.btn:not(.ghost):hover{background:#000}.back:hover,.foot button:hover{background:rgba(21,20,18,.05);color:var(--text)}}
@media (max-width:760px){body{grid-template-columns:1fr}.side{position:fixed;inset:0 auto 0 0;width:min(290px,86vw);z-index:5;transform:translateX(-100%);transition:transform .3s cubic-bezier(.32,.72,0,1)}.side.open{transform:none;box-shadow:var(--sh3)}.menu{display:inline-flex}label.opt{display:none}.msg.user{max-width:90%}}
@media (prefers-reduced-motion:reduce){*{transition-duration:.01ms!important}}
</style></head><body>
<aside class="side" id="side">
 <div class="side__top"><a class="brand" href="/simple"><span class="mark"></span>zorlo</a>
  <button class="new" id="newchat" type="button">+ New chat <kbd>Ctrl K</kbd></button>
  <input class="search" id="find" type="search" placeholder="Search chats"></div>
 <nav class="chats" id="chats" aria-label="Chats"></nav>
 <div class="side__bottom"><a class="back" href="/">&larr; Back to the agent</a></div>
</aside>
<div class="main">
<header><button class="menu" id="menu" type="button" aria-label="Show chats">&#9776;</button><h1 id="title">New chat</h1>
<label>Model <select id="model"></select></label>
<label class="opt">Max tokens <input id="maxtok" type="number" value="1024" min="1"></label>
<label class="switch" title="Let the model think first (slower)"><input type="checkbox" id="think">Think</label>
<label class="opt">Temperature <input id="temp" type="number" value="0.6" min="0" max="2" step="0.1"></label>
<span id="status">connecting...</span></header>
<div id="log"><div class="inner" id="inner"></div></div>
<div class="composer"><div class="chips" id="pending"></div>
<form id="f"><button class="attach" id="attach" type="button" title="Attach files: documents, PDFs, spreadsheets, pictures, code" aria-label="Attach files"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11.5l-7.6 7.6a5 5 0 0 1-7.1-7.1l8.1-8.1a3.3 3.3 0 0 1 4.7 4.7l-8.1 8.1a1.7 1.7 0 0 1-2.4-2.4l7.4-7.4"/></svg></button><input type="file" id="files" multiple hidden>
<textarea id="q" placeholder="Ask something, or attach a file. Enter sends, Shift+Enter for a new line."></textarea>
<button class="btn" id="send">Send</button><button class="btn ghost" id="stop" type="button" hidden>Stop</button></form></div>
<div class="drop">Drop files to attach</div>
<dialog class="zoom" id="zoom" onclick="this.close()"><img id="zoom-img" alt=""><p id="zoom-name"></p></dialog>
</div>
)HTML" R"HTML(<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uid=()=>Math.random().toString(36).slice(2,10);
const KEY='zorlo.simple.chats';
let chats=[],cur=null,att=[];const runs=new Map();   // chat id -> AbortController: a chat keeps answering while you read another
try{chats=JSON.parse(localStorage.getItem(KEY)||'[]')}catch{chats=[]}
const TOKEN=(document.querySelector('meta[name="zorlo-token"]')||{}).content||'';
const TEXT_EXT=/\.(txt|md|markdown|csv|tsv|json|jsonl|xml|ya?ml|toml|ini|cfg|conf|log|html?|css|scss|js|mjs|cjs|jsx|ts|tsx|vue|py|rb|php|java|kt|go|rs|c|h|cc|cpp|hpp|cs|swift|sh|ps1|bat|cmd|sql|r|lua|dart|env|srt|vtt|tex|rtf)$/i;
const FILE_MAX=40000;   // characters of one file kept with the message: the model reads about 40,000 in all
const kb=n=>n<1024?n+' B':n<1048576?(n/1024).toFixed(0)+' KB':(n/1048576).toFixed(1)+' MB';
// a file becomes text the model can read: text files as they are; documents, PDFs and pictures through Zorlo (OCR for pictures)
async function readFile(f){
  if(f.size<=2e6&&(f.type.startsWith('text/')||TEXT_EXT.test(f.name)||f.type==='application/json'))return {kind:'text',text:await f.text()};
  if(!TOKEN||TOKEN.startsWith('{{'))throw new Error('reading this kind of file needs Zorlo on this PC (127.0.0.1)');
  if(f.size>200e6)throw new Error('too big (over 200 MB)');
  const r=await fetch('/agent/extract?name='+encodeURIComponent(f.name),{method:'POST',headers:{'X-Zorlo-Token':TOKEN},body:f});
  const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j;
}
function describe(x){const bits=[];if(x.pages)bits.push(x.pages+(x.kind==='pptx'?' slides':x.kind==='xlsx'?' sheets':' pages'));
  if(x.width)bits.push(x.width+'x'+x.height);if(x.duration_s)bits.push(Math.round(x.duration_s)+' s');return bits.join(', ')}
// a small copy of a picture (or a video's first frame), kept with the chat so it shows after a reload
async function thumb(f,max=640){
  try{let src;
    if(f.type.startsWith('video/')){const v=document.createElement('video');v.muted=true;v.preload='auto';v.src=URL.createObjectURL(f);
      await new Promise((ok,no)=>{v.onloadeddata=ok;v.onerror=no;setTimeout(no,8000)});v.currentTime=Math.min(1,(v.duration||2)/3);
      await new Promise(ok=>{v.onseeked=ok;setTimeout(ok,3000)});src=v}
    else src=await createImageBitmap(f);
    const w=src.videoWidth||src.width,h=src.videoHeight||src.height,k=Math.min(1,max/Math.max(w,h)),c=document.createElement('canvas');
    c.width=Math.max(1,Math.round(w*k));c.height=Math.max(1,Math.round(h*k));c.getContext('2d').drawImage(src,0,0,c.width,c.height);
    if(src.src)URL.revokeObjectURL(src.src);return c.toDataURL('image/jpeg',.82)}catch{return ''}
}
async function addFiles(list){
  for(const f of list){const a={id:uid(),name:f.name,size:f.size,busy:true,video:f.type.startsWith('video/')};att.push(a);renderPending();
    if(f.type.startsWith('image/')||a.video){a.thumb=await thumb(f);renderPending()}
    try{const x=await readFile(f);let t=x.text||'';const full=t.length;if(t.length>FILE_MAX)t=t.slice(0,FILE_MAX);
      Object.assign(a,{busy:false,kind:x.kind,note:x.note||'',info:describe(x),text:t,cut:full>FILE_MAX?full:0})}
    catch(e){Object.assign(a,{busy:false,error:e.message})}
    renderPending()}
}
// pictures and videos as thumbnails, other files as cards with their type
function chip(a,rm){
  const x=rm?`<button class="x" type="button" data-rm="${a.id}" aria-label="Remove ${esc(a.name)}">&#10005;</button>`:'';
  if(a.thumb)return `<figure class="tile" title="${esc(a.error||a.note||a.name)}"><img src="${a.thumb}" alt="${esc(a.name)}" data-zoom>${a.video?'<span class="play"></span>':''}${x}</figure>`;
  const ty=(a.name.split('.').pop()||'file').toUpperCase().slice(0,5);
  return `<span class="card${a.error?' bad':''}${a.busy?' busy':''}" title="${esc(a.error||a.note||a.name)}"><span class="ty">${esc(ty)}</span><span class="mt"><b>${esc(a.name)}</b><small>${esc(a.busy?'reading':a.error?'can\'t read this file':a.info||kb(a.size))}</small></span>${x}</span>`}
document.addEventListener('click',e=>{const i=e.target.closest('[data-zoom]');if(!i)return;$('zoom-img').src=i.src;$('zoom-name').textContent=i.alt;$('zoom').showModal()});
function renderPending(){$('pending').innerHTML=att.map(a=>chip(a,true)).join('');
  $('pending').querySelectorAll('[data-rm]').forEach(b=>b.onclick=()=>{att=att.filter(x=>x.id!==b.dataset.rm);renderPending()})}
// what the model gets for a user message: the words plus each file's text
function wire(m){
  if(m.role==='assistant')return {role:'assistant',content:split(m.content).ans};
  let t=m.content||'';
  for(const f of m.files||[]){
    if(f.error){t+=`\n\n[The user attached ${f.name}, which could not be read: ${f.error}]`;continue}
    t+=`\n\n<file name="${f.name}"${f.info?` info="${f.info}"`:''}>\n${f.text||''}${f.cut?`\n[... only the first ${FILE_MAX} of ${f.cut} characters]`:''}\n</file>`+(f.note?`\n[${f.note}]`:'');
  }
  return {role:'user',content:t};
}
// chats are saved by Zorlo (%APPDATA%\Zorlo\chats), so every browser and Zorlo Desktop show the same list;
// this browser keeps a copy too, for when Zorlo can't be reached
const sent={};let flushT=null;
const sig=c=>c.updated+':'+c.title+':'+c.messages.length;
async function flush(){flushT=null;if(!TOKEN||TOKEN.startsWith('{{'))return;
  for(const c of chats){if(!c.messages.length||sent[c.id]===sig(c)||runs.has(c.id))continue;
    try{const r=await fetch('/agent/chat?kind=simple',{method:'POST',headers:{'Content-Type':'application/json','X-Zorlo-Token':TOKEN},body:JSON.stringify(c)});if(r.ok)sent[c.id]=sig(c)}catch{}}}
async function sync(){if(!TOKEN||TOKEN.startsWith('{{'))return;
  let server;try{const r=await fetch('/agent/chats?kind=simple',{headers:{'X-Zorlo-Token':TOKEN}});if(!r.ok)return;server=(await r.json()).chats}catch{return}
  const byId=new Map(chats.map(c=>[c.id,c]));let changed=false;
  for(const x of server){const m=byId.get(x.id);if(!m){chats.push(x);sent[x.id]=sig(x);changed=true}
    else if((x.updated||0)>(m.updated||0)&&!runs.has(m.id)){Object.assign(m,x);sent[x.id]=sig(x);changed=true}else if(sig(m)===sig(x))sent[x.id]=sig(x)}
  flush();
  if(changed){chats.sort((a,b)=>b.updated-a.updated);if(!cur||!cur.messages.length){const f=chats.find(c=>c.messages.length);if(f&&!cur)cur=f}renderAll()}}
addEventListener('focus',()=>sync());
function save(){if(!flushT)flushT=setTimeout(flush,500);for(let k=80;k>0;k-=10){try{localStorage.setItem(KEY,JSON.stringify(chats.slice(0,k)));return}catch{}}}
let idle='connecting...';
const status=(t,c)=>{if(!c||c===cur)$('status').textContent=t;if(!c)idle=t};
async function loadModels(){
  try{const r=await fetch('/v1/models');const j=await r.json();
    $('model').innerHTML=j.data.map(m=>`<option>${esc(m.id)}</option>`).join('');
    if(cur&&cur.model&&j.data.some(m=>m.id===cur.model))$('model').value=cur.model;
    status('ready');
  }catch(e){status('server not reachable')}
}
// reasoning (Qwen3 <think>, gpt-oss analysis channel) shown apart from the answer
function split(t){
  let think='',ans=t,m;
  if((m=t.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)/))){think=m[1];ans=m[3]}
  else if(t.includes('<|channel|>')){
    const a=t.match(/<\|channel\|>analysis<\|message\|>([\s\S]*?)(<\|end\|>|$)/);
    const f=t.match(/<\|channel\|>final<\|message\|>([\s\S]*)$/);
    think=a?a[1]:'';ans=f?f[1]:'';
  }
  return {think:think.trim(),ans:ans.replace(/<\|[a-z_]+\|>/g,'').trim()};
}
)HTML" R"HTML(function newChat(){
  if(cur&&!cur.messages.length&&!runs.has(cur.id)){$('q').focus();return}
  cur={id:uid(),title:'New chat',messages:[],model:$('model').value||'',updated:Date.now()};
  chats.unshift(cur);renderAll();$('q').focus();
}
function open(c){cur=c;if(c.model&&[...$('model').options].some(o=>o.value===c.model))$('model').value=c.model;renderAll();$('side').classList.remove('open')}
function day(t){const d=new Date(t),n=new Date(),diff=Math.floor((new Date(n.toDateString())-new Date(d.toDateString()))/864e5);
  return diff<=0?'Today':diff===1?'Yesterday':diff<7?'This week':diff<31?'This month':'Older'}
function renderSide(){
  const q=$('find').value.trim().toLowerCase(),nav=$('chats');nav.innerHTML='';let last='',n=0;
  const list=[...chats].sort((a,b)=>b.updated-a.updated);
  for(const c of list){
    if(!c.messages.length&&c!==cur)continue;
    if(q&&!c.title.toLowerCase().includes(q)&&!c.messages.some(m=>(m.content||'').toLowerCase().includes(q)||(m.files||[]).some(f=>f.name.toLowerCase().includes(q))))continue;
    const g=day(c.updated);if(g!==last){nav.insertAdjacentHTML('beforeend',`<div class="day">${g}</div>`);last=g}
    const b=document.createElement('button');b.type='button';b.className='item'+(c===cur?' on':'');
    b.innerHTML=`<span>${esc(c.title)}</span>${runs.has(c.id)?'<i class="live" title="Answering"></i>':''}<i class="ren" title="Rename">&#9998;</i><i class="del" title="Delete">&#10005;</i>`;
    b.onclick=e=>{
      if(e.target.classList.contains('del')){if(!confirm('Delete this chat?'))return;if(runs.has(c.id))runs.get(c.id).abort();
        chats=chats.filter(x=>x!==c);if(TOKEN&&!TOKEN.startsWith('{{'))fetch('/agent/chat_delete?kind=simple',{method:'POST',headers:{'Content-Type':'application/json','X-Zorlo-Token':TOKEN},body:JSON.stringify({id:c.id})}).catch(()=>{});save();if(c===cur){cur=null;const nx=chats.find(x=>x.messages.length);nx?open(nx):newChat()}else renderSide();return}
      if(e.target.classList.contains('ren')){const t=prompt('Chat name',c.title);if(t&&t.trim()){c.title=t.trim().slice(0,80);save();renderAll()}return}
      open(c)};
    nav.appendChild(b);n++;
  }
  if(!n)nav.innerHTML=`<p class="empty-list">${q?'No chats match.':'Your chats appear here.'}</p>`;
}
function bubble(m,i){
  const d=document.createElement('div');d.className='msg '+m.role+(m.error?' err':'');
  d.innerHTML=`<div class="who">${m.role==='user'?'You':esc(m.model||'Zorlo')}</div><div class="think" hidden></div><div class="body"></div><div class="foot"></div>`;
  fill(d,m.content,m.error);
  if(m.files&&m.files.length)d.insertAdjacentHTML('beforeend',`<div class="chips">${m.files.map(f=>chip(f,false)).join('')}</div>`);
  if(m.role==='user'&&m.files&&m.files.length){d.prepend(d.querySelector('.chips'));if(!m.content){d.querySelector('.body').hidden=true;d.classList.add('files')}}
  if(m.role==='assistant'){const f=d.querySelector('.foot');
    f.innerHTML=(m.stats?`<span>${esc(m.stats)}</span>`:'')+'<button type="button" data-copy>Copy</button>'+(i===cur.messages.length-1?'<button type="button" data-regen>Regenerate</button>':'');
    f.querySelector('[data-copy]').onclick=e=>{navigator.clipboard.writeText(split(m.content).ans);e.target.textContent='Copied';setTimeout(()=>e.target.textContent='Copy',1200)};
    const rg=f.querySelector('[data-regen]');if(rg)rg.onclick=()=>{if(runs.has(cur.id))return;cur.messages.pop();save();ask(cur)}}
  return d;
}
function fill(d,text,error){
  if(error){d.querySelector('.body').textContent=text;return}
  const s=split(text||''),th=d.querySelector('.think');
  th.hidden=!s.think;th.textContent=s.think;d.querySelector('.body').textContent=s.ans||(s.think?'':'...');
}
function renderLog(){
  const box=$('inner');box.innerHTML='';
  if(!cur||!cur.messages.length){box.innerHTML='<div class="hello"><b>What can I help with?</b>Your chats are saved in this browser. Start a new one any time.</div>'}
  else cur.messages.forEach((m,i)=>box.appendChild(bubble(m,i)));
  const r=runs.get(cur&&cur.id);
  if(r&&r.live){r.el=bubble(r.live,-1);box.appendChild(r.el)}
  $('log').scrollTop=1e9;
}
function renderAll(){renderSide();renderLog();$('title').textContent=cur?cur.title:'New chat';const busy=cur&&runs.has(cur.id);
  $('send').hidden=busy;$('stop').hidden=!busy;status(busy?'working...':idle==='connecting...'?idle:'ready')}
async function ask(chat){
  const model=$('model').value,ctrl=new AbortController(),run={ctrl,live:{role:'assistant',content:'',model},el:null};
  runs.set(chat.id,run);chat.model=model;renderAll();
  let text='',think='',n=0,t0=0,err='';status('working...',chat);
  const show=()=>{const t=think?'<think>'+think+(text?'</think>'+text:''):text;run.live.content=t;
    if(chat===cur&&run.el&&run.el.isConnected){fill(run.el,t);const l=$('log');if(l.scrollHeight-l.scrollTop-l.clientHeight<120)l.scrollTop=1e9}};
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',signal:ctrl.signal,headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model,messages:chat.messages.filter(m=>!m.error).map(wire),stream:true,chat_template_kwargs:{enable_thinking:$('think').checked},max_tokens:+$('maxtok').value,temperature:+$('temp').value})});
    if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error?e.error.message:r.statusText)}
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const {value,done}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});let i;
      while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
        if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
        const j=JSON.parse(line.slice(6));if(j.error)throw new Error(j.error.message);
        const dl=j.choices[0].delta,c=dl.content,rc=dl.reasoning_content;
        if(rc){think+=rc;show()}
        if(c){if(!n)t0=performance.now();n++;text+=c;show();
          const s=(performance.now()-t0)/1000;if(s>0.5)status(`generating - ${(n/s).toFixed(2)} tok/s`,chat)}
      }}
  }catch(e){if(e.name!=='AbortError')err=e.message}
  const s=t0?(performance.now()-t0)/1000:0;
  runs.delete(chat.id);
  if(text||think)chat.messages.push({role:'assistant',model,content:think?'<think>'+think+'</think>'+text:text,stats:n?`${(n/Math.max(s,1e-3)).toFixed(2)} tok/s - ${s.toFixed(1)} s`:''});
  if(err)chat.messages.push({role:'assistant',model,content:'Error: '+err,error:true});
  chat.updated=Date.now();save();
  if(chat===cur)renderAll();else renderSide();
  status('ready',chat);
}
function send(q){
  if(!cur)newChat();if(runs.has(cur.id))return;
  if(att.some(a=>a.busy)){status('still reading the attached files…');return}
  const files=att.map(({id,busy,...f})=>f);att=[];renderPending();
  cur.messages=cur.messages.filter(m=>!m.error);
  cur.messages.push(files.length?{role:'user',content:q,files}:{role:'user',content:q});
  if(cur.title==='New chat')cur.title=(q||files[0].name).replace(/\s+/g,' ').slice(0,48);
  cur.updated=Date.now();save();ask(cur);
}
$('f').onsubmit=e=>{e.preventDefault();const q=$('q').value.trim();if(!q&&!att.length)return;if(att.some(a=>a.busy)){status('still reading the attached files…');return}$('q').value='';send(q)};
$('attach').onclick=()=>$('files').click();
$('files').onchange=e=>{addFiles([...e.target.files]);e.target.value=''};
$('q').addEventListener('paste',e=>{const fs=[...(e.clipboardData&&e.clipboardData.files||[])];if(fs.length){e.preventDefault();addFiles(fs)}});
let depth=0;
addEventListener('dragenter',e=>{if(e.dataTransfer&&[...e.dataTransfer.types].includes('Files')){depth++;document.body.classList.add('dragging')}});
addEventListener('dragleave',()=>{if(--depth<=0){depth=0;document.body.classList.remove('dragging')}});
addEventListener('dragover',e=>e.preventDefault());
addEventListener('drop',e=>{e.preventDefault();depth=0;document.body.classList.remove('dragging');if(e.dataTransfer.files.length)addFiles([...e.dataTransfer.files])});
$('q').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('f').requestSubmit()}};
$('stop').onclick=()=>{const r=cur&&runs.get(cur.id);if(r)r.ctrl.abort()};
$('newchat').onclick=newChat;
$('find').oninput=renderSide;
$('menu').onclick=()=>$('side').classList.toggle('open');
addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();newChat()}});
$('model').onchange=()=>{if(cur&&cur.messages.length&&!runs.has(cur.id)){cur=null;newChat()}else if(cur)cur.model=$('model').value;status('next message uses '+$('model').value)};
cur=chats.find(c=>c.messages.length)||null;
if(!cur)newChat();else renderAll();
loadModels();sync();
</script></body></html>
)HTML";
