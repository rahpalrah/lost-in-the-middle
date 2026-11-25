"""The few HTML pages for self-service keys: register, verified (shows the key once), admin.

Plain server-rendered HTML with a little inline script; no frameworks, no external requests.
Everything dynamic is escaped (server side) or set with ``textContent`` (browser side).
"""

from __future__ import annotations

from html import escape

# Inline script and style are needed (no external files); everything else is locked down.
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'"
)

_STYLE = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#666;--bd:#d8d8d8;--ac:#0b57d0}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#aaa;--bd:#383838;--ac:#8ab4f8}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:32px 16px}
h1{font-size:1.5rem;margin:0 0 8px}p{color:var(--mut)}
input,button{font:inherit;padding:10px 12px;border:1px solid var(--bd);border-radius:8px;
background:var(--bg);color:var(--fg);width:100%;box-sizing:border-box}
button{background:var(--ac);color:#fff;border:0;cursor:pointer;margin-top:12px}
button.small{width:auto;padding:4px 10px;margin:0}
code,pre{font-family:ui-monospace,monospace;background:rgba(127,127,127,.15);border-radius:6px}
code{padding:2px 6px;word-break:break-all}pre{padding:12px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{text-align:left;padding:6px;border-bottom:1px solid var(--bd)}
.msg{margin-top:12px;min-height:1.5em}.err{color:#c5221f}.wide{max-width:1000px}
"""


def _page(title: str, body: str, script: str = "", *, wide: bool = False) -> str:
    js = f"<script>{script}</script>" if script else ""
    cls = ' class="wide"' if wide else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main{cls}>{body}</main>{js}</body></html>"
    )


def register_html() -> str:
    """The sign-up page."""
    body = (
        "<h1>Get an API key</h1>"
        "<p>Enter your work or personal email. We will send a one-time link that shows your key.</p>"
        '<form id="f"><input id="email" type="email" required autocomplete="email" '
        'placeholder="you@example.com" maxlength="254">'
        '<button type="submit">Send verification link</button></form>'
        '<div class="msg" id="msg" role="status"></div>'
    )
    script = """
const f=document.getElementById('f'),m=document.getElementById('msg');
f.addEventListener('submit',async e=>{e.preventDefault();m.className='msg';m.textContent='Sending...';
try{const r=await fetch('/v1/register',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({email:document.getElementById('email').value})});
const d=await r.json();
if(r.ok){m.textContent=d.message;if(d.verify_url){const a=document.createElement('a');
a.href=d.verify_url;a.textContent=' (dev) open link';m.appendChild(a);}}
else{m.className='msg err';m.textContent=d.message||'Something went wrong.';}}
catch(_){m.className='msg err';m.textContent='Could not reach the server.';}});
"""
    return _page("Register for an API key", body, script)


def verified_html(email: str, key: str, public_url: str) -> str:
    """The page that shows a new key once."""
    body = (
        "<h1>Your API key</h1>"
        f"<p>For <b>{escape(email)}</b>. <b>Copy it now: it is shown only once</b> and cannot be "
        "recovered. If you lose it, register again after the owner revokes it.</p>"
        f"<pre><code id='k'>{escape(key)}</code></pre>"
        "<h2 style='font-size:1.1rem'>Try it</h2>"
        f"<pre>curl -H 'Authorization: Bearer {escape(key)}' \\\n"
        f"  -F file=@contract.pdf -F id=my-doc {escape(public_url)}/v1/documents\n\n"
        f"curl -H 'Authorization: Bearer {escape(key)}' -H 'content-type: application/json' \\\n"
        '  -d \'{"question":"What is the notice period?"}\' \\\n'
        f"  {escape(public_url)}/v1/documents/my-doc/ask</pre>"
    )
    return _page("Your API key", body)


def error_html(message: str) -> str:
    """A failed-verification page."""
    return _page(
        "Verification failed",
        f"<h1>Verification failed</h1><p>{escape(message)}</p><p><a href='/register'>Register</a></p>",
    )


def admin_html() -> str:
    """The owner's account list (needs the admin token)."""
    body = (
        "<h1>Accounts</h1><p>Enter the admin token. It stays in this tab's memory only.</p>"
        '<input id="t" type="password" placeholder="Admin token" autocomplete="off">'
        '<button id="go">Load accounts</button><div class="msg" id="msg" role="status"></div>'
        '<table id="tbl" hidden><thead><tr><th>Email</th><th>Type</th><th>Status</th>'
        "<th>Registered</th><th>Key</th><th>Requests</th><th>Docs</th><th>Questions</th>"
        "<th>Last used</th><th></th></tr></thead><tbody></tbody></table>"
    )
    script = """
const t=document.getElementById('t'),m=document.getElementById('msg'),
tbl=document.getElementById('tbl'),tb=tbl.querySelector('tbody');
const H=()=>({authorization:'Bearer '+t.value});
function cell(tr,v){const td=document.createElement('td');td.textContent=v==null?'':String(v);tr.appendChild(td);}
async function load(){m.className='msg';m.textContent='Loading...';
const r=await fetch('/v1/admin/accounts',{headers:H()});
if(!r.ok){m.className='msg err';m.textContent=(await r.json()).message||'Failed';tbl.hidden=true;return;}
const rows=await r.json();tb.replaceChildren();
for(const a of rows){const tr=document.createElement('tr');
[a.email,a.kind,a.status,a.created_at,a.key_prefix?a.key_prefix+'...':'',a.requests,a.documents,
a.questions,a.last_used].forEach(v=>cell(tr,v));
const td=document.createElement('td');
if(a.status==='active'){const b=document.createElement('button');b.className='small';b.textContent='Revoke';
b.onclick=async()=>{if(!confirm('Revoke '+a.email+'?'))return;
await fetch('/v1/admin/accounts/'+encodeURIComponent(a.email)+'/revoke',{method:'POST',headers:H()});load();};
td.appendChild(b);}tr.appendChild(td);tb.appendChild(tr);}
tbl.hidden=false;m.textContent=rows.length+' account(s)';}
document.getElementById('go').onclick=load;
"""
    return _page("Admin", body, script, wide=True)
