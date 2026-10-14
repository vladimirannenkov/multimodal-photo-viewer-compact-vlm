// Общие помощники: DOM, запросы, уведомления, меню, диалоги, Markdown + LaTeX
export const $ = (s, r = document) => r.querySelector(s)
export const $$ = (s, r = document) => [...r.querySelectorAll(s)]

export function h(tag, props, ...kids) {
  const el = document.createElement(tag)
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue
    if (k === 'class') el.className = v
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v)
    else if (k === 'html') el.innerHTML = v
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v)
    else if (k in el && k !== 'list') el[k] = v
    else el.setAttribute(k, v === true ? '' : v)
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue
    el.append(kid.nodeType ? kid : document.createTextNode(kid))
  }
  return el
}

export const icons = {
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg>',
  all: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
  star: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg>',
  clip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="12" height="17" rx="2"/><path d="M9 4V3h6v1M9 11h6M9 15h4"/></svg>',
  tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1"/></svg>',
  coll: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M6 3h12M4 11h16v9a1 1 0 01-1 1H5a1 1 0 01-1-1z"/></svg>',
  flask: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 001.3 2.2h12.4a1.5 1.5 0 001.3-2.2L14 9V3"/></svg>',
  user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/></svg>',
  kind: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM17 13v7M13.5 16.5h7"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><path d="M5 12l5 5 9-10"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
}

export async function api(path, opts = {}) {
  const o = { ...opts }
  if (o.json !== undefined) {
    o.method = o.method || 'POST'
    o.headers = { 'Content-Type': 'application/json', ...(o.headers || {}) }
    o.body = JSON.stringify(o.json)
    delete o.json
  }
  const r = await fetch('/api' + path, o)
  if (!r.ok) {
    let msg = r.statusText
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j) } catch (e) {}
    throw new Error(msg)
  }
  const ct = r.headers.get('content-type') || ''
  return ct.includes('json') ? r.json() : r
}

export function toast(msg, err = false, ms = 3800) {
  const t = h('div', { class: 'toast' + (err ? ' err' : '') }, msg)
  $('#toasts').append(t)
  setTimeout(() => { t.style.transition = 'opacity .25s'; t.style.opacity = 0; setTimeout(() => t.remove(), 260) }, ms)
}

export function fmtSize(b) {
  if (b == null) return ''
  const u = ['Б', 'КБ', 'МБ', 'ГБ']; let i = 0
  while (b >= 1024 && i < 3) { b /= 1024; i++ }
  return (i ? b.toFixed(1) : b) + ' ' + u[i]
}
export function fmtDate(ts, time = true) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  const date = d.toLocaleDateString('ru-RU')
  return time ? date + ' ' + d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : date
}
export const KIND_RU = { photo: 'фото', screenshot: 'скриншот', document: 'документ', receipt: 'чек', chart: 'график', heatmap: 'карта',
  diagram: 'схема', formula: 'формула', table: 'таблица', meme: 'мем/рисунок' }

// ---------- контекстное меню ----------
let menuEl = null
export function closeMenu() { menuEl?.remove(); menuEl = null }
export function menu(items, x, y) {
  closeMenu()
  menuEl = h('div', { class: 'menu' }, items.map(it => it === '-' ? h('hr') :
    h('div', { class: 'mi' + (it.danger ? ' danger' : ''), onClick: () => { closeMenu(); it.run() } }, it.label, it.key && h('kbd', null, it.key))))
  document.body.append(menuEl)
  const r = menuEl.getBoundingClientRect()
  menuEl.style.left = Math.min(x, innerWidth - r.width - 8) + 'px'
  menuEl.style.top = Math.min(y, innerHeight - r.height - 8) + 'px'
}
addEventListener('pointerdown', e => { if (menuEl && !menuEl.contains(e.target)) closeMenu() }, true)
addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu() })

// ---------- диалоги ----------
export function dialog({ title, body, buttons = [], wide = false, onClose }) {
  const dlg = h('dialog', { style: wide ? { maxWidth: 'min(1180px, 96vw)' } : {} })
  const close = () => { dlg.close(); dlg.remove(); onClose?.() }
  dlg.append(
    h('div', { class: 'dlg-h' }, h('h3', null, title), h('button', { class: 'btn icon ghost', onClick: close, 'aria-label': 'Закрыть' }, '✕')),
    h('div', { class: 'dlg-b' }, body))
  if (buttons.length) dlg.append(h('div', { class: 'dlg-f' }, buttons.map(b => h('button', {
    class: 'btn ' + (b.cls || ''), onClick: async ev => { const r = await b.run?.(ev, close); if (b.close !== false && r !== false) close() },
  }, b.label))))
  dlg.addEventListener('cancel', e => { e.preventDefault(); close() })
  $('#dialogs').append(dlg)
  dlg.showModal()
  return { el: dlg, close }
}

export function ask(title, label, value = '', ok = 'Сохранить') {
  return new Promise(res => {
    const inp = h('input', { type: 'text', value, autofocus: true })
    const d = dialog({ title, body: h('label', { class: 'fld' }, h('span', null, label), inp),
      buttons: [{ label: 'Отмена', run: () => res(null) }, { label: ok, cls: 'primary', run: () => res(inp.value.trim() || null) }], onClose: () => res(null) })
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') { res(inp.value.trim() || null); d.close() } })
    setTimeout(() => { inp.focus(); inp.select() }, 30)
  })
}

export function confirmBox(title, text, ok = 'Удалить') {
  return new Promise(res => dialog({ title, body: h('p', null, text),
    buttons: [{ label: 'Отмена', run: () => res(false) }, { label: ok, cls: 'danger', run: () => res(true) }], onClose: () => res(false) }))
}

// ---------- Markdown + LaTeX ----------
let mdi = null
const ALLOWED = new Set(['TABLE', 'THEAD', 'TBODY', 'TFOOT', 'TR', 'TD', 'TH', 'BR', 'B', 'I', 'STRONG', 'EM', 'P', 'SPAN', 'SUP', 'SUB', 'CODE', 'PRE', 'UL', 'OL', 'LI', 'H1', 'H2', 'H3', 'H4', 'BLOCKQUOTE', 'HR', 'DIV', 'A'])
function sanitize(html) {
  const t = document.createElement('template')
  t.innerHTML = html
  const walk = n => {
    for (const c of [...n.children]) {
      if (!ALLOWED.has(c.tagName)) { c.replaceWith(document.createTextNode(c.textContent)); continue }
      for (const a of [...c.attributes]) if (!['colspan', 'rowspan', 'border', 'cellpadding', 'class', 'style'].includes(a.name) || /expression|url\(/i.test(a.value)) c.removeAttribute(a.name)
      walk(c)
    }
  }
  walk(t.content)
  return t.innerHTML
}
export function renderMd(text) {
  if (!text) return ''
  mdi = mdi || window.markdownit({ html: true, breaks: false, linkify: false })
  const maths = []
  const keep = (tex, display) => { maths.push([tex, display]); return `MATHPH${maths.length - 1}X` }
  let s = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => keep(t, true))
  s = s.replace(/\$([^$\n]+?)\$/g, (m, t) => /[\^_{}=]/.test(t) || (/^[a-zA-Zα-ωΑ-Ω]$/.test(t)) ? keep(t.trim(), false) : m)  // пробелы у $ допустимы, если внутри есть признаки формулы (деньги «$5 и $6» не трогаем)
  let html = sanitize(mdi.render(s))
  html = html.replace(/MATHPH(\d+)X/g, (_, i) => {
    const [tex, display] = maths[+i]
    try { return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'html' }) }
    catch (e) { return `<code>${tex.replace(/</g, '&lt;')}</code>` }
  })
  return html
}

export function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms) } }
export function copyText(t) { return navigator.clipboard.writeText(t).then(() => toast('Скопировано'), () => toast('Не удалось скопировать', true)) }
export function download(blob, name) {
  const a = h('a', { href: URL.createObjectURL(blob), download: name }); document.body.append(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 4000)
}
