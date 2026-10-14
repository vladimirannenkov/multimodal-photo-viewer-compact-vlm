// Виртуализированная сетка: рисуются только видимые строки, данные подгружаются страницами
import { S, emit, on } from './state.js'
import { $, h, api, icons, KIND_RU, menu, toast, copyText } from './util.js'
import * as actions from './actions.js'

const PAGE = 300
const wrap = $('#gridWrap'), grid = $('#grid'), empty = $('#empty')
let total = 0, pages = new Map(), loading = new Set(), cols = 1, cell = 190, gap = 10, rowH = 190, cellW = 190
let lastClick = null, gen = 0
const nodes = new Map() // index -> элемент

export const getTotal = () => total
export function itemAt(i) { const p = pages.get(Math.floor(i / PAGE)); return p ? p[i % PAGE] : null }
export function loadedItems() { return [...pages.entries()].sort((a, b) => a[0] - b[0]).flatMap(([, p]) => p) }
export function indexOf(id) {
  for (const [pi, p] of pages) { const k = p.findIndex(x => x.id === id); if (k >= 0) return pi * PAGE + k }
  return -1
}
export const selectedIds = () => [...S.selection]

function params(offset) {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(S.filters)) if (v) q.set(k, v)
  if (S.hasText !== '') q.set('has_text', S.hasText)
  q.set('sort', S.sort); q.set('desc', S.desc); q.set('offset', offset); q.set('limit', PAGE)
  return q
}

export async function reload(keepScroll = false) {
  const g = ++gen
  const top = wrap.scrollTop
  if (S.search) {
    pages = new Map()
    const items = S.search.items
    for (let i = 0; i < items.length; i += PAGE) pages.set(i / PAGE, items.slice(i, i + PAGE))
    total = items.length
  } else {
    const r = await api('/images?' + params(0))
    if (g !== gen) return
    pages = new Map([[0, r.items]]); total = r.total
  }
  loading.clear()
  for (const n of nodes.values()) n.remove()
  nodes.clear()
  $('#count').textContent = total ? `${total.toLocaleString('ru-RU')} шт.` : ''
  layout()
  wrap.scrollTop = keepScroll ? top : 0
  render()
  emit('grid:loaded', { total })
}

function layout() {
  const w = wrap.clientWidth - 28
  cell = +$('#zoom').value
  cols = Math.max(1, Math.floor((w + gap) / (cell + gap)))
  cellW = Math.floor((w - gap * (cols - 1)) / cols)
  rowH = cellW + gap
  grid.style.height = Math.ceil(total / cols) * rowH + 'px'
  empty.hidden = total > 0
  if (!total) renderEmpty()
}

function renderEmpty() {
  empty.innerHTML = ''
  const noLib = (S.sidebar?.total || 0) === 0
  empty.append(h('div', { class: 'box' },
    h('h3', null, S.search ? 'Ничего не найдено' : noLib ? 'Библиотека пуста' : 'Здесь пока пусто'),
    h('p', { class: 'muted' }, S.search ? 'Попробуйте другие слова или уберите фильтры.' : noLib ? 'Добавьте папку с изображениями, перетащите файлы в окно или вставьте картинку из буфера через Ctrl+V.' : 'В этом разделе нет изображений.'),
    noLib && h('button', { class: 'btn primary', onClick: () => actions.openImport() }, 'Импортировать')))
}

function render() {
  const top = wrap.scrollTop, vh = wrap.clientHeight
  const r0 = Math.max(0, Math.floor(top / rowH) - 2), r1 = Math.ceil((top + vh) / rowH) + 2
  const i0 = r0 * cols, i1 = Math.min(total, r1 * cols)
  for (const [i, n] of nodes) if (i < i0 || i >= i1) { n.remove(); nodes.delete(i) }
  const need = new Set()
  for (let i = i0; i < i1; i++) {
    const it = itemAt(i)
    if (!it) { need.add(Math.floor(i / PAGE)); continue }
    let n = nodes.get(i)
    if (!n || n._id !== it.id) { n?.remove(); n = makeCell(it, i); nodes.set(i, n); grid.append(n) }
    n.style.width = n.style.height = cellW + 'px'
    n.style.transform = ''
    n.style.left = (i % cols) * (cellW + gap) + 'px'
    n.style.top = Math.floor(i / cols) * rowH + 'px'
  }
  for (const p of need) fetchPage(p)
}

async function fetchPage(p) {
  if (S.search || loading.has(p) || pages.has(p)) return
  loading.add(p)
  const g = gen
  try {
    const r = await api('/images?' + params(p * PAGE))
    if (g === gen) { pages.set(p, r.items); render() }
  } finally { loading.delete(p) }
}

function makeCell(it, i) {
  const n = h('div', { class: 'cell', title: it.name, role: 'button', tabIndex: -1 })
  n._id = it.id
  const img = h('img', { src: `/api/images/${it.id}/thumb`, loading: 'lazy', alt: it.name, draggable: false })
  img.onload = () => img.classList.add('ld')
  img.onerror = () => { img.classList.add('ld'); img.style.opacity = .2 }
  n.append(img, h('span', { class: 'ck', html: icons.check }))
  const b = h('div', { class: 'badges' })
  if (it.kind && it.kind !== 'photo') b.append(h('span', { class: 'bd' }, KIND_RU[it.kind] || it.kind))
  if (it.frames > 1) b.append(h('span', { class: 'bd' }, `${it.frames} кадр.`))
  n.append(b)
  if (it.favorite) n.append(h('span', { class: 'fav' }, '★'))
  if (it.status === 'new' || it.status === 'processing' || it.status === 'queued') n.append(h('span', { class: 'proc', title: 'Обрабатывается' }))
  applySel(n, it.id)
  n.addEventListener('click', e => onClick(e, it, i))
  n.addEventListener('dblclick', () => actions.openViewer(i))
  n.addEventListener('contextmenu', e => { e.preventDefault(); if (!S.selection.has(it.id)) selectOnly(it.id); actions.contextMenu(e.clientX, e.clientY, it) })
  n.draggable = true
  n.addEventListener('dragstart', e => { e.dataTransfer.setData('application/x-imagehub-ids', JSON.stringify(S.selection.has(it.id) ? [...S.selection] : [it.id])); e.dataTransfer.effectAllowed = 'copy' })
  return n
}

function applySel(n, id) {
  n.classList.toggle('sel', S.selection.has(id))
  n.classList.toggle('act', S.active === id)
}
export function refreshSel() {
  for (const n of nodes.values()) applySel(n, n._id)
  document.body.classList.toggle('selecting', S.selection.size > 0)
  emit('selection', S.selection)
}
export function selectOnly(id) { S.selection = new Set([id]); refreshSel() }

function onClick(e, it, i) {
  if (e.shiftKey && lastClick != null) {
    const [a, b] = [Math.min(lastClick, i), Math.max(lastClick, i)]
    for (let k = a; k <= b; k++) { const x = itemAt(k); if (x) S.selection.add(x.id) }
  } else if (e.ctrlKey || e.metaKey || e.target.closest('.ck')) {
    S.selection.has(it.id) ? S.selection.delete(it.id) : S.selection.add(it.id)
    lastClick = i
  } else {
    S.selection = new Set([it.id]); lastClick = i
  }
  setActive(it.id)
  refreshSel()
}

export function setActive(id) {
  if (S.active === id) return
  S.active = id
  for (const n of nodes.values()) applySel(n, n._id)
  emit('active', id)
}

export function scrollTo(i) {
  if (i < 0) return
  const y = Math.floor(i / cols) * rowH
  if (y < wrap.scrollTop) wrap.scrollTop = y
  else if (y + rowH > wrap.scrollTop + wrap.clientHeight) wrap.scrollTop = y + rowH - wrap.clientHeight
}
export function selectAll() {
  for (const it of loadedItems()) S.selection.add(it.id)
  refreshSel()
}
export function clearSel() { S.selection.clear(); refreshSel() }

export function patchItem(id, patch) {
  for (const p of pages.values()) { const it = p.find(x => x.id === id); if (it) Object.assign(it, patch) }
  for (const [i, n] of nodes) if (n._id === id) { n.remove(); nodes.delete(i) }
  render()
}

wrap.addEventListener('scroll', () => requestAnimationFrame(render), { passive: true })
new ResizeObserver(() => { if (total) { layout(); for (const n of nodes.values()) n.remove(); nodes.clear(); render() } }).observe(wrap)
$('#zoom').addEventListener('input', () => { layout(); render(); try { localStorage.setItem('zoom', $('#zoom').value) } catch (e) {} })
wrap.addEventListener('click', e => { if (e.target === wrap || e.target === grid) clearSel() })
try { const z = localStorage.getItem('zoom'); if (z) $('#zoom').value = z } catch (e) {}
on('refresh-grid', () => reload(true))
