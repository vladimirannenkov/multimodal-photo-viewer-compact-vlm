// Полноэкранный просмотр: панорама и масштаб, лупа, поворот, оверлеи (объекты, лица, текст), рисование, кадрирование, сравнение
import { S, emit, on } from './state.js'
import { $, h, api, toast, copyText, icons, ask } from './util.js'
import * as grid from './grid.js'
import * as actions from './actions.js'

const root = $('#viewer')
const NS = 'http://www.w3.org/2000/svg'
const COLORS = ['#e5484d', '#f5a524', '#30a46c', '#3e63dd', '#8e4ec6', '#ffffff', '#111111']
let st = null // состояние открытого просмотра

const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); return e }

export async function open(index) {
  const it = grid.itemAt(index)
  if (!it) return
  st = { index, id: it.id, d: null, s: 1, px: 0, py: 0, rot: 0, tool: 'pan', color: '#e5484d', wf: 1, shapes: [], drawing: null, crop: null,
    ov: { objects: false, faces: false, text: false }, loupe: false, cmp: null, hl: [], w: 1, h: 1 }
  root.hidden = false
  document.body.style.overflow = 'hidden'
  build()
  await show(index)
}

function close() {
  if (!st) return
  saveAnn.flush?.()
  root.hidden = true; root.innerHTML = ''; st = null
  document.body.style.overflow = ''
}

function build() {
  root.innerHTML = ''
  const btn = (label, title, fn, id) => h('button', { class: 'btn', title, id, onClick: fn }, label)
  const tool = (k, label, title) => h('button', { class: 'btn', 'data-tool': k, title, onClick: () => setTool(k) }, label)
  const bar = h('div', { id: 'vbar' },
    btn('✕', 'Закрыть (Esc)', close), h('span', { class: 'sep' }),
    btn('‹', 'Предыдущее (←)', () => nav(-1)), btn('›', 'Следующее (→)', () => nav(1)), h('span', { class: 'sep' }),
    btn('−', 'Уменьшить (-)', () => zoomBy(1 / 1.25)), btn('+', 'Увеличить (+)', () => zoomBy(1.25)), btn('По размеру', 'Вписать (0)', fit), btn('100%', 'Реальный размер (1)', () => zoomTo(1)),
    h('span', { class: 'sep' }), btn('↺', 'Повернуть против часовой (Shift+R)', () => rotate(-90)), btn('↻', 'Повернуть по часовой (R)', () => rotate(90)),
    btn('Копия…', 'Сохранить повёрнутую копию в каталог', saveCopy), h('span', { class: 'sep' }),
    btn('Лупа', 'Лупа (L)', () => { st.loupe = !st.loupe; sync() }, 'bLoupe'),
    btn('Объекты', 'Рамки объектов', () => toggleOv('objects'), 'bObjects'), btn('Лица', 'Рамки лиц', () => toggleOv('faces'), 'bFaces'), btn('Текст', 'Блоки распознанного текста', () => toggleOv('text'), 'bText'),
    h('span', { class: 'sep' }),
    tool('pan', '✋', 'Перемещение'), tool('rect', '▭', 'Рамка'), tool('ellipse', '◯', 'Овал'), tool('arrow', '↗', 'Стрелка'), tool('line', '╱', 'Линия'), tool('pen', '✎', 'Карандаш'), tool('text', 'T', 'Текст'),
    tool('crop', '⛶', 'Выделение для кадрирования'),
    h('input', { type: 'color', value: st.color, title: 'Цвет', onInput: e => { st.color = e.target.value } }),
    h('input', { type: 'range', min: 1, max: 8, value: 2, title: 'Толщина', onInput: e => { st.wf = +e.target.value / 2 } }),
    btn('↶', 'Отменить последний штрих (Ctrl+Z)', undo), btn('Разметка в копию', 'Разметка вшивается в новую копию', exportAnn), h('span', { class: 'sep' }),
    btn('Сравнить', 'Сравнить со следующим выделенным или соседним (C)', toggleCompare, 'bCmp'),
    btn('★', 'Избранное (F)', async () => { await actions.toggleFavorite([st.id]) }), btn('Копировать', 'Копировать изображение в буфер (Ctrl+C)', () => actions.copyImage(st.id)), btn('⧉', 'Копировать путь', () => actions.copyPath(st.id)))
  const stage = h('div', { id: 'vstage' },
    h('div', { id: 'vcanvas' }), h('button', { class: 'vnav l', onClick: () => nav(-1), 'aria-label': 'Назад' }, '‹'), h('button', { class: 'vnav r', onClick: () => nav(1), 'aria-label': 'Вперёд' }, '›'),
    h('div', { id: 'vinfo' }), h('div', { id: 'loupe', hidden: true }))
  root.append(bar, stage)
  bindStage(stage)
}

async function show(index) {
  const it = grid.itemAt(index)
  if (!it) return
  st.index = index; st.id = it.id; st.shapes = []; st.hist = []; st.redo = []; st.crop = null; st.cmp = null; st.rot = 0
  grid.setActive(it.id)
  const cv = $('#vcanvas')
  cv.innerHTML = ''
  const img = h('img', { src: `/api/images/${it.id}/file`, alt: it.name, draggable: false })
  const done = new Promise(r => { img.onload = r; img.onerror = r })
  cv.append(img)
  st.w = it.width || 1; st.h = it.height || 1
  const dp = api('/images/' + it.id).then(d => { st.d = d; st.shapes = d.annotations || [] })
  await Promise.all([done, dp])
  st.w = img.naturalWidth || st.w; st.h = img.naturalHeight || st.h
  cv.style.width = st.w + 'px'; cv.style.height = st.h + 'px'
  const svg = svgEl('svg', { id: 'vsvg', width: st.w, height: st.h, viewBox: `0 0 ${st.w} ${st.h}` })
  cv.append(svg)
  fit()
  drawSvg()
  sync()
  ;[1, -1].forEach(dlt => { const n = grid.itemAt(index + dlt); if (n) new Image().src = `/api/images/${n.id}/file` })
}

// ---------- геометрия ----------
const stageRect = () => $('#vstage').getBoundingClientRect()
function applyT() {
  const cv = $('#vcanvas')
  cv.style.transformOrigin = '50% 50%'
  cv.style.transform = `translate(${st.px - st.w / 2}px, ${st.py - st.h / 2}px) scale(${st.s}) rotate(${st.rot}deg)`
  const z = Math.round(st.s * 100)
  $('#vinfo').textContent = `${st.d?.name || ''}  ·  ${st.index + 1} из ${grid.getTotal()}  ·  ${st.w}×${st.h}  ·  ${z}%`
}
function fit() {
  const r = stageRect()
  const [bw, bh] = st.rot % 180 ? [st.h, st.w] : [st.w, st.h]
  st.s = Math.min(r.width / bw, r.height / bh, 3) * 0.96
  st.px = r.width / 2; st.py = r.height / 2
  applyT(); drawSvg()
}
function zoomAt(f, cx, cy) {
  const ns = Math.min(40, Math.max(0.03, st.s * f)), k = ns / st.s
  st.px = cx - (cx - st.px) * k; st.py = cy - (cy - st.py) * k; st.s = ns
  applyT(); drawSvg()
}
function zoomBy(f) { const r = stageRect(); zoomAt(f, r.width / 2, r.height / 2) }
function zoomTo(z) { const r = stageRect(); zoomAt(z / st.s, r.width / 2, r.height / 2) }
function rotate(d) { const from = st.rot; st.rot = (st.rot + d + 360) % 360; record({ k: 'rot', from, to: st.rot }); fit() }
function toImg(clientX, clientY) {
  const r = stageRect()
  const dx = (clientX - r.left - st.px) / st.s, dy = (clientY - r.top - st.py) / st.s
  const a = st.rot * Math.PI / 180
  return [dx * Math.cos(a) + dy * Math.sin(a) + st.w / 2, -dx * Math.sin(a) + dy * Math.cos(a) + st.h / 2]
}
async function nav(dlt) {
  const i = st.index + dlt
  if (i < 0 || i >= grid.getTotal()) return
  if (!grid.itemAt(i)) { toast('Загрузка…'); return }
  saveAnn.flush?.()
  await show(i)
}

// ---------- события ----------
function bindStage(stage) {
  stage.addEventListener('wheel', e => { e.preventDefault(); const r = stageRect(); zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top) }, { passive: false })
  stage.addEventListener('dblclick', e => { if (st.tool === 'pan') { Math.abs(st.s - 1) < 0.02 ? fit() : zoomAt(1 / st.s, e.clientX - stageRect().left, e.clientY - stageRect().top) } })
  stage.addEventListener('pointerdown', e => {
    if (e.target.closest('.vnav')) return
    stage.setPointerCapture(e.pointerId)
    const [ix, iy] = toImg(e.clientX, e.clientY)
    if (st.cmp && e.target.closest('#cmpHandle')) { st.cmpDrag = true; return }
    if (st.tool === 'pan') { st.pan = { x: e.clientX, y: e.clientY, px: st.px, py: st.py }; stage.classList.add('grabbing') }
    else if (st.tool === 'text') {
      ask('Надпись', 'Текст', '', 'Добавить').then(t => { if (t && st) { const sh = { type: 'text', at: [ix, iy], text: t, color: st.color, size: Math.round(Math.max(st.w, st.h) / 30 * st.wf) }; st.shapes.push(sh); record({ k: 'shape', s: sh }); drawSvg(); saveAnn() } })
    } else if (st.tool === 'crop') { st.cropPrev = st.crop && { a: [...st.crop.a], b: [...st.crop.b] }; st.crop = { a: [ix, iy], b: [ix, iy], drag: true } }
    else {
      const w = Math.max(2, Math.round(Math.max(st.w, st.h) / 280 * st.wf * 2))
      st.drawing = { type: st.tool === 'pen' ? 'freehand' : st.tool, color: st.color, width: w, a: [ix, iy], points: [[ix, iy]], box: [ix, iy, ix, iy] }
    }
  })
  stage.addEventListener('pointermove', e => {
    const [ix, iy] = toImg(e.clientX, e.clientY)
    if (st.loupe) moveLoupe(e)
    if (st.cmpDrag) { const r = stageRect(); st.cmp.pos = Math.min(1, Math.max(0, (e.clientX - r.left - (st.px - st.s * st.w / 2)) / (st.s * st.w))); applyCompare(); return }
    if (st.pan) { st.px = st.pan.px + e.clientX - st.pan.x; st.py = st.pan.py + e.clientY - st.pan.y; applyT(); return }
    if (st.crop?.drag) { st.crop.b = [ix, iy]; drawSvg(); return }
    const d = st.drawing
    if (!d) return
    if (d.type === 'freehand') d.points.push([ix, iy])
    else { d.box = [d.a[0], d.a[1], ix, iy]; d.points = [d.a, [ix, iy]] }
    drawSvg()
  })
  const end = e => {
    stage.classList.remove('grabbing')
    st.pan = null; st.cmpDrag = false
    if (st.crop?.drag) { st.crop.drag = false; record({ k: 'crop', from: st.cropPrev || null, to: { a: [...st.crop.a], b: [...st.crop.b] } }); drawSvg() }
    const d = st.drawing
    if (d) {
      st.drawing = null
      const ok = d.type === 'freehand' ? d.points.length > 2 : Math.hypot(d.box[2] - d.box[0], d.box[3] - d.box[1]) > 4
      if (ok) {
        const [x1, y1, x2, y2] = d.box
        const s = { type: d.type, color: d.color, width: d.width }
        if (d.type === 'rect' || d.type === 'ellipse') s.box = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]
        else s.points = d.points
        st.shapes.push(s); record({ k: 'shape', s }); saveAnn()
      }
      drawSvg()
    }
  }
  stage.addEventListener('pointerup', end)
  stage.addEventListener('pointercancel', end)
}

addEventListener('keydown', e => {
  if (!st || root.hidden) return
  if (e.target.matches('input[type=text], input[type=search], textarea')) return
  if (e.target.matches('input, select, button')) e.target.blur()  // фокус на ползунке или цвете не должен глушить стрелки
  const k = e.key, mod = e.ctrlKey || e.metaKey
  if (k === 'Escape') close()
  else if (mod && e.code === 'KeyZ') { e.preventDefault(); e.shiftKey ? redo() : undo() }
  else if (mod && e.code === 'KeyY') { e.preventDefault(); redo() }
  else if (mod && e.code === 'KeyC') actions.copyImage(st.id)
  else if (mod) return
  else if (k === 'ArrowLeft') nav(-1)
  else if (k === 'ArrowRight') nav(1)
  else if (k === '+' || k === '=') zoomBy(1.25)
  else if (k === '-') zoomBy(1 / 1.25)
  else if (k === '0') fit()
  else if (k === '1') zoomTo(1)
  else if (k.toLowerCase() === 'r') rotate(e.shiftKey ? -90 : 90)
  else if (k.toLowerCase() === 'l') { st.loupe = !st.loupe; sync() }
  else if (k.toLowerCase() === 'c') toggleCompare()
  else if (k.toLowerCase() === 'f') actions.toggleFavorite([st.id])
  else if (k === 'Delete') { const id = st.id; close(); actions.deleteImages([id]) }
  else return
  e.stopPropagation()
}, true)

// ---------- оверлеи и рисование ----------
function toggleOv(k) { st.ov[k] = !st.ov[k]; sync(); drawSvg() }
function setTool(k) { st.tool = k; st.crop = k === 'crop' ? st.crop : null; sync(); drawSvg() }
function sync() {
  if (!st) return
  const stage = $('#vstage')
  stage.classList.toggle('draw', !['pan', 'crop'].includes(st.tool)); stage.classList.toggle('crop', st.tool === 'crop')
  root.querySelectorAll('[data-tool]').forEach(b => b.classList.toggle('on', b.dataset.tool === st.tool))
  $('#bLoupe').classList.toggle('on', st.loupe); $('#bCmp').classList.toggle('on', !!st.cmp)
  ;['objects', 'faces', 'text'].forEach(k => $('#b' + k[0].toUpperCase() + k.slice(1)).classList.toggle('on', st.ov[k]))
  if (!st.loupe) $('#loupe').hidden = true
  applyT()
}

function drawSvg() {
  const svg = $('#vsvg')
  if (!svg || !st) return
  svg.innerHTML = ''
  const px = 1 / st.s // толщина линий оверлея не зависит от масштаба
  const lbl = (t, x, y) => { const e = svgEl('text', { x: x + 3 * px, y: y + 14 * px, 'font-size': 13 * px }); e.textContent = t; return e }
  const box = (b, cls = 'hl', label, pulse) => {
    const r = svgEl('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], class: cls + (pulse ? ' pulse' : ''), 'stroke-width': 2 * px, 'vector-effect': 'non-scaling-stroke' })
    svg.append(r)
    if (label) svg.append(lbl(label, b[0], b[1]))
  }
  const d = st.d
  if (d && st.ov.objects) (d.objects || []).forEach(o => box(o.box, 'hl', o.label))
  if (d && st.ov.faces) (d.faces || []).forEach(f => box(f.box, 'hl', f.person || 'лицо'))
  if (d && st.ov.text && !d.text_rotation) (d.ocr_blocks || []).forEach(b => b.box && box(b.box, 'hl', null))
  st.hl.forEach(b => box(b.box, 'hl', b.label, true))
  const all = [...st.shapes, ...(st.drawing ? [st.drawing] : [])]
  all.forEach(s => {
    const c = s.color, w = s.width
    const base = { stroke: c, 'stroke-width': w, fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }
    if (s.type === 'rect') svg.append(svgEl('rect', { ...base, x: s.box[0], y: s.box[1], width: s.box[2] - s.box[0], height: s.box[3] - s.box[1] }))
    else if (s.type === 'ellipse') svg.append(svgEl('ellipse', { ...base, cx: (s.box[0] + s.box[2]) / 2, cy: (s.box[1] + s.box[3]) / 2, rx: Math.abs(s.box[2] - s.box[0]) / 2, ry: Math.abs(s.box[3] - s.box[1]) / 2 }))
    else if (s.type === 'line') svg.append(svgEl('line', { ...base, x1: s.points[0][0], y1: s.points[0][1], x2: s.points.at(-1)[0], y2: s.points.at(-1)[1] }))
    else if (s.type === 'freehand') svg.append(svgEl('polyline', { ...base, points: s.points.map(p => p.join(',')).join(' ') }))
    else if (s.type === 'arrow') {
      const [x1, y1] = s.points[0], [x2, y2] = s.points.at(-1)
      const a = Math.atan2(y2 - y1, x2 - x1), L = Math.max(14, w * 4)
      svg.append(svgEl('line', { ...base, x1, y1, x2, y2 }))
      for (const da of [Math.PI - 0.45, Math.PI + 0.45]) svg.append(svgEl('line', { ...base, x1: x2, y1: y2, x2: x2 + L * Math.cos(a + da), y2: y2 + L * Math.sin(a + da) }))
    } else if (s.type === 'text') { const t = svgEl('text', { x: s.at[0], y: s.at[1] + s.size, 'font-size': s.size, fill: c, style: 'stroke:none;paint-order:normal;font-weight:600' }); t.textContent = s.text; svg.append(t) }
  })
  const c = st.crop
  if (c) {
    const x1 = Math.min(c.a[0], c.b[0]), y1 = Math.min(c.a[1], c.b[1]), x2 = Math.max(c.a[0], c.b[0]), y2 = Math.max(c.a[1], c.b[1])
    svg.append(svgEl('path', { d: `M0 0H${st.w}V${st.h}H0Z M${x1} ${y1}V${y2}H${x2}V${y1}Z`, fill: 'rgba(0,0,0,.55)', 'fill-rule': 'evenodd' }))
    svg.append(svgEl('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: 'none', stroke: '#fff', 'stroke-width': 2 * px, 'stroke-dasharray': `${6 * px} ${4 * px}` }))
    if (!c.drag && x2 - x1 > 8) cropBar([x1, y1, x2, y2])
  }
}

function cropBar(b) {
  let bar = $('#cropbar')
  if (!bar) {
    bar = h('div', { id: 'cropbar', class: 'row', style: { position: 'absolute', left: '50%', bottom: '56px', transform: 'translateX(-50%)', zIndex: 6, background: 'rgba(22,25,34,.92)', padding: '6px 10px', borderRadius: '999px' } })
    $('#vstage').append(bar)
  }
  bar.innerHTML = ''
  bar.append(h('span', { class: 'small' }, `${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])} px`),
    h('button', { class: 'btn primary sm', onClick: () => { actions.editDialog(st.id, [...(st.rot ? [{ type: 'rotate', deg: st.rot }] : []), { type: 'crop', box: b.map(Math.round) }]); } }, 'Сохранить копию'),
    h('button', { class: 'btn sm', onClick: () => { record({ k: 'crop', from: st.crop, to: null }); st.crop = null; bar.remove(); drawSvg() } }, 'Сбросить'))
}

// единая история: штрихи, поворот и выделение отменяются по Ctrl+Z, возвращаются по Ctrl+Y или Ctrl+Shift+Z
function record(a) { st.hist.push(a); st.redo = []; if (st.hist.length > 200) st.hist.shift() }
function applyStep(a, back) {
  if (a.k === 'shape') { if (back) st.shapes = st.shapes.filter(x => x !== a.s); else st.shapes.push(a.s); saveAnn() }
  else if (a.k === 'rot') { st.rot = back ? a.from : a.to; fit() }
  else if (a.k === 'crop') { st.crop = back ? a.from : a.to; $('#cropbar')?.remove(); if (st.crop) st.tool = 'crop'; sync() }
  drawSvg()
}
function undo() { const a = st.hist.pop(); if (!a) return toast('Нечего отменять'); applyStep(a, true); st.redo.push(a) }
function redo() { const a = st.redo.pop(); if (!a) return; applyStep(a, false); st.hist.push(a) }
const saveAnn = (() => {
  let t, pending = null
  const fn = () => { pending = { id: st.id, shapes: st.shapes.slice() }; clearTimeout(t); t = setTimeout(fn.flush, 600) }
  fn.flush = () => { clearTimeout(t); if (pending) { const p = pending; pending = null; api(`/images/${p.id}/annotations`, { method: 'PUT', json: { shapes: p.shapes } }).catch(() => {}) } }
  return fn
})()

async function exportAnn() {
  saveAnn.flush()
  if (!st.shapes.length) return toast('Нет разметки', true)
  await new Promise(r => setTimeout(r, 120))
  const r = await api(`/images/${st.id}/annotations/export`, { method: 'POST' })
  toast('Копия с разметкой добавлена в каталог'); actions.refreshAll()
}
function saveCopy() { st.rot ? actions.editDialog(st.id, [{ type: 'rotate', deg: st.rot }]) : toast('Сначала поверните изображение (R)') }

// ---------- лупа и сравнение ----------
function moveLoupe(e) {
  const l = $('#loupe'), r = stageRect(), z = 3
  const [ix, iy] = toImg(e.clientX, e.clientY)
  l.hidden = false
  l.style.left = e.clientX - r.left - 100 + 'px'; l.style.top = e.clientY - r.top - 100 + 'px'
  l.style.backgroundImage = `url(/api/images/${st.id}/file)`
  const scale = Math.max(1, st.s) * z / Math.max(1, st.s) // увеличение относительно оригинала не меньше 3x от экранного
  const k = Math.max(st.s, 1) * z
  l.style.backgroundSize = `${st.w * k}px ${st.h * k}px`
  l.style.backgroundPosition = `${100 - ix * k}px ${100 - iy * k}px`
}

function toggleCompare() {
  if (st.cmp) { st.cmp = null; $('#cmpImg')?.remove(); $('#cmpHandle')?.remove(); sync(); return }
  const sel = [...S.selection].filter(i => i !== st.id)
  const other = sel[0] ?? grid.itemAt(st.index + 1)?.id ?? grid.itemAt(st.index - 1)?.id
  if (!other) return toast('Нет второго изображения для сравнения', true)
  st.cmp = { id: other, pos: .5 }
  const cv = $('#vcanvas')
  cv.append(h('img', { id: 'cmpImg', src: `/api/images/${other}/file`, draggable: false, style: { position: 'absolute', left: 0, top: 0, width: st.w + 'px', height: st.h + 'px', objectFit: 'contain' } }))
  cv.append(h('div', { id: 'cmpHandle', style: { position: 'absolute', top: 0, bottom: 0, width: '26px', marginLeft: '-13px', cursor: 'ew-resize', zIndex: 3, background: 'linear-gradient(90deg, transparent 12px, #fff 12px, #fff 14px, transparent 14px)' } }))
  applyCompare(); sync()
  toast('Тяните вертикальную линию, слева текущее изображение, справа второе')
}
function applyCompare() {
  const c = st.cmp; if (!c) return
  $('#cmpImg').style.clipPath = `inset(0 0 0 ${c.pos * 100}%)`
  $('#cmpHandle').style.left = c.pos * 100 + '%'
}

on('open-viewer', open)
on('overlay', boxes => { if (st) { st.hl = boxes; drawSvg() } })
on('active', () => { if (st && !root.hidden && S.active && S.active !== st.id) { const i = grid.indexOf(S.active); if (i >= 0) show(i) } })
