// Правая панель: превью, описание, текст (raw / rendered), лица, чат с VLM, сведения о файле
import { S, emit, on } from './state.js'
import { $, h, api, toast, copyText, renderMd, fmtSize, fmtDate, KIND_RU, ask, icons } from './util.js'
import * as actions from './actions.js'
import * as grid from './grid.js'

const preview = $('#preview'), tabsEl = $('#tabs'), panel = $('#panel')
const TABS = [['desc', 'Описание'], ['text', 'Текст'], ['faces', 'Лица'], ['chat', 'Чат'], ['file', 'Файл']]
let descLang = (() => { try { return localStorage.getItem('descLang') || 'ru' } catch (e) { return 'ru' } })()
let tab = 'desc', d = null, timer = null, textMode = 'rendered', overlayBoxes = []

tabsEl.append(...TABS.map(([k, n]) => h('button', { class: 'tab' + (k === tab ? ' active' : ''), 'data-k': k, onClick: () => setTab(k) }, n)))
function setTab(k) { tab = k; $$tabs(); draw() }
const $$tabs = () => tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.k === tab))

export async function load(id) {
  clearTimeout(timer)
  if (id == null) { d = null; preview.innerHTML = ''; panel.innerHTML = ''; panel.append(h('div', { class: 'muted small' }, 'Выберите изображение, чтобы увидеть описание, текст, лица и задать вопрос по картинке.')); return }
  try { d = await api('/images/' + id) } catch (e) { d = null; return }
  if (S.active !== id) return
  drawPreview()
  draw()
  if (['new', 'queued', 'processing'].includes(d.status)) {
    api(`/images/${id}/prioritize`, { method: 'POST' }).catch(() => {})
    timer = setTimeout(() => S.active === id && load(id), 2500)
  }
}

function drawPreview() {
  preview.innerHTML = ''
  const wrap = h('div', { style: { position: 'relative' } })
  const img = h('img', { class: 'pv', src: `/api/images/${d.id}/thumb`, alt: d.name, role: 'button', title: 'Открыть', onClick: () => actions.openViewer(Math.max(0, grid.indexOf(d.id))) })
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', `0 0 ${d.width} ${d.height}`)
  svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none'
  wrap.append(img, svg)
  wrap._svg = svg
  preview.append(wrap,
    h('div', { class: 'nm' }, d.name),
    h('div', { class: 'row small muted' }, `${d.width}×${d.height}`, '·', fmtSize(d.size), d.kind && h('span', { class: 'chip accent' }, KIND_RU[d.kind] || d.kind),
      h('span', { class: 'spacer' }),
      h('button', { class: 'btn icon ghost sm', title: 'Избранное', onClick: async () => { await actions.toggleFavorite([d.id]); load(d.id) } }, d.favorite ? '★' : '☆'),
      h('button', { class: 'btn icon ghost sm', title: 'Копировать как путь', onClick: () => copyText(d.path) }, '⧉'),
      h('button', { class: 'btn icon ghost sm', title: 'Переименовать', onClick: () => actions.renameImage(d.id) }, '✎')))
  drawOverlay()
}

function drawOverlay() {
  const svg = preview.firstChild?._svg
  if (!svg) return
  svg.innerHTML = ''
  const sw = Math.max(d.width, d.height) / 120
  overlayBoxes.forEach(b => {
    const [x1, y1, x2, y2] = b.box
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    r.setAttribute('x', x1); r.setAttribute('y', y1); r.setAttribute('width', x2 - x1); r.setAttribute('height', y2 - y1)
    r.setAttribute('fill', 'rgba(138,138,242,.18)'); r.setAttribute('stroke', '#8a8af2'); r.setAttribute('stroke-width', sw)
    svg.append(r)
  })
}
export function highlight(boxes) { overlayBoxes = boxes || []; drawOverlay(); emit('overlay', overlayBoxes) }

function draw() {
  if (!d) return
  panel.innerHTML = ''
  ;({ desc: drawDesc, text: drawText, faces: drawFaces, chat: drawChat, file: drawFile })[tab]()
}

const chip = (t, cls = '', extra = {}) => h('span', { class: 'chip ' + cls, ...extra }, t)

function drawDesc() {
  const busy = ['new', 'queued', 'processing'].includes(d.status)
  if (busy) panel.append(h('div', { class: 'card' }, h('div', { class: 'row' }, h('span', { class: 'proc', style: { position: 'static', borderColor: 'var(--accent)', borderTopColor: 'transparent' } }), 'Изображение в обработке'), h('div', { class: 'bar', style: { marginTop: '8px' } }, h('i', { style: { width: '40%' } }))))
  if (d.error) panel.append(h('div', { class: 'card', style: { borderColor: 'var(--danger)' } }, h('b', { style: { color: 'var(--danger)' } }, 'Ошибки обработки'), h('div', { class: 'small' }, d.error)))
  if (d.summary) {
    const langs = [['ru', 'Русский', d.summary_ru], ['en', 'English', d.summary]].filter(l => l[2])
    if (!langs.some(l => l[0] === descLang)) descLang = langs[0][0]
    panel.append(h('div', { class: 'card' },
      h('div', { class: 'row' }, h('h4', { style: { margin: 0 } }, 'Описание'), h('span', { class: 'spacer' }),
        langs.length > 1 && h('div', { class: 'seg' }, langs.map(([k, n]) => h('button', { class: descLang === k ? 'on' : '', onClick: () => { descLang = k; try { localStorage.setItem('descLang', k) } catch (e) {} draw() } }, n)))),
      h('div', { class: 'small', style: { whiteSpace: 'pre-wrap', marginTop: '8px' } }, langs.find(l => l[0] === descLang)[2])))
  }
  const meta = h('div', { class: 'row', style: { flexWrap: 'wrap' } })
  if (d.kind) meta.append(chip('тип: ' + (KIND_RU[d.kind] || d.kind), 'accent'))
  if (d.event) meta.append(chip('событие: ' + d.event, 'accent'))
  if (d.has_text) meta.append(chip('текст: ' + (d.text_lang || '')))
  if (meta.children.length) panel.append(h('div', { class: 'card' }, h('h4', null, 'Определено'), meta))
  if (d.objects?.length) {
    const counts = {}
    d.objects.forEach(o => (counts[o.label] = (counts[o.label] || 0) + 1))
    panel.append(h('div', { class: 'card' }, h('h4', null, `Объекты и координаты (${d.objects.length})`),
      d.objects.slice(0, 40).map(o => h('div', { class: 'obj', role: 'button', title: 'Показать на изображении',
        onMouseenter: () => highlight([o]), onMouseleave: () => highlight([]), onClick: () => actions.openViewer(Math.max(0, grid.indexOf(d.id))) },
        h('b', null, o.label), h('span', { class: 'faint mono xs' }, o.box.map(Math.round).join(', '))))))
  }
  if (d.regions?.length) panel.append(h('details', { class: 'card' }, h('summary', { class: 'small muted' }, `Регионы (${d.regions.length})`),
    d.regions.map(o => h('div', { class: 'obj', onMouseenter: () => highlight([o]), onMouseleave: () => highlight([]) }, o.label))))
  if (d.chart?.stats?.series?.length) panel.append(h('div', { class: 'card' }, h('h4', null, 'График: ' + (d.chart.title || '')),
    h('table', { class: 't' }, h('tr', null, ['Серия', 'Макс.', 'Плато'].map(x => h('th', null, x))),
      d.chart.stats.series.map(s => h('tr', null, h('td', null, s.name || '—'), h('td', null, (+s.max).toPrecision(4)), h('td', null, s.plateau != null ? (+s.plateau).toPrecision(4) : '—'))))))
  // категории
  const cats = h('div', { class: 'row', style: { flexWrap: 'wrap' } })
  d.categories.forEach(c => cats.append(chip([c.name, h('span', { class: 'x', role: 'button', title: 'Убрать', onClick: async e => { e.stopPropagation(); await api('/categories/assign', { json: { ids: [d.id], remove: [c.name] } }); actions.refreshSidebar(); load(d.id) } }, '✕')], c.source === 'auto' ? 'auto' : 'accent')))
  cats.append(h('button', { class: 'btn sm', onClick: () => actions.categoriesDialog([d.id]) }, '+ категория'))
  panel.append(h('div', { class: 'card' }, h('h4', null, 'Категории'), cats))
  const colls = h('div', { class: 'row', style: { flexWrap: 'wrap' } }, d.collections.map(c => chip(c.name)), h('button', { class: 'btn sm', onClick: () => actions.collectionDialog([d.id]) }, '+ коллекция'))
  panel.append(h('div', { class: 'card' }, h('h4', null, 'Коллекции'), colls))
  if (d.experiment) panel.append(h('div', { class: 'card' }, h('h4', null, 'Эксперимент'), chip(d.experiment.name, 'accent')))
}

function drawText() {
  const has = d.ocr_md || d.ocr_raw
  if (!has) {
    panel.append(h('div', { class: 'card' }, d.has_text === 0 ? 'Связный текст не найден.' : (['new', 'queued', 'processing'].includes(d.status) ? 'Текст ещё распознаётся…' : 'Текст не распознавался.'),
      d.text_sample && h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'Фрагмент: ' + d.text_sample)))
    return
  }
  const edited = d.ocr_raw !== d.ocr_md
  const modes = [['rendered', 'Отрендеренный'], ['raw', 'Исходный Markdown'], ...(edited ? [['orig', 'До правки']] : [])]
  panel.append(h('div', { class: 'row', style: { marginBottom: '8px', flexWrap: 'wrap' } },
    h('div', { class: 'seg' }, modes.map(([k, n]) => h('button', { class: textMode === k ? 'on' : '', onClick: () => { textMode = k; draw() } }, n))),
    h('span', { class: 'spacer' }), h('button', { class: 'btn sm', onClick: () => copyText(textMode === 'orig' ? d.ocr_raw : d.ocr_md) }, 'Копировать')))
  panel.append(h('div', { class: 'small muted', style: { marginBottom: '6px' } }, `Язык: ${d.text_lang || '—'}` + (d.text_rotation ? `, поворот ${d.text_rotation}°` : '') + (edited ? ', опечатки исправлены' : '')))
  if (textMode === 'rendered') panel.append(h('div', { class: 'md', html: renderMd(d.ocr_md) }))
  else panel.append(h('pre', { class: 'ocr-raw' }, textMode === 'orig' ? d.ocr_raw : d.ocr_md))
}

function drawFaces() {
  if (!d.faces.length) { panel.append(h('div', { class: 'card muted' }, 'Лица не найдены.')); }
  d.faces.forEach(f => panel.append(h('div', { class: 'face-row', onMouseenter: () => highlight([{ box: f.box }]), onMouseleave: () => highlight([]) },
    h('img', { src: `/api/faces/${f.id}/crop?size=96`, loading: 'lazy' }),
    h('div', { class: 'grow' }, h('b', null, f.person || 'без имени'), h('div', { class: 'xs faint' }, f.labeled ? 'имя задано' : 'имя не задано')),
    h('button', { class: 'btn sm', onClick: async () => { const n = await ask('Имя', 'Как зовут этого человека', f.labeled ? f.person : ''); if (n) { await api('/people/' + f.person_id, { method: 'PATCH', json: { name: n } }); actions.refreshAll(); load(d.id) } } }, 'Назвать'))))
  panel.append(h('button', { class: 'btn primary', style: { marginTop: '10px' }, onClick: () => { setTab('chat'); startFaceFlow() } }, 'Разметить лица в чате'))
}

// ---------- чат ----------
let faceCur = null
let warmed = false
async function drawChat() {
  if (!warmed) { warmed = true; api('/chat/warmup', { method: 'POST' }).catch(() => { warmed = false }) }
  const box = h('div', { id: 'chatBox' })
  const input = h('input', { type: 'text', placeholder: faceCur ? 'Имя, «да» или «не знаю»' : 'Вопрос по изображению…' })
  const send = async () => {
    const t = input.value.trim(); if (!t) return
    input.value = ''
    if (faceCur) return answerFace(t, false)
    box.append(h('div', { class: 'msg user' }, t))
    const wait = h('div', { class: 'msg assistant' }, 'Думаю… (ждём очередь GPU и загрузку модели)'); box.append(wait)
    box.scrollIntoView({ block: 'end' })
    try {
      const r = await fetch(`/api/images/${d.id}/chat/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: t }) })
      if (!r.ok) throw new Error(r.statusText)
      const rd = r.body.getReader(), dec = new TextDecoder()
      let acc = '', last = 0
      wait.classList.add('md')
      for (;;) { const { done, value } = await rd.read(); if (done) break; acc += dec.decode(value, { stream: true }); const now = performance.now(); if (now - last > 120) { wait.innerHTML = renderMd(acc); last = now; wait.scrollIntoView({ block: 'end' }) } }
      wait.innerHTML = renderMd(acc)
    } catch (e) { wait.textContent = 'Ошибка: ' + e.message }
  }
  panel.append(box, h('form', { id: 'chatForm', onSubmit: e => { e.preventDefault(); send() } }, input,
    h('button', { class: 'btn primary', type: 'submit' }, 'Отправить')),
    h('div', { class: 'row', style: { marginTop: '8px' } }, h('button', { class: 'btn sm', onClick: () => startFaceFlow() }, 'Разметить лица'),
      h('button', { class: 'btn sm ghost', onClick: async () => { faceCur = null; highlight([]); await api(`/images/${d.id}/chat`, { method: 'DELETE' }); draw() } }, 'Очистить чат')))
  window._chatBox = box
  const cur = d.id
  const hist = await api(`/images/${cur}/chat`)
  if (!d || d.id !== cur || tab !== 'chat') return  // пользователь уже перешёл к другому изображению
  hist.forEach(m => box.append(m.role === 'assistant' ? h('div', { class: 'msg assistant md', html: renderMd(m.content) }) : h('div', { class: 'msg ' + m.role }, m.content)))
  if (!hist.length) box.append(h('div', { class: 'msg sys' }, 'Спросите, что на картинке, найти текст, посчитать объекты или объяснить график. Отвечает Qwen3-VL-2B, первый ответ занимает дольше из-за загрузки модели'))
  if (faceCur) box.append(faceMsg())
  input.focus()
}

function faceMsg() {
  const f = faceCur
  const sug = f.suggestions
  const el = h('div', { class: 'msg faceq' }, h('div', { class: 'row' }, h('img', { src: `/api/faces/${f.face.face_id}/crop?size=120` }),
    h('div', null, h('b', null, 'Кто это?'), h('div', { class: 'small muted' }, `Рамка подсвечена на изображении. Этот человек встречается на ${f.other_images} фото`))),
    f.candidate && h('div', null, `Похоже на «${f.candidate.name}» (сходство ${Math.round(f.candidate.sim * 100)}%). Ответьте «да», если это он`),
    h('div', { class: 'row', style: { flexWrap: 'wrap' } },
      f.candidate && h('button', { class: 'chip accent', onClick: () => answerFace(f.candidate.name, false) }, 'да, ' + f.candidate.name),
      sug.known.map(n => h('button', { class: 'chip', onClick: () => answerFace(n, false) }, n)),
      sug.roles.map(n => h('button', { class: 'chip auto', title: 'Общий класс', onClick: () => answerFace(n, true) }, n)),
      h('button', { class: 'chip', onClick: () => answerFace('', false, true) }, 'не знаю')))
  return el
}

export async function startFaceFlow() {
  const r = await api('/faces/next')
  if (!r.face) { faceCur = null; toast('Все лица размечены'); if (tab === 'chat') draw(); return }
  faceCur = r
  if (S.active !== r.image.id) { S.active = r.image.id; emit('open-image', r.image.id); await load(r.image.id) }
  if (tab !== 'chat') setTab('chat'); else draw()
  highlight([{ box: r.face.box }])
}

async function answerFace(text, role, skip = false) {
  const f = faceCur
  let name = text
  if (!skip && /^(да|yes|ага|верно)$/i.test(text) && f.candidate) name = f.candidate.name
  try { await api(`/faces/${f.face.face_id}/label`, { json: skip ? { skip: true } : { name, role } }) } catch (e) { return toast(e.message, true) }
  window._chatBox?.append(h('div', { class: 'msg user' }, skip ? 'не знаю' : name))
  await actions.refreshSidebar()
  highlight([])
  await startFaceFlow()
}

function drawFile() {
  const kv = [['Имя', d.name], ['Путь', d.path], ['Размер', fmtSize(d.size)], ['Разрешение', `${d.width}×${d.height}`], ['Формат', (d.format || '').toUpperCase() + (d.frames > 1 ? `, кадров ${d.frames}` : '')],
    ['Снято', fmtDate(d.taken_at)], ['Изменён', fmtDate(d.mtime)], ['Добавлен', fmtDate(d.added_at)], ['Хеш', d.hash], ['Статус', d.status + (d.from_clipboard ? ', из буфера' : '')]]
  panel.append(h('dl', { class: 'kv card' }, kv.flatMap(([k, v]) => [h('dt', null, k), h('dd', { class: k === 'Хеш' || k === 'Путь' ? 'mono xs' : '' }, v ?? '—')])))
  if (d.timings) panel.append(h('div', { class: 'card' }, h('h4', null, 'Время стадий, с на изображение'), h('div', { class: 'row', style: { flexWrap: 'wrap' } }, Object.entries(d.timings).map(([k, v]) => chip(`${k} ${v}`)))))
  panel.append(h('div', { class: 'row' }, h('button', { class: 'btn', onClick: () => copyText(d.path) }, 'Копировать путь'), h('button', { class: 'btn', onClick: () => actions.reprocess([d.id]) }, 'Обработать заново'),
    h('a', { class: 'btn', href: `/api/images/${d.id}/original`, download: d.name }, 'Скачать оригинал')))
}

on('active', id => { faceCur = null; highlight([]); load(id) })
on('open-image', () => {})
on('refresh-grid', () => { if (S.active) setTimeout(() => load(S.active), 400) })
on('similar', () => {})
load(null)
