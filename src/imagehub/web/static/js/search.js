// Поиск: текст, картинка-образец (файл, вставка, перетаскивание), смесь с весом, похожие изображения
import { S, emit, on } from './state.js'
import { $, h, api, toast, debounce } from './util.js'
import * as grid from './grid.js'
import * as sidebar from './sidebar.js'

const form = $('#searchForm'), q = $('#q'), imgBox = $('#searchImg'), weightBox = $('#weightBox'), weight = $('#weight')
let imgBlob = null, imgUrl = null

function updateUi() {
  imgBox.hidden = !imgBlob
  weightBox.hidden = !(imgBlob && q.value.trim())
}

export function setImage(blob) {
  imgBlob = blob
  if (imgUrl) URL.revokeObjectURL(imgUrl)
  imgUrl = blob ? URL.createObjectURL(blob) : null
  $('img', imgBox).src = imgUrl || ''
  updateUi()
  if (blob) run()
}

async function run() {
  const text = q.value.trim()
  if (!text && !imgBlob) return clear()
  const fd = new FormData()
  fd.append('q', text)
  fd.append('weight', weight.value)
  fd.append('filters', JSON.stringify({ ...S.filters, ...(S.hasText !== '' ? { has_text: S.hasText } : {}) }))
  if (imgBlob) fd.append('image', imgBlob, 'query.png')
  try {
    const r = await api('/search', { method: 'POST', body: fd })
    show(r.items, text ? `«${text}»` : 'по картинке', imgBlob && text ? `картинка ×${(1 - weight.value).toFixed(2)} + текст ×${(+weight.value).toFixed(2)}` : '')
  } catch (e) { toast('Ошибка поиска: ' + e.message, true) }
}

function show(items, label, note = '') {
  S.search = { items }
  $('#title').textContent = 'Поиск'
  const b = $('#banner')
  b.hidden = false
  b.innerHTML = ''
  b.append(h('span', null, `Найдено ${items.length}: ${label}`), note && h('span', { class: 'xs' }, note), h('span', { class: 'spacer' }),
    h('button', { class: 'btn sm', onClick: () => { q.value = ''; setImage(null); clear() } }, 'Сбросить поиск'))
  emit('refresh-grid')
}

function clear() {
  if (!S.search) return
  S.search = null
  $('#banner').hidden = true
  $('#title').textContent = S.title
  emit('refresh-grid')
}

form.addEventListener('submit', e => { e.preventDefault(); run() })
q.addEventListener('input', () => { updateUi(); if (!q.value.trim() && !imgBlob) clear() })
q.addEventListener('input', debounce(() => { if (q.value.trim().length >= 3) run() }, 450))
weight.addEventListener('input', debounce(run, 250))
$('#clearImg').addEventListener('click', () => { setImage(null); if (!q.value.trim()) clear(); else run() })
$('#btnPickImg').addEventListener('click', () => $('#searchFile').click())
$('#searchFile').addEventListener('change', e => { if (e.target.files[0]) setImage(e.target.files[0]); e.target.value = '' })
form.addEventListener('dragover', e => { e.preventDefault(); form.style.outline = '2px dashed var(--accent)' })
form.addEventListener('dragleave', () => { form.style.outline = '' })
form.addEventListener('drop', e => {
  e.preventDefault(); e.stopPropagation(); form.style.outline = ''
  const f = [...e.dataTransfer.files].find(x => x.type.startsWith('image/'))
  if (f) setImage(f)
  const raw = e.dataTransfer.getData('application/x-imagehub-ids')
  if (raw) { const id = JSON.parse(raw)[0]; similar(id) }
})
q.addEventListener('paste', e => {
  const f = [...(e.clipboardData?.files || [])].find(x => x.type.startsWith('image/'))
  if (f) { e.preventDefault(); e.stopPropagation(); setImage(f) }
})

export async function similar(id) {
  const fd = new FormData()
  fd.append('similar_to', id)
  fd.append('filters', '{}')
  const r = await api('/search', { method: 'POST', body: fd })
  q.value = ''
  show(r.items, `похожие на изображение #${id}`)
}
on('similar', similar)
on('search-cleared', () => { imgBlob = null; $('img', imgBox).src = ''; updateUi() })
on('active', () => {})
on('filters-changed', () => { if (S.search) run() })
