// Точка входа: связывает панели, горячие клавиши, перетаскивание, вставку из буфера и опрос очереди
import { S, emit, on } from './state.js'
import { $, h, api, toast, menu } from './util.js'
import * as grid from './grid.js'
import * as sidebar from './sidebar.js'
import * as panel from './panel.js'
import * as actions from './actions.js'
import { THEMES } from './themes.js'
import './viewer.js'
import './search.js'

const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d } catch (e) { return d } },
  set: (k, v) => { try { localStorage.setItem(k, v) } catch (e) {} },
}

// ---------- панели и тема ----------
const body = document.body
if (store.get('noLeft') === '1' || innerWidth < 900) body.classList.add('no-left')
if (store.get('noRight') === '1' || innerWidth < 900) body.classList.add('no-right')
$('#btnLeft').onclick = () => { body.classList.toggle('no-left'); store.set('noLeft', body.classList.contains('no-left') ? 1 : 0) }
$('#btnRight').onclick = () => { body.classList.toggle('no-right'); store.set('noRight', body.classList.contains('no-right') ? 1 : 0) }
$('#btnTheme').onclick = e => {
  const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  const r = e.currentTarget.getBoundingClientRect()
  menu(THEMES.map(([k, n]) => ({ label: (k === cur ? '● ' : '○ ') + n, run: () => { document.documentElement.dataset.theme = k; store.set('theme', k) } }))
    .concat(['-', { label: 'Галерея тем (ссылки)', run: () => window.open('/themes', '_blank') }]), r.right - 220, r.bottom + 6)
}

// ---------- тулбар ----------
const sortSel = $('#sort')
sortSel.value = store.get('sort', 'date'); S.sort = sortSel.value
S.desc = store.get('desc', '1') === '1'; $('#btnDesc').textContent = S.desc ? '↓' : '↑'
sortSel.onchange = () => { S.sort = sortSel.value; store.set('sort', S.sort); emit('refresh-grid') }
$('#btnDesc').onclick = e => { S.desc = !S.desc; e.target.textContent = S.desc ? '↓' : '↑'; store.set('desc', S.desc ? 1 : 0); emit('refresh-grid') }
$('#filterText').onchange = e => { S.hasText = e.target.value; emit('filters-changed'); S.search ? null : emit('refresh-grid') }
$('#btnImport').onclick = () => actions.openImport()
$('#btnTools').onclick = () => actions.toolsDialog()

// ---------- панель выбора ----------
const selbar = $('#selbar')
function drawSel() {
  const n = S.selection.size
  selbar.classList.toggle('show', n > 0)
  if (!n) return
  const ids = [...S.selection]
  const b = (t, title, fn, cls = '') => h('button', { class: 'btn ghost ' + cls, title, onClick: fn }, t)
  selbar.innerHTML = ''
  selbar.append(h('span', { class: 'small', style: { padding: '0 8px' } }, `Выбрано: ${n}`),
    b('★', 'Избранное (F)', () => actions.toggleFavorite(ids)), b('Категории', 'Категории', () => actions.categoriesDialog(ids)),
    b('Коллекция', 'В коллекцию', () => actions.collectionDialog(ids)), b('Экспорт', 'Экспорт и конвертация', () => actions.exportDialog(ids)),
    ...(n > 1 ? [b('Склеить N×M', 'Склейка в сетку', () => actions.mergeDialog(ids))] : []),
    b('Сводка', 'Сводка по выбранным', () => actions.toolsDialog()),
    b('Удалить', 'Удалить (Del)', () => actions.deleteImages(ids), 'danger'), b('✕', 'Снять выделение (Esc)', () => grid.clearSel()))
}
on('selection', drawSel)

// ---------- очередь обработки ----------
let lastPending = -1, lastDone = -1, qTimer = null
async function pollQueue() {
  try {
    const q = await api('/queue')
    S.queue = q
    const chip = $('#queueChip')
    chip.classList.toggle('busy', q.pending > 0 && !q.paused); chip.classList.toggle('paused', q.paused)
    const names = { embed: 'эмбеддинги', textdet: 'поиск текста', ocr: 'распознавание текста', fix: 'правка опечаток', scene: 'описание сцен', faces: 'лица', chart: 'графики', textemb: 'индекс', idle: '', paused: 'пауза' }
    $('#queueText').textContent = q.paused ? `пауза, в очереди ${q.pending}` : q.pending ? `${names[q.stage] || 'подготовка'}, осталось ${q.pending}` : 'всё обработано'
    $('#btnPause').textContent = q.paused ? '▶' : '⏸'
    $('#queueChip').title = `Устройство: ${q.device}` + (q.last_error ? `\nПоследняя ошибка: ${q.last_error}` : '')
    const done = q.counts.done || 0
    if (lastDone >= 0 && (done !== lastDone || (lastPending > 0 && q.pending === 0))) { actions.refreshSidebar(); emit('refresh-grid') }
    lastPending = q.pending; lastDone = done
  } catch (e) {}
  clearTimeout(qTimer)
  qTimer = setTimeout(pollQueue, (S.queue?.pending || 0) > 0 ? 2500 : 7000)
}
$('#btnPause').onclick = async () => { const r = await api('/queue/' + (S.queue?.paused ? 'resume' : 'pause'), { method: 'POST' }); S.queue = r; pollQueue() }

// ---------- перетаскивание файлов и вставка ----------
let dragDepth = 0
const hasFiles = e => [...(e.dataTransfer?.types || [])].includes('Files')
addEventListener('dragenter', e => { if (hasFiles(e)) { dragDepth++; $('#dropVeil').hidden = false } })
addEventListener('dragleave', e => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $('#dropVeil').hidden = true } })
addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault() })
addEventListener('drop', e => {
  dragDepth = 0; $('#dropVeil').hidden = true
  if (hasFiles(e)) { e.preventDefault(); actions.uploadFiles(e.dataTransfer.files) }
})
addEventListener('paste', e => {
  if (e.target.matches?.('input, textarea')) return
  const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith('image/'))
  if (files.length) { e.preventDefault(); actions.uploadFiles(files, true) }
})

// ---------- горячие клавиши ----------
addEventListener('keydown', e => {
  if ($('#viewer') && !$('#viewer').hidden) return
  if (document.querySelector('dialog[open]')) return
  if (e.target.matches?.('select, input[type=range], input[type=checkbox]') && e.key.startsWith('Arrow')) e.target.blur()
  const typing = e.target.matches?.('input[type=text], input[type=search], input[type=number], textarea')
  if (e.key === '/' && !typing) { e.preventDefault(); $('#q').focus(); return }
  if (typing) return
  const ids = grid.selectedIds()
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') { e.preventDefault(); grid.selectAll() }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c' && S.active && !getSelection().toString()) actions.copyImage(S.active)
  else if (e.key === 'Escape') grid.clearSel()
  else if (e.key === 'Delete' && ids.length) actions.deleteImages(ids)
  else if (e.key.toLowerCase() === 'f' && ids.length && !e.ctrlKey) actions.toggleFavorite(ids)
  else if (e.key === 'F2' && ids.length === 1) actions.renameImage(ids[0])
  else if (e.key === 'Enter' && S.active) actions.openViewer(grid.indexOf(S.active))
  else if (e.key.startsWith('Arrow') && grid.getTotal()) {
    e.preventDefault()
    const cur = S.active ? grid.indexOf(S.active) : -1
    const cols = Math.max(1, Math.floor(($('#gridWrap').clientWidth - 28 + 10) / (+$('#zoom').value + 10)))
    const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols }[e.key]
    const it = grid.itemAt(Math.min(grid.getTotal() - 1, Math.max(0, cur + d)))
    if (it) { grid.selectOnly(it.id); grid.setActive(it.id); grid.scrollTo(grid.indexOf(it.id)) }
  }
})

// ---------- запуск ----------
on('open-viewer', () => {})
;(async () => {
  await actions.refreshSidebar()
  sidebar.navigate('all', {}, 'Все изображения')
  pollQueue()
})()
