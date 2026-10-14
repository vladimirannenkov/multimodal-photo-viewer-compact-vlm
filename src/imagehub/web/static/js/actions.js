// Действия над изображениями и диалоги (импорт, категории, коллекции, экспорт, склейка, инструменты)
import { S, emit } from './state.js'
import { $, h, api, toast, dialog, ask, confirmBox, menu, copyText, download, fmtSize, fmtDate, KIND_RU, icons } from './util.js'
import * as grid from './grid.js'

export const refreshAll = async () => { await refreshSidebar(); emit('refresh-grid') }
export async function refreshSidebar() { S.sidebar = await api('/sidebar'); emit('sidebar') }
const ids = () => grid.selectedIds()

// ---------- импорт ----------
export async function uploadFiles(files, clipboard = false) {
  const list = [...files].filter(f => f.type.startsWith('image/') || /\.(tiff?|heic|heif|avif)$/i.test(f.name))
  if (!list.length) return toast('Нет изображений среди файлов', true)
  const fd = new FormData()
  list.forEach(f => fd.append('files', f, f.name))
  fd.append('clipboard', clipboard)
  toast(`Загрузка: ${list.length} шт.`)
  try {
    const r = await api('/import/upload', { method: 'POST', body: fd })
    toast(`Добавлено: ${r.ids.length}. Обработка идёт в фоне`)
    await refreshAll()
  } catch (e) { toast('Не удалось загрузить: ' + e.message, true) }
}

export function openImport() {
  let path = ''
  const list = h('div', { class: 'folder-list' })
  const pathInp = h('input', { type: 'text', placeholder: 'C:\\Users\\…\\Pictures', value: '' })
  const info = h('div', { class: 'small muted' })
  const watch = h('input', { type: 'checkbox', checked: true })
  async function browse(p) {
    const r = await api('/fs/browse?path=' + encodeURIComponent(p || ''))
    path = r.path; pathInp.value = r.path
    info.textContent = r.path ? `Изображений в этой папке: ${r.images}` : 'Выберите диск'
    list.innerHTML = ''
    if (r.path) list.append(h('div', { class: 'tree-item', onClick: () => browse(r.parent) }, '↰ Вверх'))
    r.dirs.forEach(d => list.append(h('div', { class: 'tree-item', onClick: () => browse(r.path ? r.path.replace(/[\\/]$/, '') + (r.path.includes('/') ? '/' : '\\') + d : d) },
      h('span', { class: 'ico', html: icons.folder }), h('span', { class: 'nm' }, d))))
  }
  pathInp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); browse(pathInp.value).catch(err => toast(err.message, true)) } })
  const fileInp = h('input', { type: 'file', multiple: true, accept: 'image/*,.tif,.tiff,.heic', hidden: true, onChange: e => { uploadFiles(e.target.files); d.close() } })
  const dropzone = h('div', { class: 'card', style: { textAlign: 'center', padding: '26px', borderStyle: 'dashed' }, role: 'button',
    onClick: () => fileInp.click() }, h('b', null, 'Выберите файлы'), h('div', { class: 'muted small' }, 'или перетащите их в окно приложения. Копии сохраняются в «Изображения\\ImageHub»'))
  const body = h('div', { class: 'col' },
    h('div', { class: 'card' }, h('h4', null, 'Папка с изображениями'), h('div', { class: 'col' },
      h('div', { class: 'row' }, pathInp, h('button', { class: 'btn', onClick: () => browse(pathInp.value).catch(err => toast(err.message, true)) }, 'Открыть')),
      list, info,
      h('label', { class: 'row small' }, watch, 'Следить за папкой: новые файлы подхватываются автоматически'))),
    dropzone, fileInp,
    h('div', { class: 'card' }, h('h4', null, 'Буфер обмена'), h('div', { class: 'row' },
      h('button', { class: 'btn', onClick: async () => { await pasteFromClipboard(); d.close() } }, 'Вставить картинку из буфера'),
      h('span', { class: 'muted small' }, 'или нажмите Ctrl+V в любом месте'))))
  const d = dialog({ title: 'Импорт', body, buttons: [{ label: 'Закрыть' }, {
    label: 'Добавить папку', cls: 'primary', run: async () => {
      if (!path) { toast('Откройте нужную папку', true); return false }
      try {
        const r = await api('/import/folder', { json: { path, watched: watch.checked } })
        toast(`Найдено новых: ${r.new}. Обработка идёт в фоне`)
        await refreshAll()
      } catch (e) { toast(e.message, true); return false }
    } }] })
  browse('').catch(() => {})
}

export async function pasteFromClipboard() {
  try {
    const items = await navigator.clipboard.read()
    const files = []
    for (const it of items) for (const t of it.types) if (t.startsWith('image/')) files.push(new File([await it.getType(t)], 'clipboard.' + t.split('/')[1], { type: t }))
    if (!files.length) return toast('В буфере нет изображения', true)
    await uploadFiles(files, true)
  } catch (e) { toast('Нет доступа к буферу. Нажмите Ctrl+V в окне приложения', true) }
}

// ---------- отдельные действия ----------
export async function toggleFavorite(list) {
  const items = list.map(i => grid.loadedItems().find(x => x.id === i)).filter(Boolean)
  const val = items.some(i => !i.favorite) ? 1 : 0
  await Promise.all(list.map(i => api('/images/' + i, { method: 'PATCH', json: { favorite: val } })))
  list.forEach(i => grid.patchItem(i, { favorite: val }))
  toast(val ? 'Добавлено в избранное' : 'Убрано из избранного')
  refreshSidebar()
}

export async function deleteImages(list) {
  if (!list.length) return
  const cb = h('input', { type: 'checkbox' })
  const ok = await new Promise(res => dialog({ title: `Удалить ${list.length} шт.?`,
    body: h('div', { class: 'col' }, h('p', null, 'Изображения исчезнут из каталога. Файлы на диске останутся, если не отметить пункт ниже.'),
      h('label', { class: 'row small' }, cb, 'Удалить также файлы с диска')),
    buttons: [{ label: 'Отмена', run: () => res(false) }, { label: 'Удалить', cls: 'danger', run: () => res(true) }], onClose: () => res(false) }))
  if (!ok) return
  await api('/images/delete', { json: { ids: list, remove_file: cb.checked } })
  S.selection.clear()
  toast(`Удалено: ${list.length}`)
  S.active = null; emit('active', null)
  await refreshAll()
}

export async function renameImage(id) {
  const d = await api('/images/' + id)
  const name = await ask('Переименовать файл', 'Новое имя с расширением', d.name)
  if (!name || name === d.name) return
  try { await api('/images/' + id, { method: 'PATCH', json: { name } }); toast('Переименовано'); emit('refresh-grid'); emit('active', id) }
  catch (e) { toast(e.message, true) }
}

export async function copyImage(id) {
  try {
    const r = await fetch(`/api/images/${id}/file`)
    let blob = await r.blob()
    if (blob.type !== 'image/png') {  // буфер обмена браузера принимает PNG
      const bmp = await createImageBitmap(blob)
      const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height
      c.getContext('2d').drawImage(bmp, 0, 0)
      blob = await new Promise(res => c.toBlob(res, 'image/png'))
    }
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
    toast('Изображение скопировано в буфер обмена')
  } catch (e) { toast('Не удалось скопировать изображение: ' + e.message, true) }
}
export async function copyPath(id) { const d = await api('/images/' + id); copyText(d.path) }
export async function reprocess(list) { const r = await api('/reprocess', { json: { ids: list } }); toast(`В очереди: ${r.queued}`); emit('refresh-grid') }

export async function categoriesDialog(list) {
  const cats = await api('/categories')
  const inp = h('input', { type: 'text', placeholder: 'Новая категория, Enter', list: 'catlist' })
  const dl = h('datalist', { id: 'catlist' }, cats.map(c => h('option', { value: c.name })))
  const chips = h('div', { class: 'row', style: { flexWrap: 'wrap' } })
  const add = new Set(), remove = new Set()
  const draw = () => {
    chips.innerHTML = ''
    cats.forEach(c => chips.append(h('span', { class: 'chip' + (add.has(c.name) ? ' accent' : remove.has(c.name) ? ' auto' : ''), role: 'button', onClick: () => {
      if (add.has(c.name)) { add.delete(c.name); remove.add(c.name) } else if (remove.has(c.name)) remove.delete(c.name); else add.add(c.name); draw() } },
      (add.has(c.name) ? '+ ' : remove.has(c.name) ? '− ' : '') + c.name, h('span', { class: 'faint' }, c.n))))
  }
  inp.addEventListener('keydown', e => { if (e.key === 'Enter' && inp.value.trim()) { e.preventDefault(); const n = inp.value.trim(); if (!cats.find(c => c.name === n)) cats.unshift({ name: n, n: 0 }); add.add(n); inp.value = ''; draw() } })
  draw()
  dialog({ title: `Категории: ${list.length} шт.`, body: h('div', { class: 'col' },
    h('p', { class: 'muted small' }, 'У изображения может быть несколько категорий. Нажмите на категорию, чтобы добавить, ещё раз, чтобы убрать.'), inp, dl, chips),
    buttons: [{ label: 'Отмена' }, { label: 'Применить', cls: 'primary', run: async () => {
      await api('/categories/assign', { json: { ids: list, add: [...add], remove: [...remove] } }); toast('Категории обновлены'); await refreshAll(); emit('active', S.active) } }] })
}

export async function collectionDialog(list) {
  const colls = await api('/collections')
  const sel = h('select', null, h('option', { value: '' }, 'Новая коллекция…'), colls.map(c => h('option', { value: c.id }, c.name)))
  const inp = h('input', { type: 'text', placeholder: 'Название коллекции' })
  dialog({ title: `В коллекцию: ${list.length} шт.`, body: h('div', { class: 'col' }, sel, inp), buttons: [{ label: 'Отмена' }, { label: 'Добавить', cls: 'primary', run: async () => {
    let cid = sel.value
    if (!cid) { if (!inp.value.trim()) { toast('Введите название', true); return false } cid = (await api('/collections', { json: { name: inp.value, ids: [] } })).id }
    await api(`/collections/${cid}/images`, { json: { add: list } }); toast('Добавлено в коллекцию'); await refreshAll(); emit('active', S.active) } }] })
}

export async function assignExperiment(list) {
  const exps = await api('/experiments')
  if (!exps.length) return toast('Сначала создайте эксперимент в левой панели', true)
  const sel = h('select', null, exps.map(e => h('option', { value: e.id }, e.name)))
  dialog({ title: 'Перенести в эксперимент', body: sel, buttons: [{ label: 'Отмена' }, { label: 'Перенести', cls: 'primary', run: async () => {
    await api(`/experiments/${sel.value}/images`, { json: { ids: list } }); toast('Готово'); await refreshAll() } }] })
}

export function exportDialog(list) {
  const fmt = h('select', null, ['png', 'jpg', 'webp', 'tif', 'gif', 'bmp'].map(f => h('option', { value: f }, f.toUpperCase())))
  const q = h('input', { type: 'number', value: 90, min: 10, max: 100 })
  const ms = h('input', { type: 'number', placeholder: 'без ограничения', min: 16 })
  dialog({ title: `Экспорт и конвертация: ${list.length} шт.`, body: h('div', { class: 'grid2' },
    h('label', { class: 'fld' }, h('span', null, 'Формат'), fmt), h('label', { class: 'fld' }, h('span', null, 'Качество JPEG и WebP'), q),
    h('label', { class: 'fld' }, h('span', null, 'Длинная сторона, px'), ms)),
    buttons: [{ label: 'Отмена' }, { label: 'Скачать', cls: 'primary', run: async () => {
      try {
        const r = await api('/export', { json: { ids: list, format: fmt.value, quality: +q.value, max_side: ms.value ? +ms.value : null } })
        download(await r.blob(), list.length === 1 ? `image.${fmt.value}` : 'export.zip')
      } catch (e) { toast(e.message, true); return false } } }] })
}

export function mergeDialog(list) {
  const n = list.length
  const rows = h('input', { type: 'number', min: 0, value: 0 }), cols = h('input', { type: 'number', min: 0, value: Math.ceil(Math.sqrt(n)) })
  const cell = h('input', { type: 'number', min: 64, value: 512 }), gapI = h('input', { type: 'number', min: 0, value: 8 })
  const bg = h('input', { type: 'color', value: '#ffffff' })
  const fit = h('select', null, h('option', { value: 'contain' }, 'Вписать целиком'), h('option', { value: 'cover' }, 'Заполнить ячейку'))
  const pv = h('img', { style: { maxWidth: '100%', maxHeight: '46vh', borderRadius: '8px', background: 'var(--surface-2)' } })
  const body = () => ({ ids: list, rows: +rows.value, cols: +cols.value, cell: +cell.value, gap: +gapI.value, bg: bg.value, fit: fit.value })
  let t
  const update = () => { clearTimeout(t); t = setTimeout(async () => {
    try { const r = await api('/merge', { json: { ...body(), cell: Math.min(+cell.value, 220) } }); pv.src = URL.createObjectURL(await r.blob()) } catch (e) { toast(e.message, true) } }, 250) }
  ;[rows, cols, cell, gapI, bg, fit].forEach(x => x.addEventListener('input', update))
  update()
  dialog({ title: `Склейка ${n} изображений в сетку N×M`, wide: true, body: h('div', { class: 'col' },
    h('div', { class: 'grid2' }, h('label', { class: 'fld' }, h('span', null, 'Строк (0 = авто)'), rows), h('label', { class: 'fld' }, h('span', null, 'Столбцов (0 = авто)'), cols),
      h('label', { class: 'fld' }, h('span', null, 'Ячейка, px'), cell), h('label', { class: 'fld' }, h('span', null, 'Отступ, px'), gapI),
      h('label', { class: 'fld' }, h('span', null, 'Фон'), bg), h('label', { class: 'fld' }, h('span', null, 'Вписывание'), fit)),
    h('div', { style: { textAlign: 'center' } }, pv)),
    buttons: [{ label: 'Закрыть' },
      { label: 'Скачать PNG', run: async () => { const r = await api('/merge', { json: body() }); download(await r.blob(), 'merge.png'); return false }, close: false },
      { label: 'Сохранить в каталог', cls: 'primary', run: async () => { await api('/merge', { json: { ...body(), save: true } }); toast('Склейка добавлена в каталог'); await refreshAll() } }] })
}

export function editDialog(id, ops = []) {
  const deg = h('select', null, [90, 180, 270].map(d => h('option', { value: d }, `Повернуть на ${d}°`)))
  const fmt = h('select', null, ['png', 'jpg', 'webp'].map(f => h('option', { value: f }, f.toUpperCase())))
  dialog({ title: 'Сохранить копию', body: h('div', { class: 'col' }, h('p', { class: 'muted small' }, 'Оригинал не изменяется, результат появится в каталоге отдельным изображением.'),
    ops.length ? h('p', null, 'Применить: ' + ops.map(o => o.type === 'crop' ? 'кадрирование' : o.type).join(', ')) : deg, h('label', { class: 'fld' }, h('span', null, 'Формат'), fmt)),
    buttons: [{ label: 'Отмена' }, { label: 'Сохранить', cls: 'primary', run: async () => {
      const o = ops.length ? ops : [{ type: 'rotate', deg: +deg.value }]
      const r = await api(`/images/${id}/edit`, { json: { ops: o, format: fmt.value } }); toast('Копия добавлена в каталог'); await refreshAll(); return r } }] })
}

// ---------- контекстное меню ----------
export function contextMenu(x, y, it) {
  const sel = ids()
  const many = sel.length > 1
  menu([
    { label: 'Открыть', key: 'Enter', run: () => openViewer(grid.indexOf(it.id)) },
    { label: it.favorite ? 'Убрать из избранного' : 'В избранное', key: 'F', run: () => toggleFavorite(sel) },
    '-',
    { label: 'Категории…', run: () => categoriesDialog(sel) },
    { label: 'В коллекцию…', run: () => collectionDialog(sel) },
    { label: 'Перенести в эксперимент…', run: () => assignExperiment(sel) },
    '-',
    ...(many ? [{ label: `Склеить в сетку N×M (${sel.length})…`, run: () => mergeDialog(sel) }] : [{ label: 'Найти похожие', run: () => emit('similar', it.id) }]),
    { label: 'Экспорт и конвертация…', run: () => exportDialog(sel) },
    ...(many ? [] : [{ label: 'Копировать изображение', key: 'Ctrl+C', run: () => copyImage(it.id) }, { label: 'Копировать как путь', run: () => copyPath(it.id) }, { label: 'Переименовать…', key: 'F2', run: () => renameImage(it.id) }]),
    { label: 'Обработать заново', run: () => reprocess(sel) },
    '-',
    { label: many ? `Удалить (${sel.length})…` : 'Удалить…', key: 'Del', danger: true, run: () => deleteImages(sel) },
  ], x, y)
}

export function openViewer(index) { emit('open-viewer', index) }

// ---------- инструменты ----------
export function toolsDialog() {
  const mk = (title, text, fn) => h('div', { class: 'card', role: 'button', onClick: fn, style: { cursor: 'pointer' } }, h('b', null, title), h('div', { class: 'muted small' }, text))
  const d = dialog({ title: 'Инструменты', body: h('div', { class: 'col' },
    mk('Сводка', 'Типы, события, теги, языки текста и люди по выбранным изображениям или по всей библиотеке', () => { d.close(); summaryDialog() }),
    mk('Дубликаты', 'Точные копии по хешу и похожие кадры по эмбеддингу', () => { d.close(); duplicatesDialog() }),
    mk('Проверить новые файлы', 'Быстрая проверка по изменению каталогов', async () => { d.close(); const r = await api('/scan', { json: {} }); toast(`Новых файлов: ${r.reduce((a, x) => a + x.new, 0)}`); refreshAll() }),
    mk('Глубокая проверка', 'Сверка каждого файла по размеру и дате, дольше быстрой', async () => { d.close(); const r = await api('/scan', { json: { deep: true } }); toast(`Новых: ${r.reduce((a, x) => a + x.new, 0)}, изменённых: ${r.reduce((a, x) => a + x.updated, 0)}`); refreshAll() }),
    mk('Обработать всё заново', 'Повторный запуск всех стадий для каждого изображения', async () => { d.close(); if (await confirmBox('Запустить заново?', 'Это займёт время, очередь можно поставить на паузу.', 'Запустить')) reprocess([]) })) })
}

async function summaryDialog() {
  const list = ids()
  const s = await api('/summary', { json: { ids: list } })
  if (!s.count) return toast('Нет данных', true)
  const row = (arr, f = x => x) => arr.length ? h('div', { class: 'row', style: { flexWrap: 'wrap' } }, arr.map(([n, c]) => h('span', { class: 'chip' }, f(n), h('b', null, c)))) : h('span', { class: 'muted small' }, 'нет')
  dialog({ title: list.length ? `Сводка по выбранным: ${s.count}` : `Сводка по библиотеке: ${s.count}`, wide: true, body: h('div', { class: 'col' },
    h('div', { class: 'stat' }, h('div', null, h('b', null, s.count), 'изображений'), h('div', null, h('b', null, fmtSize(s.bytes)), 'объём'),
      h('div', null, h('b', null, s.with_text), 'с текстом'), h('div', null, h('b', null, fmtDate(s.from, false)), `до ${fmtDate(s.to, false)}`)),
    h('div', { class: 'card' }, h('h4', null, 'Типы'), row(s.kinds, k => KIND_RU[k] || k)),
    h('div', { class: 'card' }, h('h4', null, 'События'), row(s.events)),
    h('div', { class: 'card' }, h('h4', null, 'Язык текста'), row(s.languages)),
    h('div', { class: 'card' }, h('h4', null, 'Частые объекты и теги'), row(s.tags)),
    h('div', { class: 'card' }, h('h4', null, 'Люди'), row(s.people))), buttons: [{ label: 'Закрыть' }] })
}

async function duplicatesDialog() {
  const groups = await api('/duplicates')
  const toDelete = new Set()
  const body = h('div', null, groups.length ? groups.map(g => h('div', { class: 'dup-group' },
    h('div', { style: { width: '100%' }, class: 'small muted' }, g.type === 'exact' ? 'Точные копии' : 'Похожие'),
    g.items.map((it, k) => h('figure', null, h('img', { src: `/api/images/${it.id}/thumb` }), h('div', null, it.name),
      h('div', { class: 'faint' }, `${it.width}×${it.height}, ${fmtSize(it.size)}`),
      h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: k > 0 && g.type === 'exact', onChange: e => e.target.checked ? toDelete.add(it.id) : toDelete.delete(it.id) }), 'убрать'))))) : h('p', { class: 'muted' }, 'Дубликатов нет'))
  groups.forEach(g => g.items.forEach((it, k) => { if (k > 0 && g.type === 'exact') toDelete.add(it.id) }))
  dialog({ title: `Дубликаты: групп ${groups.length}`, wide: true, body, buttons: [{ label: 'Закрыть' }, { label: 'Убрать отмеченные', cls: 'danger', run: async () => {
    if (!toDelete.size) return; await api('/images/delete', { json: { ids: [...toDelete], remove_file: false } }); toast(`Убрано из каталога: ${toDelete.size}`); refreshAll() } }] })
}

export async function newExperiment() {
  const name = await ask('Новый эксперимент', 'Название (графики, добавленные дальше, попадут в него)', 'Эксперимент ' + new Date().toLocaleDateString('ru-RU'), 'Создать')
  if (!name) return
  await api('/experiments', { json: { name } }); toast('Эксперимент создан'); await refreshAll()
}
export async function newCollection() {
  const name = await ask('Новая коллекция', 'Название'); if (!name) return
  await api('/collections', { json: { name } }); await refreshSidebar()
}
