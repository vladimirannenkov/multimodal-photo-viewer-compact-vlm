// Левая панель: библиотека, папки, категории (топ-5 и все), коллекции, эксперименты, люди, типы
import { S, emit, on } from './state.js'
import { $, h, api, icons, KIND_RU, menu, ask, confirmBox, toast, closeMenu } from './util.js'
import * as actions from './actions.js'

const left = $('#left')
let showAllCats = false

export function navigate(key, filters, title) {
  S.navKey = key; S.filters = filters; S.title = title; S.search = null
  $('#q').value = ''
  emit('search-cleared')
  $('#title').textContent = title
  $('#banner').hidden = true
  emit('refresh-grid')
  renderExperimentCard()
  draw()
}

function item(key, icon, name, n, filters, title, extra = {}) {
  const el = h('div', { class: 'tree-item' + (S.navKey === key ? ' active' : ''), role: 'button', tabIndex: 0, title: extra.title || name,
    onClick: () => navigate(key, filters, title || name), onContextmenu: e => { if (extra.menu) { e.preventDefault(); menu(extra.menu, e.clientX, e.clientY) } } },
    extra.avatar ? h('img', { class: 'person-av', src: extra.avatar, loading: 'lazy' }) : h('span', { class: 'ico', html: icon }),
    h('span', { class: 'nm' }, name), n != null && h('span', { class: 'n' }, n))
  el.addEventListener('keydown', e => { if (e.key === 'Enter') el.click() })
  el.addEventListener('dragover', e => { if (extra.drop) { e.preventDefault(); el.style.background = 'var(--accent-soft)' } })
  el.addEventListener('dragleave', () => { el.style.background = '' })
  el.addEventListener('drop', async e => {
    el.style.background = ''
    const raw = e.dataTransfer.getData('application/x-imagehub-ids')
    if (extra.drop && raw) { e.preventDefault(); e.stopPropagation(); await extra.drop(JSON.parse(raw)) }
  })
  return el
}

function section(title, children, action) {
  return h('div', { class: 'sec' }, h('div', { class: 'sec-h' }, h('span', null, title), action), children)
}

export function draw() {
  const sb = S.sidebar
  if (!sb) return
  left.innerHTML = ''
  left.append(section('Библиотека', [
    item('all', icons.all, 'Все изображения', sb.total, {}, 'Все изображения'),
    item('fav', icons.star, 'Избранное', null, { favorite: 1 }, 'Избранное'),
    sb.clipboard ? item('clip', icons.clip, 'Из буфера', sb.clipboard, { clipboard: 1 }, 'Вставлено из буфера') : null,
  ]))
  left.append(section('Папки', sb.folders.length ? sb.folders.map(f => item('f' + f.id, icons.folder, f.path.split(/[\\/]/).filter(Boolean).pop() || f.path, f.n, { folder: f.id }, f.path, {
    title: f.path, menu: [{ label: 'Проверить новые файлы', run: async () => { await api('/scan', { json: {} }); actions.refreshAll() } },
      { label: 'Убрать папку из каталога', danger: true, run: async () => { if (await confirmBox('Убрать папку?', 'Файлы на диске останутся нетронутыми.', 'Убрать')) { await api('/folders/' + f.id, { method: 'DELETE' }); navigate('all', {}, 'Все изображения'); actions.refreshAll() } } }] }))
    : h('div', { class: 'muted small', style: { padding: '4px 8px' } }, 'Папок нет'),
    h('button', { class: 'btn icon ghost sm', title: 'Добавить папку', onClick: () => actions.openImport(), html: icons.plus })))
  const cats = showAllCats ? sb.categories : sb.top_categories
  left.append(section(showAllCats ? 'Все категории' : 'Топ-5 категорий', [
    ...cats.map(c => item('c' + c.id, icons.tag, c.name, c.n, { category: c.id }, 'Категория: ' + c.name, {
      drop: async ids => { await api('/categories/assign', { json: { ids, add: [c.name] } }); toast(`В категорию «${c.name}»: ${ids.length}`); actions.refreshSidebar() },
      menu: [{ label: 'Переименовать (или слить с другой)…', run: async () => { const n = await ask('Категория', 'Название', c.name); if (n) { await api('/categories/' + c.id, { method: 'PATCH', json: { name: n } }); actions.refreshAll() } } },
        { label: 'Удалить категорию', danger: true, run: async () => { await api('/categories/' + c.id, { method: 'DELETE' }); actions.refreshAll() } }] })),
    sb.categories.length > 5 && h('div', { class: 'tree-item', onClick: () => { showAllCats = !showAllCats; draw() } }, h('span', { class: 'muted small' }, showAllCats ? 'Свернуть' : `Показать все (${sb.categories.length})`)),
  ]))
  left.append(section('Коллекции', [
    ...sb.collections.map(c => item('k' + c.id, icons.coll, c.name, c.n, { collection: c.id }, 'Коллекция: ' + c.name, {
      drop: async ids => { await api(`/collections/${c.id}/images`, { json: { add: ids } }); toast(`В коллекцию «${c.name}»: ${ids.length}`); actions.refreshSidebar() },
      menu: [{ label: 'Переименовать…', run: async () => { const n = await ask('Коллекция', 'Название', c.name); if (n) { await api('/collections/' + c.id, { method: 'PATCH', json: { name: n } }); actions.refreshSidebar() } } },
        { label: 'Удалить коллекцию', danger: true, run: async () => { await api('/collections/' + c.id, { method: 'DELETE' }); navigate('all', {}, 'Все изображения'); actions.refreshAll() } }] })),
    !sb.collections.length && h('div', { class: 'muted small', style: { padding: '4px 8px' } }, 'Перетащите сюда изображения'),
  ], h('button', { class: 'btn icon ghost sm', title: 'Новая коллекция', onClick: actions.newCollection, html: icons.plus })))
  left.append(section('Эксперименты', [
    ...sb.experiments.map(e => item('e' + e.id, icons.flask, e.name, e.n, { experiment: e.id }, 'Эксперимент: ' + e.name, {
      drop: async ids => { await api(`/experiments/${e.id}/images`, { json: { ids } }); actions.refreshAll() },
      menu: [{ label: 'Переименовать…', run: async () => { const n = await ask('Эксперимент', 'Название', e.name); if (n) { await api('/experiments/' + e.id, { method: 'PATCH', json: { name: n } }); actions.refreshSidebar() } } },
        { label: 'Удалить эксперимент (графики останутся)', danger: true, run: async () => { await api('/experiments/' + e.id, { method: 'DELETE' }); actions.refreshAll() } }] })),
    !sb.experiments.length && h('div', { class: 'muted small', style: { padding: '4px 8px' } }, 'Первый эксперимент создастся сам по первому графику'),
  ], h('button', { class: 'btn icon ghost sm', title: 'Новый эксперимент', onClick: actions.newExperiment, html: icons.plus })))
  if (sb.people.length) left.append(section('Люди', sb.people.slice(0, 40).map(p => item('p' + p.id, icons.user, p.name + (p.labeled ? '' : ' ?'), p.n, { person: p.id }, 'Человек: ' + p.name, {
    avatar: p.face_id ? `/api/faces/${p.face_id}/crop?size=44` : null,
    menu: [{ label: 'Назвать…', run: async () => { const n = await ask('Имя', 'Как зовут этого человека', p.labeled ? p.name : ''); if (n) { await api('/people/' + p.id, { method: 'PATCH', json: { name: n } }); actions.refreshAll() } } }] }))))
  if (sb.kinds.length) left.append(section('Типы', sb.kinds.map(k => item('t' + k.kind, icons.kind, KIND_RU[k.kind] || k.kind, k.n, { kind: k.kind }, 'Тип: ' + (KIND_RU[k.kind] || k.kind)))))
}

async function renderExperimentCard() {
  const el = $('#expCard')
  const eid = S.filters.experiment
  if (!eid) { el.hidden = true; return }
  const s = await api(`/experiments/${eid}/summary`)
  el.hidden = false
  el.innerHTML = ''
  const rows = s.charts.filter(c => c.series.length)
  el.append(h('div', { class: 'row' }, h('b', null, s.experiment.name), h('span', { class: 'muted small' }, 'создан ' + new Date(s.experiment.created_at * 1000).toLocaleDateString('ru-RU')),
    h('span', { class: 'spacer' }), s.best && h('span', { class: 'chip accent' }, `лучшая серия: ${s.best.name || '—'}, максимум ${(+s.best.max).toPrecision(4)}`)))
  if (rows.length) el.append(h('table', { class: 't', style: { marginTop: '8px' } },
    h('tr', null, ['График', 'Серия', 'Максимум', 'Плато', 'Финал'].map(x => h('th', null, x))),
    rows.flatMap(c => c.series.map(sr => h('tr', null, h('td', null, c.title || c.name), h('td', null, sr.name || '—'), h('td', null, (+sr.max).toPrecision(4)),
      h('td', null, sr.plateau != null ? (+sr.plateau).toPrecision(4) : '—'), h('td', null, (+sr.final).toPrecision(4)))))))
  else el.append(h('div', { class: 'muted small', style: { marginTop: '4px' } }, 'Точки графиков ещё не извлечены или в эксперименте нет графиков'))
}

on('sidebar', draw)
on('sidebar', () => { if (S.filters.experiment) renderExperimentCard() })
