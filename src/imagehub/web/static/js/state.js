// Общее состояние и простая шина событий
export const S = {
  filters: {},            // folder, collection, category, experiment, person, kind, favorite, clipboard
  title: 'Все изображения',
  navKey: 'all',
  sort: 'date', desc: true, hasText: '',
  search: null,           // {items, query} в режиме поиска
  selection: new Set(),
  active: null,           // id открытого в правой панели
  sidebar: null,
  queue: null,
}
const bus = new EventTarget()
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }))
export const on = (name, fn) => bus.addEventListener(name, e => fn(e.detail))
