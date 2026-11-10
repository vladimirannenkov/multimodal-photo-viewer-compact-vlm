(() => {
  const $ = s => document.querySelector(s)
  const node = (tag, cls, value) => { const el = document.createElement(tag); if (cls) el.className = cls; if (value !== undefined) el.textContent = value; return el }
  const number = (n, digits = 1) => Number(n).toLocaleString('ru-RU', { maximumFractionDigits: digits })
  const bytes = n => n == null ? 'Нет данных' : n >= 1024 ** 3 ? `${number(n / 1024 ** 3, 2)} ГиБ` : `${number(n / 1024 ** 2, 0)} МиБ`
  const seconds = n => n >= 60 ? `${number(n / 60, 1)} мин` : `${number(n, 2)} с`
  const name = key => key.replace(/^VLM\.generate:api:/, '').replace(/^VLM\.generate:.*[\\/]/, '').replace(/^stage\./, '').replace(/^image\./, 'Изображение: ')
  function card(label, value, sub) { const c = node('div', 'card'); c.append(node('div', 'label', label), node('div', 'value', value), node('div', 'sub', sub || '')); return c }
  function chart(title, entries, format, color, note, fixedMax) {
    const box = node('div', 'chart'); box.append(node('h4', '', title)); if (note) box.append(node('p', 'chart-note', note))
    entries = entries.filter(x => Number.isFinite(x[1]) && x[1] >= 0).sort((a, b) => b[1] - a[1]).slice(0, 16)
    if (!entries.length) { box.append(node('div', 'empty', 'Нет измерений')); return box }
    const max = fixedMax || Math.max(...entries.map(x => x[1]), 1e-12)
    for (const [label, value] of entries) { const row = node('div', 'bar-row'); const caption = node('span', 'bar-label', name(label)); caption.title = label; const track = node('div', 'bar-track'); const bar = node('div', `bar ${color || ''}`); bar.style.width = `${Math.max(value / max * 100, value ? 1 : 0)}%`; track.append(bar); row.append(caption, track, node('span', 'bar-value', format(value))); box.append(row) }
    return box
  }
  function plots(profile, target) {
    const rows = Object.entries(profile || {}); const stages = rows.filter(([k]) => k.startsWith('stage.') || k.startsWith('image.')); const models = rows.filter(([k]) => !k.startsWith('stage.') && !k.startsWith('image.'))
    if (!rows.length) { target.append(node('div', 'empty', 'В этом процессе пока нет вызовов моделей. Результаты отдельных запусков показаны ниже.')); return }
    if (stages.length) target.append(chart('Время этапов', stages.map(([k,v]) => [k,v.seconds]), seconds, '', 'Суммарное время вызовов, включая загрузку.'))
    if (stages.some(([,v]) => v.cpu_seconds != null)) target.append(chart('Время CPU по этапам', stages.map(([k,v]) => [k,v.cpu_seconds]), seconds, 'orange', 'Время CPU процесса ImageHub; внешние процессы Ollama и импорта не входят.'))
    target.append(chart('Время моделей', models.map(([k,v]) => [k,v.seconds]), seconds, '', 'Вложенные вызовы могут пересекаться с этапами.'))
    target.append(chart('Количество вызовов', models.map(([k,v]) => [k,v.calls]), n => number(n, 0), 'teal'))
    target.append(chart('Время CPU', models.map(([k,v]) => [k,v.cpu_seconds]), seconds, 'orange', 'Процессорное время процесса приложения; сервер Ollama в него не входит. Параллельные и вложенные вызовы могут пересекаться.'))
    target.append(chart('Средняя загрузка CPU', models.map(([k,v]) => [k,v.seconds > 0 && v.cpu_seconds != null ? 100*v.cpu_seconds/v.seconds : undefined]), x => `${number(x, 0)}%`, 'teal', '100% соответствует одному полностью занятому ядру; несколько ядер могут дать более 100%.'))
    target.append(chart('Память процесса', models.map(([k,v]) => [k,v.max_observed_rss_bytes]), bytes, 'orange', 'Максимум на границах вызовов за время жизни процесса; включает память других моделей и может относиться к предыдущим запускам.'))
    target.append(chart('Память GPU', models.map(([k,v]) => [k, v.ollama_max_observed_size_vram_bytes ?? v.max_observed_cuda_allocated_bytes]), bytes, 'teal', 'Накопленный максимум: для Ollama — объём модели по серверу, для локальных моделей — память PyTorch на границах вызовов.'))
  }
  function title(run) { const s = run.summary || {}; const text = String(s.dataset || s.kind || run.name || 'Оценка'); if (text.includes('flickr')) return 'Flickr30k — описания'; if (text.includes('invoice')) return 'Invoices and receipts — OCR'; if (text.includes('pipeline-profile-cpu')) return 'Загрузка CPU — обработка 500 изображений'; if (text.includes('pipeline-profile')) return 'Пакетная обработка — профиль'; return ({application_pipeline:'Профиль обработки библиотеки',batch:'Пакетная обработка',describe:'Описание изображения',ask:'Вопрос к изображению',chat:'Чат по изображению'})[s.kind] || text }
  function metric(label, value) { const m = node('div', 'metric'); m.append(node('strong', '', value), node('span', '', label)); return m }
  function addMetrics(summary, into) {
    const s = summary || {}; const dataset = s.dataset
    if (dataset) into.append(metric('Датасет / источник', Array.isArray(dataset) ? dataset.join(' / ') : String(dataset)))
    for (const [key,label] of [['processed','Обработано'],['completed','Успешно'],['requested','Запрошено'],['images','Изображений'],['images_with_errors','С ошибками'],['errors','Ошибки'],['elapsed_seconds','Время'],['mean_cosine','Средний косинус'],['mean_max_cosine','Средний максимум']]) if (typeof s[key] === 'number') into.append(metric(label, key === 'elapsed_seconds' ? seconds(s[key]) : number(s[key], key.includes('cosine') ? 3 : 0)))
    for (const [key,label] of [['date_correct','Дата: точность'],['total_correct','Сумма: точность'],['date_found','Дата: найдена в тексте'],['total_found','Сумма: найдена в тексте']]) { const x = s[key]; if (x && typeof x.eligible === 'number') into.append(metric(label, `${x.matched}/${x.eligible} · ${x.eligible ? number(100*x.matched/x.eligible, 1) : '—'}%`)) }
    if (s.status) into.append(metric('Статус', s.status === 'running' ? 'Выполняется' : s.status === 'completed' ? 'Завершено' : String(s.status)))
    if (s.cumulative_profile) into.append(metric('Профиль', 'Накопительный с запуска'))
  }
  function renderRun(run, index) {
    const s = run.summary || {}; const box = node('details', 'run'); box.open = index === 0; const head = node('summary'); const titleBox = node('div','title'); titleBox.append(node('h3','',title(run)), node('div','time', `${run.name} · ${new Date(run.modified*1000).toLocaleString('ru-RU')}`)); head.append(titleBox, node('span','badge', s.status === 'running' ? 'Выполняется' : 'Сохранено')); box.append(head)
    const body = node('div','run-body'); const metrics = node('div','metrics'); addMetrics(s, metrics); body.append(metrics)
    if (s.status === 'running' && s.images && typeof s.processed === 'number') { const p = node('div','progress'); const fill = node('div'); fill.style.width = `${Math.min(100,100*s.processed/s.images)}%`; p.append(fill); body.append(p) }
    const graphs = node('div','charts'); const quality = []
    if (typeof s.mean_cosine === 'number') quality.push(['Среднее сходство подписей', s.mean_cosine])
    if (typeof s.mean_max_cosine === 'number') quality.push(['Лучшее сходство подписи', s.mean_max_cosine])
    for (const [key,label] of [['date_correct','Дата: точность'],['total_correct','Сумма: точность'],['date_found','Дата: присутствие'],['total_found','Сумма: присутствие']]) { const x = s[key]; if (x && x.eligible) quality.push([label, x.matched / x.eligible]) }
    if (quality.length) graphs.append(chart('Качество на выборке', quality, x => `${number(x*100,1)}%`, 'teal', 'Шкала от 0 до 100%. Сходство — косинус текстовых эмбеддингов; точность полей — доля совпадений.', 1))
    plots(run.profile, graphs); body.append(graphs)
    if (s.description || s.answer) { const result = node('div','panel'); result.append(node('h3','',s.answer ? 'Ответ' : 'Описание'), node('p','',s.answer || s.description)); body.append(result) }
    if (Array.isArray(run.records) && run.records.length) {
      const details = node('details','panel'); details.append(node('summary','',`Результаты по изображениям (${run.records.length})`))
      for (const item of run.records) { const row = node('details','record'); let caption = item.filename || item.id || `Строка ${item.row_idx}`; if (item.error) caption += ' · ошибка'; else if (typeof item.cosine_mean === 'number') caption += ` · сходство ${number(item.cosine_mean,3)}`; else if (item.date_correct != null || item.total_correct != null) caption += ` · дата ${item.date_correct ? '✓' : '✗'}, сумма ${item.total_correct ? '✓' : '✗'}`; row.append(node('summary','',caption)); if (item.error) row.append(node('p','failure',item.error)); if (item.caption) row.append(node('p','',item.caption)); if (item.references) row.append(node('p','muted',`Эталоны: ${item.references.join(' · ')}`)); if (item.expected_date || item.expected_total) row.append(node('p','',`Дата: ${item.predicted_date || 'не найдена'} (эталон ${item.expected_date || 'нет'}); сумма: ${item.predicted_total || 'не найдена'} (эталон ${item.expected_total || 'нет'})`)); if (item.ocr) row.append(node('pre','ocr',item.ocr)); details.append(row) }
      body.append(details)
    }
    box.append(body); return box
  }
  async function refresh() {
    $('#error').textContent = ''; $('#refresh').disabled = true
    try {
      const [live, history] = await Promise.all(['/api/profile','/api/evaluations'].map(async url => { const r = await fetch(url, {cache:'no-store'}); if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() }))
      const cards = $('#current-cards'); cards.replaceChildren(); cards.append(card('Память приложения', bytes(live.memory?.rss_bytes), 'RSS текущего процесса'), card('Память CUDA', bytes(live.memory?.cuda_allocated_bytes), 'PyTorch текущего процесса'), card('Вызовов моделей', number(Object.values(live.calls || {}).reduce((a,v) => a + (v.calls || 0),0),0), 'с запуска приложения'))
      for (const model of live.ollama_loaded_models || []) cards.append(card(model.name, bytes(model.size_vram), 'модель в GPU по данным Ollama'))
      const current = $('#current-charts'); current.replaceChildren(); plots(live.calls, current)
      const root = $('#runs'); root.replaceChildren(); const runs = history.evaluations || []; $('#run-count').textContent = `${runs.length} сохранённых запусков`; if (!runs.length) root.append(node('div','empty','Сохранённых оценок пока нет.')); runs.forEach((run,i) => root.append(renderRun(run,i)))
      $('#updated').textContent = `Обновлено ${new Date().toLocaleTimeString('ru-RU')}`
    } catch (e) { $('#error').textContent = `Не удалось получить данные: ${e.message}. Проверьте, что сервер ImageHub работает.` } finally { $('#refresh').disabled = false }
  }
  $('#refresh').addEventListener('click', refresh); refresh()
})()
