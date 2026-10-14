"""Структура Markdown для чеков и документов без надёжного макета.

Вход: текст страницы от OCR (строки и пустые строки между разделами сохранены).
Выход: Markdown: разделы, жёсткие переносы строк, пары «поле: значение» жирным, таблица позиций из числовых строк."""
import re

NUM = re.compile(r"^[\(\-+]?\d[\d.,]*%?\)?$")


def _num_tokens(line):
    return sum(1 for t in line.split() if NUM.match(t))


def is_numeric_row(line):
    toks = line.split()
    return len(toks) >= 3 and _num_tokens(line) >= 3 and _num_tokens(line) >= len(toks) - 1


def is_header_row(line):
    toks = line.split()
    return 3 <= len(toks) <= 8 and _num_tokens(line) == 0 and all(len(t) <= 14 for t in toks)


def _md_row(cells):
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"


def _table(header, rows, desc):
    n = max(len(r) for r in rows)
    if header and len(header) == n:
        cols = header
    else:
        cols = [f"Колонка {i + 1}" for i in range(n)]
    cols = cols + (["Описание"] if desc else [])
    out = [_md_row(cols), "|" + "---|" * len(cols)]
    for r, d in zip(rows, desc or [""] * len(rows)):
        out.append(_md_row(r + [""] * (n - len(r)) + ([d] if desc else [])))
    return "\n".join(out)


def _kv(line):
    """«Ключ : значение» с коротким ключом -> **Ключ:** значение."""
    m = re.match(r"^([^:]{2,32}?)\s*:\s*(\S.*)$", line)
    if m and not re.search(r"\d{2}:\d{2}", m.group(1)):
        return f"**{m.group(1).strip()}:** {m.group(2).strip()}"
    return None


def structure_section(lines):
    """Один раздел (группа строк между пустыми строками) -> блок Markdown."""
    out, i = [], 0
    header = None
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        # таблица позиций: заголовок (необязательно) + подряд идущие числовые строки (+ строки описания)
        merged = None  # заголовок, перенесённый на вторую строку одним словом («Item Qty (RM) (RM)» + «Amount»)
        if i + 2 < n and len(lines[i + 1].split()) == 1 and is_numeric_row(lines[i + 2].strip())                 and is_header_row(ln + " " + lines[i + 1].strip()):
            merged = (ln + " " + lines[i + 1].strip()).split()
        if merged or is_numeric_row(ln) or (is_header_row(ln) and i + 1 < n and is_numeric_row(lines[i + 1].strip())):
            header = None
            if merged:
                header = merged
                i += 2
            elif is_header_row(ln):
                header = ln.split()
                i += 1
            rows, desc = [], []
            while i < n and is_numeric_row(lines[i].strip()):
                rows.append(lines[i].split())
                i += 1
                if i < n and lines[i].strip() and not is_numeric_row(lines[i].strip()):
                    desc.append(lines[i].strip())
                    i += 1
                else:
                    desc.append("")
            out.append(_table(header, rows, desc if any(desc) else None))
            continue
        kv = _kv(ln)
        out.append((kv or ln) + "  ")  # два пробела в конце: жёсткий перенос строки
        i += 1
    # соседние строки текста склеиваем в один абзац с жёсткими переносами, таблицы отделяем пустыми строками
    res, buf = [], []
    for item in out:
        if item.startswith("|"):
            if buf:
                res.append("\n".join(buf))
                buf = []
            res.append(item)
        else:
            buf.append(item)
    if buf:
        res.append("\n".join(buf).rstrip())
    return "\n\n".join(res)


def _drop_blanks_in_tables(lines):
    """Пустые строки между числовыми строками позиций и их описаниями (шум от границ полос) убираем."""
    out, i, n = [], 0, len(lines)
    while i < n:
        out.append(lines[i])
        if lines[i].strip() and i + 1 < n and not lines[i + 1].strip():
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            prev_in_table = is_numeric_row(lines[i].strip()) or (
                len(out) >= 2 and out[-2].strip() and is_numeric_row(out[-2].strip()))
            if j < n and prev_in_table and (is_numeric_row(lines[j].strip()) or
                                            (j + 1 < n and is_numeric_row(lines[j + 1].strip()))):
                i = j
                continue
        i += 1
    return out


def structure_page_text(text):
    lines = _drop_blanks_in_tables(text.replace(chr(13), "").split(chr(10)))
    sections, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln)
        elif cur:
            sections.append(cur)
            cur = []
    if cur:
        sections.append(cur)
    blocks = [structure_section(s) for s in sections]
    # соседние абзацы из одной строки «**ключ:** значение» объединяем в один блок с жёсткими переносами
    out = []
    for blk in blocks:
        one_kv = blk.startswith("**") and chr(10) not in blk.strip()
        if one_kv and out and out[-1].startswith("**") and not out[-1].startswith("|"):
            out[-1] = out[-1].rstrip() + "  " + chr(10) + blk
        else:
            out.append(blk)
    return (chr(10) * 2).join(out)


def layout_unreliable(blocks, size):
    """Макет считаем ненадёжным (чеки, сканы): огромная таблица, низкая уверенность, сильные перекрытия."""
    w, h = size
    page = w * h
    if not blocks:
        return True
    for b in blocks:
        x1, y1, x2, y2 = b["box"]
        if b["cls"] == "table" and (x2 - x1) * (y2 - y1) > 0.45 * page:
            return True
    if len(blocks) >= 5 and sum(b["score"] for b in blocks) / len(blocks) < 0.5:
        return True
    ov = 0
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            ix = max(0, min(a["box"][2], b["box"][2]) - max(a["box"][0], b["box"][0]))
            iy = max(0, min(a["box"][3], b["box"][3]) - max(a["box"][1], b["box"][1]))
            inter = ix * iy
            small = min((a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1]),
                        (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])) or 1
            if inter / small > 0.5:
                ov += 1
    return ov >= 2


def split_bands(im, lines_per_band=10, ink_thr=160):
    """Режет длинную страницу на горизонтальные полосы по пустым промежуткам между строками.
    -> [(y1, y2, sep)], sep: разделитель перед полосой (пустой для первой, один или два перевода строки по ширине промежутка)."""
    import numpy as np
    g = np.asarray(im.convert("L"))
    h, w = g.shape
    thr = min(ink_thr + 40, g.mean() - 0.35 * g.std())  # адаптивный порог: бледные строки тоже считаются текстом
    rows = (g < thr).sum(axis=1) > max(2, 0.004 * w)
    runs, start = [], None
    for y, v in enumerate(rows):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, h))
    if len(runs) <= lines_per_band + 2:
        return [(0, h, "")]
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    med = sorted(gaps)[len(gaps) // 2] or 1
    bands, first = [], 0
    k = max(2, round(len(runs) / lines_per_band))
    target = len(runs) / k
    cuts = []
    for j in range(1, k):
        ideal = int(round(j * target))
        lo, hi = max(1, ideal - 2), min(len(runs) - 1, ideal + 2)
        cuts.append(max(range(lo, hi + 1), key=lambda c: gaps[c - 1]))  # самый большой промежуток рядом с идеальным местом
    cuts = sorted(set(cuts))
    out, prev_line = [], 0
    for c in cuts + [len(runs)]:
        y1 = 0 if prev_line == 0 else (runs[prev_line - 1][1] + runs[prev_line][0]) // 2
        y2 = h if c == len(runs) else (runs[c - 1][1] + runs[c][0]) // 2
        sep = "" if prev_line == 0 else (chr(10) * 2 if gaps[prev_line - 1] > 1.8 * med else chr(10))
        out.append((max(0, y1 - 3), min(h, y2 + 3), sep))
        prev_line = c
    return out
