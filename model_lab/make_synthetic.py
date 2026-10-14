"""python make_synthetic.py  Синтетические страницы с эталоном: samples/ru_text, samples/mixed, samples/formulas_hq."""
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent / "samples"
DPI = 250
W, H, MARGIN = 8.27, 11.69, 0.95
FS, LINE_H = 11.5, 0.40
random.seed(7)

# абзацы; формулы внутри $...$ (без пробелов, пробелы задаются через \,)
TEXTS = [
    r"Пусть функция $f(x)=x^{2}$ непрерывна на отрезке $[0,1]$. Тогда определённый интеграл $\int_{0}^{1}x^{2}\,dx=\frac{1}{3}$ существует, а его значение не зависит от разбиения.",
    r"Для классификации используется сигмоида $\sigma(x)=\frac{1}{1+e^{-x}}$, а качество обучения оценивается функцией потерь $\mathcal{L}=-\sum_{i}y_{i}\log\hat{y}_{i}$ на валидационной выборке.",
    r"Энергия покоя тела определяется формулой $E=mc^{2}$, где $m$ обозначает массу, а $c$ скорость света. Эта величина служит основой для оценки энергетического баланса системы.",
    r"Среднее значение выборки вычисляется как $\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_{i}$, а дисперсия равна $s^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}$. Результаты приведены ниже.",
    r"Скорость обучения подбиралась из интервала $\alpha\in[10^{-5},10^{-2}]$, при этом оптимизатор обновляет веса по правилу $w_{t+1}=w_{t}-\alpha\nabla\mathcal{L}(w_{t})$ на каждом шаге.",
    r"Решение квадратного уравнения $ax^{2}+bx+c=0$ находится по формуле $x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}$, если дискриминант $D=b^{2}-4ac$ неотрицателен.",
]
PLAIN = [
    "Результаты эксперимента показали, что точность модели выросла после увеличения числа эпох обучения. Дальнейший рост наблюдался до тридцатой эпохи, после чего метрика вышла на плато.",
    "В докладе рассматривается задача распознавания текста на изображениях. Особое внимание уделено формулам и таблицам в научных статьях, а также устойчивости к шуму и повороту.",
    "Договор заключён между сторонами в городе Москве. Срок действия соглашения составляет двенадцать месяцев с даты подписания, после чего он может быть продлён по обоюдному согласию.",
    "Вчера на празднике во дворе собралось много людей. Дети играли, а взрослые угощались пирогами и чаем, обсуждая планы на предстоящие выходные.",
    "Настоящая инструкция описывает порядок установки программного обеспечения. Перед началом убедитесь, что на диске свободно не менее десяти гигабайт.",
]
TABLES = [
    [["Модель", "Точность, %", "Время, мс"], ["GLM-OCR", "94.5", "820"], ["Qwen3-VL", "91.2", "1400"], ["Florence-2", "88.7", "310"]],
    [["Эпоха", "Loss", "Accuracy"], ["1", "2.31", "0.12"], ["10", "0.84", "0.71"], ["30", "0.21", "0.94"]],
    [["Параметр", "Значение"], ["Размер батча", "32"], ["Скорость обучения", "0.001"], ["Число эпох", "30"]],
    [["Город", "Население, тыс.", "Год"], ["Москва", "13 010", "2024"], ["Казань", "1 330", "2024"], ["Тула", "470", "2024"]],
]
FORMULAS = [r"E=mc^{2}", r"\int_{0}^{1}x^{2}\,dx=\frac{1}{3}", r"\sum_{i=1}^{n}i=\frac{n(n+1)}{2}",
            r"\sigma(x)=\frac{1}{1+e^{-x}}", r"\mathcal{L}=-\sum_{i}y_{i}\log\hat{y}_{i}"]


def md_table(rows):
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


class Page:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0); self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()
        self.y, self.gt = MARGIN, []

    def width(self, s, **kw):
        t = self.ax.text(0, 0, s, fontsize=kw.get("fs", FS), fontweight=kw.get("bold", "normal"))
        w = t.get_window_extent(self.r).width / DPI
        t.remove()
        return w

    def paragraph(self, text):
        tokens = re.findall(r"\$[^$]+\$[^\s$]*|\S+", text)
        lines, cur = [], ""
        for tok in tokens:
            trial = (cur + " " + tok).strip()
            if self.width(trial) > W - 2 * MARGIN and cur:
                lines.append(cur); cur = tok
            else:
                cur = trial
        lines.append(cur)
        for i, ln in enumerate(lines):
            self.ax.text(MARGIN, self.y + 0.26, ln, fontsize=FS, va="baseline")
            self.y += LINE_H
        self.y += 0.12
        self.gt.append(text)

    def table(self, rows):
        pad, rh = 0.16, 0.30
        cw = [max(self.width(r[c], bold="bold" if i == 0 else "normal") for i, r in enumerate(rows)) + 2 * pad
              for c in range(len(rows[0]))]
        tw, x0 = sum(cw), (W - sum(cw)) / 2
        self.y += 0.06
        self.ax.add_patch(Rectangle((x0, self.y), tw, rh, color="#eceff3", lw=0))
        for i, row in enumerate(rows):
            x = x0
            for c, cell in enumerate(row):
                self.ax.text(x + pad, self.y + i * rh + rh / 2, cell, fontsize=FS - 0.5, va="center",
                             fontweight="bold" if i == 0 else "normal")
                x += cw[c]
        for yy, lw in [(self.y, 1.2), (self.y + rh, 0.8), (self.y + len(rows) * rh, 1.2)]:
            self.ax.plot([x0, x0 + tw], [yy, yy], color="#222", lw=lw)
        self.y += len(rows) * rh + 0.22
        self.gt.append(md_table(rows))

    def save(self, folder, name):
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out / f"{name}.png", dpi=DPI, facecolor="white")
        plt.close(self.fig)
        (out / f"{name}.md").write_text("\n\n".join(self.gt), encoding="utf-8")


for i in range(6):
    p = Page()
    for t in random.sample(PLAIN, 4):
        p.paragraph(t)
    p.save("ru_text", f"ru_text_{i + 1:02d}")

for i in range(8):
    p = Page()
    t, plain, tabs = random.sample(TEXTS, 4), random.sample(PLAIN, 2), random.sample(TABLES, 2)
    p.paragraph(plain[0]); p.paragraph(t[0]); p.table(tabs[0])
    p.paragraph(t[1]); p.paragraph(plain[1]); p.paragraph(t[2]); p.table(tabs[1]); p.paragraph(t[3])
    p.save("mixed", f"mixed_{i + 1:02d}")

for i, l in enumerate(FORMULAS):
    fig = plt.figure(figsize=(6, 1.4), dpi=300)
    fig.text(0.5, 0.5, f"${l}$", fontsize=30, ha="center", va="center")
    (ROOT / "formulas_hq").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "formulas_hq" / f"formula_hq_{i + 1:02d}.png", dpi=300)
    plt.close(fig)
    (ROOT / "formulas_hq" / f"formula_hq_{i + 1:02d}.md").write_text(f"$${l}$$", encoding="utf-8")
print("OK")
