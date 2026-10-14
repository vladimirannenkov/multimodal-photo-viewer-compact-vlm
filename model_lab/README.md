# model_lab

```powershell
cd C:\ML\Project-Image\model_lab
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python download.py            # показать размеры
python download.py --yes      # скачать веса в weights/
python run.py "C:\path\img.png"                      # все модели
python run.py a.png b.png --only qwen3vl,text_embed  # сравнение описаний
python run.py doc.png --only paddleocr_vl,qwen3_fix  # OCR + правка
```
Результаты: `results/<имя>__<время>/` (report.md, `<модель>.md|json`, `boxes_*.png`).
