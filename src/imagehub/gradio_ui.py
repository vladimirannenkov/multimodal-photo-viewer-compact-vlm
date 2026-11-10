"""Optional Gradio client; the API server must be running separately."""

import io
from html import escape

import httpx
from PIL import Image


def create_ui(base_url="http://127.0.0.1:8765"):
    import gradio as gr

    def single(path, prompt):
        with open(path, "rb") as f:
            response = httpx.post(
                base_url + ("/ask" if prompt.strip() else "/describe"),
                files={"file": f},
                data={"prompt": prompt} if prompt.strip() else {},
                timeout=300,
            )
        response.raise_for_status()
        return response.json()

    def batch(path):
        response = httpx.post(base_url + "/batch", json={"path": path}, timeout=30)
        response.raise_for_status()
        return response.json()

    def status(job):
        response = httpx.get(base_url + "/batch/" + job.strip(), timeout=30)
        response.raise_for_status()
        return response.json()

    def search(query):
        response = httpx.get(base_url + "/search", params={"q": query}, timeout=300)
        response.raise_for_status()
        result = response.json()
        gallery = []
        for item in result["items"]:
            picture = httpx.get(f"{base_url}/api/images/{item['id']}/thumb", timeout=30)
            picture.raise_for_status()
            gallery.append((Image.open(io.BytesIO(picture.content)).copy(), item["name"]))
        return gallery, result

    with gr.Blocks(title="ImageHub") as app:
        gr.Markdown("# ImageHub")
        with gr.Tab("Одна картинка"):
            image = gr.Image(type="filepath", label="Изображение")
            prompt = gr.Textbox(label="Вопрос (пустое поле — описание)")
            answer = gr.JSON(label="Результат")
            gr.Button("Выполнить").click(single, [image, prompt], answer)
        with gr.Tab("Пакет"):
            folder = gr.Textbox(label="Путь к папке на сервере")
            result = gr.JSON(label="Задача")
            gr.Button("Запустить").click(batch, folder, result)
            job = gr.Textbox(label="ID задачи")
            gr.Button("Обновить прогресс").click(status, job, result)
        with gr.Tab("Поиск"):
            query = gr.Textbox(label="Описание")
            gallery = gr.Gallery(label="Найденные изображения")
            results = gr.JSON(label="Найденные изображения")
            gr.Button("Найти").click(search, query, [gallery, results])
        with gr.Tab("Оценки и профиль"):
            url = escape(base_url.rstrip("/") + "/evaluations", quote=True)
            gr.HTML(
                f'<p><a href="{url}" target="_blank" rel="noopener">Открыть оценки на отдельной странице</a></p>'
                f'<iframe src="{url}" title="Оценки и профиль" style="width:100%;height:1000px;border:0"></iframe>'
            )
        with gr.Tab("Настройки"):
            url = escape(base_url.rstrip("/") + "/settings", quote=True)
            gr.HTML(f'<iframe src="{url}" title="Настройки" style="width:100%;height:800px;border:0"></iframe>')
    return app
