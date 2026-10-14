import numpy as np
from PIL import ImageDraw


def run(mid, cfg, imgs, device, ctx):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                       if device == "cuda" else ["CPUExecutionProvider"])
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    res = {"images": []}
    embs = []
    for n, img in enumerate(imgs):
        faces = app.get(np.array(img)[:, :, ::-1])
        d = ImageDraw.Draw(img := img.copy())
        items = []
        for f in faces:
            d.rectangle(f.bbox.tolist(), outline="lime", width=3)
            items.append({"bbox": f.bbox.round().tolist(), "det_score": float(f.det_score)})
            embs.append((n, len(items) - 1, f.normed_embedding))
        img.save(ctx["out"] / f"boxes_{mid}_{n + 1}.png")
        res["images"].append({"faces": items})
    sims = [{"a": f"img{a}#{i}", "b": f"img{b}#{j}", "cos": float(ea @ eb)}
            for x, (a, i, ea) in enumerate(embs) for (b, j, eb) in embs[x + 1:]]
    res["similarities"] = sims
    res["markdown"] = "\n".join(f"Изображение {i + 1}: лиц {len(r['faces'])}" for i, r in enumerate(res["images"])) + \
        "\n\n" + "\n".join(f"{s['a']} ~ {s['b']}: {s['cos']:.3f}" for s in sims)
    return res
