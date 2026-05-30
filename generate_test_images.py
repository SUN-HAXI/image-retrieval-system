"""Generate augmented variants of database images for testing (Unicode-safe)."""
import os
import cv2
import numpy as np

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "database")


def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


images = [
    f
    for f in os.listdir(DB)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]
print(f"Found {len(images)} images")

for img_name in images:
    if any(
        tag in img_name
        for tag in ["_flipH", "_flipV", "_rot90", "_crop", "_small", "_rot180"]
    ):
        continue

    path = os.path.join(DB, img_name)
    img = imread_unicode(path)
    if img is None:
        print(f"  SKIP (unreadable): {img_name}")
        continue

    h, w = img.shape[:2]
    base = os.path.splitext(img_name)[0]

    variants = {
        f"{base}_flipH.png": cv2.flip(img, 1),
        f"{base}_flipV.png": cv2.flip(img, 0),
        f"{base}_rot90.png": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        f"{base}_rot180.png": cv2.rotate(img, cv2.ROTATE_180),
        f"{base}_crop.png": img[
            int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)
        ],
        f"{base}_small.png": cv2.resize(
            img, (max(w // 2, 50), max(h // 2, 50))
        ),
    }

    for name, variant in variants.items():
        if variant is not None and variant.size > 0:
            imwrite_unicode(os.path.join(DB, name), variant)
            print(f"  Created: {name}")

total = len(
    [f for f in os.listdir(DB) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
)
print(f"Total images: {total}")
