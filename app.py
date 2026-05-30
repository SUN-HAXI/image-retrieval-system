import os
import uuid
import time
from flask import Flask, request, jsonify, render_template, send_from_directory

from engine import ImageRetrievalEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "data", "database")
INDEX_DIR = os.path.join(BASE_DIR, "index")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

for d in [DATABASE_DIR, INDEX_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

engine = ImageRetrievalEngine(vocab_size=1000)
_loaded = engine.load_index(INDEX_DIR)

# ── Routes ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify(
        {
            "indexed": len(engine.image_paths),
            "ready": len(engine.image_paths) > 0,
            "loaded": _loaded,
            "index_dir": INDEX_DIR,
        }
    )


@app.route("/images/database/<path:filename>")
def serve_database_image(filename):
    return send_from_directory(DATABASE_DIR, filename)


@app.route("/images/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/search", methods=["POST"])
def search():
    if len(engine.image_paths) == 0:
        return jsonify({"error": "请先构建索引 (POST /build)"}), 400

    query_url = None
    query_name = None

    f = request.files.get("image")
    if f is None:
        # Search by existing database image name
        data = request.get_json(silent=True) or {}
        db_name = data.get("db_image")
        if db_name:
            query_path = os.path.join(DATABASE_DIR, db_name)
            if not os.path.exists(query_path):
                return jsonify({"error": f"图片不存在: {db_name}"}), 404
            query_name = db_name
            query_url = f"/images/database/{db_name}"
        else:
            return jsonify({"error": "请上传查询图片"}), 400
    else:
        ext = os.path.splitext(f.filename or "query.jpg")[1] or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        query_path = os.path.join(UPLOAD_DIR, name)
        f.save(query_path)
        query_name = name
        query_url = f"/images/uploads/{name}"

    t0 = time.time()
    results = engine.search(query_path, k=5)
    elapsed = round((time.time() - t0) * 1000)

    for r in results:
        rel = os.path.relpath(r["path"], DATABASE_DIR)
        r["url"] = f"/images/database/{rel.replace(os.sep, '/')}"
        r["name"] = os.path.basename(r["path"])

    return jsonify(
        {
            "results": results,
            "query_url": query_url,
            "query_name": query_name,
            "elapsed_ms": elapsed,
        }
    )


@app.route("/build", methods=["POST"])
def build():
    try:
        engine.build_index(DATABASE_DIR)
        engine.save_index(INDEX_DIR)
        return jsonify({"ok": True, "count": len(engine.image_paths)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/database-images")
def list_database_images():
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    images = []
    for root, _dirs, files in os.walk(DATABASE_DIR):
        for f in files:
            if f.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(root, f), DATABASE_DIR)
                images.append(rel.replace(os.sep, "/"))
    images.sort()
    return jsonify({"images": images, "count": len(images)})


if __name__ == "__main__":
    print(f"\n  数据库目录: {DATABASE_DIR}")
    print(f"  索引目录:   {INDEX_DIR}")
    print(f"  已加载索引: {'是' if _loaded else '否 – 请先放置图片到 data/database/ 然后点击构建索引'}")
    print(f"\n  打开浏览器访问 → http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
