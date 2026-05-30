"""Draw the full architecture diagram using matplotlib — works offline, supports Chinese."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

# Use a font that supports Chinese
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.size"] = 8

# ── Color palette ──
C = {
    "storage": "#C8E6C9",
    "preprocess": "#BBDEFB",
    "feature": "#FFCDD2",
    "clustering": "#FFE0B2",
    "index": "#E1BEE7",
    "search": "#B2EBF2",
    "geometry": "#FFCCBC",
    "score": "#FFF9C4",
    "web": "#CFD8DC",
    "border": "#37474F",
    "arrow": "#546E7A",
    "bg": "#FAFAFA",
    "phase_offline": "#E3F2FD",
    "phase_online": "#FFF3E0",
}


def box(ax, x, y, w, h, text, color, fontsize=7, bold=False, ha="center", va="center"):
    """Draw a rounded box with text."""
    weight = "bold" if bold else "normal"
    fbb = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", facecolor=color,
        edgecolor=C["border"], linewidth=0.8, zorder=2,
    )
    ax.add_patch(fbb)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, fontweight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=0.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=1)


def phase_box(ax, x, y, w, h, color, alpha=0.15):
    """Background phase box."""
    rect = mpatches.Rectangle((x, y), w, h, facecolor=color, edgecolor=color,
                               alpha=alpha, linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(rect)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(36, 24))
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # =====================================================================
    # PHASE 1: OFFLINE INDEX BUILDING (left 60%)
    # =====================================================================
    phase_box(ax, 0.3, 0.3, 21.2, 23.4, C["phase_offline"])
    ax.text(10.9, 23.2, "离线阶段：索引构建 (build_index)", fontsize=13, fontweight="bold",
            ha="center", va="center", color="#1565C0")

    # ── Row 1: Image database ──
    y1 = 22.2
    box(ax, 10.9, y1, 5.0, 0.7, "[图像数据库]\n(data/database/ 目录)", C["storage"], bold=True)
    arrow(ax, 10.9, y1 - 0.35, 10.9, y1 - 0.85)

    # ── Row 2: File scanning ──
    y2 = 21.0
    box(ax, 10.9, y2, 5.5, 0.7, "os.walk 遍历目录\n筛选 .jpg/.jpeg/.png/.bmp/.gif/.webp", C["preprocess"])
    arrow(ax, 10.9, y2 - 0.35, 10.9, y2 - 0.85)

    # ── Row 3: Sort ──
    y3 = 19.7
    box(ax, 10.9, y3, 3.5, 0.5, "排序 → image_paths 列表", C["preprocess"])

    # ── Split to two pipes ──
    arrow(ax, 10.9, y3 - 0.25, 5.5, 18.5)
    arrow(ax, 10.9, y3 - 0.25, 16.3, 18.5)

    # ============================
    # LEFT PIPE: SIFT + BoW
    # ============================
    sx, sy = 2.8, 18.3
    ax.text(sx + 1.8, sy + 0.3, "SIFT 局部特征管线", fontsize=9, fontweight="bold",
            ha="center", color="#C62828")

    steps_sift = [
        ("cv2.imdecode\n(Unicode 安全读取)", C["preprocess"]),
        ("cv2.cvtColor\n(BGR → GRAY)", C["preprocess"]),
        ("cv2.SIFT_create()\nSIFT 特征检测器初始化", C["feature"]),
        ("sift.detectAndCompute()\n关键点检测\n128维描述符提取", C["feature"]),
        ("收集所有描述符\n(全部图片合并)", C["feature"]),
    ]
    for i, (text, color) in enumerate(steps_sift):
        y = sy - i * 1.2
        box(ax, sx + 1.8, y, 4.0, 0.85, text, color, fontsize=6.5)
        if i > 0:
            arrow(ax, sx + 1.8, y + 0.43, sx + 1.8, y - 0.43)

    # ── Vocabulary building ──
    vy = sy - len(steps_sift) * 1.2 - 0.4
    ax.text(sx + 1.8, vy + 0.2, "视觉词汇表构建", fontsize=9, fontweight="bold",
            ha="center", color="#E65100")

    vocab_steps = [
        ("np.vstack\n堆叠所有 SIFT 描述符", C["preprocess"]),
        ("np.random.choice\n随机采样 ≤150,000 个", C["preprocess"]),
        ("sklearn.cluster.MiniBatchKMeans\nn_clusters=1000, batch_size=2000\nn_init=3, random_state=42", C["clustering"]),
        ("kmeans.fit()\n训练聚类中心 → 视觉词汇", C["clustering"]),
    ]
    for i, (text, color) in enumerate(vocab_steps):
        y = vy - 0.6 - i * 1.15
        box(ax, sx + 1.8, y, 4.5, 0.85, text, color, fontsize=6)
        arrow(ax, sx + 1.8, y + 0.43, sx + 1.8, y - 0.43)

    # ── BoW encoding ──
    be_y = vy - 0.6 - len(vocab_steps) * 1.15 - 0.3
    ax.text(sx + 1.8, be_y + 0.1, "BoW 编码 (每张图片)", fontsize=9, fontweight="bold",
            ha="center", color="#E65100")

    bow_enc = [
        ("kmeans.predict()\n描述符 → 最近视觉词分配", C["clustering"]),
        ("np.bincount\n词频直方图 histogram", C["clustering"]),
        ("TF 归一化\nhist / (sum(hist) + 1e-8)", C["clustering"]),
        ("IDF 计算\nlog((N+1)/(df+1)) + 1.0", C["clustering"]),
        ("TF × IDF 加权", C["clustering"]),
        ("sklearn.preprocessing.normalize\nL2 归一化 (norm='l2')", C["clustering"]),
    ]
    bow_last_y = be_y
    for i, (text, color) in enumerate(bow_enc):
        y = be_y - 0.6 - i * 1.0
        bow_last_y = y
        box(ax, sx + 1.8, y, 4.0, 0.75, text, color, fontsize=6)
        arrow(ax, sx + 1.8, y + 0.38, sx + 1.8, y - 0.38)

    # ============================
    # RIGHT PIPE: ResNet50 Deep Features
    # ============================
    dx, dy = 13.0, 18.3
    ax.text(dx + 2.5, dy + 0.3, "ResNet50 深度特征管线", fontsize=9, fontweight="bold",
            ha="center", color="#C62828")

    deep_steps = [
        ("PIL.Image.open\n(RGB 读取)", C["preprocess"]),
        ("T.Resize(256)\n缩放至 256×256", C["preprocess"]),
        ("T.CenterCrop(224)\n中心裁剪 224×224", C["preprocess"]),
        ("T.ToTensor()\n转为张量 [0, 1]", C["preprocess"]),
        ("T.Normalize\nmean=[0.485,0.456,0.406]\nstd=[0.229,0.224,0.225]", C["preprocess"]),
        ("torch.nn.Sequential\nResNet50 去掉最后 FC 层\nweights=IMAGENET1K_V2", C["feature"]),
        ("model.eval()\ntorch.no_grad() 前向推理", C["feature"]),
        ("np.linalg.norm\nL2 归一化 → 2048-d 向量", C["feature"]),
    ]
    deep_last_y = dy
    for i, (text, color) in enumerate(deep_steps):
        y = dy - i * 1.2
        deep_last_y = y
        box(ax, dx + 2.5, y, 4.5, 0.85, text, color, fontsize=6.5)
        if i > 0:
            arrow(ax, dx + 2.5, y + 0.43, dx + 2.5, y - 0.43)

    # ── FAISS index building ──
    fi_y = min(bow_last_y, deep_last_y) - 0.8
    ax.text(10.9, fi_y + 0.15, "FAISS 双索引构建", fontsize=10, fontweight="bold",
            ha="center", color="#6A1B9A")

    # BoW index (left)
    box(ax, 6.0, fi_y - 0.6, 5.0, 0.55, "faiss.IndexFlatL2\n创建 L2 距离索引 (BoW 直方图维度)", C["index"], fontsize=6.5)
    arrow(ax, 6.0, fi_y - 0.15, 6.0, fi_y - 0.87)
    box(ax, 6.0, fi_y - 1.4, 4.0, 0.55, "bow_index.add()\n写入所有 BoW 向量", C["index"], fontsize=6.5)
    arrow(ax, 6.0, fi_y - 0.87, 6.0, fi_y - 1.67)

    # Deep index (right)
    box(ax, 15.8, fi_y - 0.6, 5.0, 0.55, "faiss.IndexFlatIP\n创建内积相似度索引 (2048维)", C["index"], fontsize=6.5)
    arrow(ax, 15.8, fi_y - 0.15, 15.8, fi_y - 0.87)
    box(ax, 15.8, fi_y - 1.4, 4.0, 0.55, "deep_index.add()\n写入所有 Deep 向量", C["index"], fontsize=6.5)
    arrow(ax, 15.8, fi_y - 0.87, 15.8, fi_y - 1.67)

    # Arrow from BoW encoding to FAISS
    arrow(ax, sx + 1.8, bow_last_y - 0.38, 6.0, fi_y - 0.15)
    # Arrow from Deep features to FAISS
    arrow(ax, dx + 2.5, deep_last_y - 0.43, 15.8, fi_y - 0.15)

    # ── Persistence ──
    ps_y = fi_y - 2.2
    ax.text(10.9, ps_y, "索引持久化 (save_index)", fontsize=9, fontweight="bold",
            ha="center", color="#37474F")

    pers_items = [
        ("pickle.dump → bow_data.pkl\n(kmeans, idf, vocab_size, image_paths)", C["storage"]),
        ("faiss.serialize_index\nbow_index → bow.faiss", C["storage"]),
        ("faiss.serialize_index\ndeep_index → deep.faiss", C["storage"]),
    ]
    for i, (text, color) in enumerate(pers_items):
        y = ps_y - 0.55 - i * 0.7
        box(ax, 10.9, y, 6.5, 0.55, text, color, fontsize=6.5)

    # arrows to persistence
    arrow(ax, 6.0, fi_y - 1.67, 10.9, ps_y - 0.15)
    arrow(ax, 15.8, fi_y - 1.67, 10.9, ps_y - 0.15)

    # =====================================================================
    # PHASE 2: ONLINE QUERY (right 42%)
    # =====================================================================
    phase_box(ax, 21.7, 0.3, 14.0, 23.4, C["phase_online"])
    ax.text(28.7, 23.2, "在线阶段：查询检索 (search)", fontsize=13, fontweight="bold",
            ha="center", va="center", color="#E65100")

    # Query input
    qy = 22.0
    box(ax, 28.7, qy, 5.5, 0.7, " 用户上传 / 选择库内图片", C["web"], bold=True, fontsize=7)
    arrow(ax, 28.7, qy - 0.35, 28.7, qy - 0.8)

    box(ax, 28.7, qy - 1.1, 4.0, 0.5, "Flask request 接收请求", C["web"])
    arrow(ax, 28.7, qy - 1.35, 28.7, qy - 1.8)

    # Split query reading
    box(ax, 26.0, qy - 2.2, 3.5, 0.5, "cv2.imdecode\n(Unicode 安全读取)", C["preprocess"], fontsize=6.5)
    box(ax, 31.4, qy - 2.2, 3.5, 0.5, "PIL.Image.open\n(RGB 读取)", C["preprocess"], fontsize=6.5)
    arrow(ax, 28.7, qy - 1.55, 26.0, qy - 2.45)
    arrow(ax, 28.7, qy - 1.55, 31.4, qy - 2.45)

    # ── Query SIFT pipeline ──
    qsy = qy - 3.4
    ax.text(26.0, qsy + 0.25, "SIFT + BoW 查询编码", fontsize=8, fontweight="bold",
            ha="center", color="#C62828")

    qsift_steps = [
        ("cv2.cvtColor\nBGR → GRAY", C["preprocess"]),
        ("cv2.SIFT_create()\nsift.detectAndCompute()\n关键点 + 128维描述符", C["feature"]),
        ("kmeans.predict()\n描述符 → 视觉词分配", C["clustering"]),
        ("np.bincount\n词频直方图", C["clustering"]),
        ("TF归一化\nhist/sum(hist)", C["clustering"]),
        ("IDF加权\nhist × idf", C["clustering"]),
        ("np.linalg.norm\nL2归一化", C["clustering"]),
    ]
    qsift_last = qsy
    for i, (text, color) in enumerate(qsift_steps):
        y = qsy - 0.55 - i * 0.85
        qsift_last = y
        box(ax, 26.0, y, 3.8, 0.65, text, color, fontsize=5.8)
        arrow(ax, 26.0, y + 0.33, 26.0, y - 0.33)

    arrow(ax, 26.0, qy - 2.45, 26.0, qsy - 0.25)

    # ── Query Deep pipeline ──
    qdy = qy - 3.4
    ax.text(31.4, qdy + 0.25, "ResNet50 查询编码", fontsize=8, fontweight="bold",
            ha="center", color="#C62828")

    qdeep_steps_1 = [
        ("PIL.Image.open → RGB", C["preprocess"]),
        ("T.Resize(256)", C["preprocess"]),
        ("T.CenterCrop(224)", C["preprocess"]),
        ("T.ToTensor()", C["preprocess"]),
        ("T.Normalize(ImageNet stats)", C["preprocess"]),
        ("ResNet50 (no FC)\nmodel.eval()\ntorch.no_grad()", C["feature"]),
        ("np.linalg.norm\nL2归一化 → 2048-d", C["feature"]),
    ]
    qdeep_last_1 = qdy
    for i, (text, color) in enumerate(qdeep_steps_1):
        y = qdy - 0.55 - i * 0.85
        qdeep_last_1 = y
        box(ax, 31.4, y, 3.5, 0.65, text, color, fontsize=5.8)
        arrow(ax, 31.4, y + 0.33, 31.4, y - 0.33)

    arrow(ax, 31.4, qy - 2.45, 31.4, qdy - 0.25)

    # Flip branch
    flip_y = qdeep_last_1 - 0.5
    box(ax, 31.4, flip_y, 3.5, 0.5, "PIL.Image.FLIP_LEFT_RIGHT\n水平翻转", C["preprocess"], fontsize=6.5, bold=True)
    arrow(ax, 31.4, qdeep_last_1 - 0.33, 31.4, flip_y - 0.25)

    flip_steps = [
        ("T.Resize(256) → CenterCrop(224)\nToTensor() → Normalize", C["preprocess"]),
        ("ResNet50 (no FC) → L2归一化\n翻转图 2048-d 特征", C["feature"]),
    ]
    flip_last = flip_y
    for i, (text, color) in enumerate(flip_steps):
        y = flip_y - 0.55 - i * 0.8
        flip_last = y
        box(ax, 31.4, y, 3.5, 0.6, text, color, fontsize=5.8)
        arrow(ax, 31.4, y + 0.3, 31.4, y - 0.3)

    # ── FAISS Search ──
    fs_y = min(qsift_last, flip_last) - 0.7
    ax.text(28.7, fs_y + 0.15, "FAISS 双索引检索", fontsize=9, fontweight="bold",
            ha="center", color="#006064")

    # Bow search
    box(ax, 25.5, fs_y - 0.5, 4.0, 0.55, "faiss.IndexFlatL2.search()\nBoW L2 距离搜索 (k×20)", C["search"], fontsize=6)
    arrow(ax, 26.0, qsift_last - 0.33, 25.5, fs_y - 0.15)
    box(ax, 25.5, fs_y - 1.2, 4.0, 0.55, "L2→相似度: bow_sim = 1/(1+d)", C["score"], fontsize=6)
    arrow(ax, 25.5, fs_y - 0.77, 25.5, fs_y - 1.47)

    # Deep search (original)
    box(ax, 29.5, fs_y - 0.5, 4.0, 0.55, "faiss.IndexFlatIP.search()\nDeep 内积搜索 (原始图)", C["search"], fontsize=6)
    arrow(ax, 31.4, qdeep_last_1 - 0.33, 29.5, fs_y - 0.15)

    # Deep search (flipped)
    box(ax, 33.3, fs_y - 0.5, 4.0, 0.55, "faiss.IndexFlatIP.search()\nDeep 内积搜索 (翻转图)", C["search"], fontsize=6)
    arrow(ax, 31.4, flip_last - 0.3, 33.3, fs_y - 0.15)

    # Score normalization
    score_y = fs_y - 2.0
    box(ax, 28.7, score_y, 5.0, 0.55, "np.maximum → 原图/翻转取 max 相似度", C["score"], fontsize=6.5)
    arrow(ax, 29.5, fs_y - 0.77, 28.7, score_y - 0.28)
    arrow(ax, 33.3, fs_y - 0.77, 28.7, score_y - 0.28)

    box(ax, 28.7, score_y - 0.8, 5.0, 0.55, "Min-Max 归一化\ndeep_sim = (d-d_min)/(d_max-d_min)", C["score"], fontsize=6.5)
    arrow(ax, 28.7, score_y - 0.28, 28.7, score_y - 1.08)

    # ── Score Fusion ──
    fuse_y = score_y - 1.6
    ax.text(28.7, fuse_y + 0.15, "多模态分数融合", fontsize=9, fontweight="bold",
            ha="center", color="#F57F17")

    box(ax, 28.7, fuse_y - 0.55, 7.0, 0.55,
        "加权求和: bow_weight(0.35)×bow_sim + deep_weight(0.65)×deep_sim",
        C["score"], fontsize=6.5)
    arrow(ax, 25.5, fs_y - 1.47, 28.7, fuse_y - 0.15)
    arrow(ax, 28.7, score_y - 1.08, 28.7, fuse_y - 0.15)
    arrow(ax, 28.7, fuse_y - 0.83, 28.7, fuse_y - 1.15)

    box(ax, 28.7, fuse_y - 1.5, 4.5, 0.5, "sorted(reverse=True)\n按融合分数降序排列", C["score"], fontsize=6)
    arrow(ax, 28.7, fuse_y - 1.75, 28.7, fuse_y - 2.2)

    box(ax, 28.7, fuse_y - 2.55, 4.0, 0.5, "取前 15 个候选\n进入几何验证", C["score"], fontsize=6, bold=True)

    # ── Geometric Verification ──
    gv_y = fuse_y - 3.35
    ax.text(28.7, gv_y + 0.2, "几何空间验证 (前 15 候选逐一执行)", fontsize=9, fontweight="bold",
            ha="center", color="#BF360C")

    geo_steps = [
        ("cv2.imdecode → 读取候选图片", C["preprocess"]),
        ("cv2.cvtColor(BGR→GRAY) ×2\n查询图 + 候选图 分别转灰度", C["preprocess"]),
        ("cv2.SIFT_create()\nsift.detectAndCompute() ×2\n查询图关键点+描述符\n候选图关键点+描述符", C["feature"]),
        ("cv2.FlannBasedMatcher\n(algorithm=1 KDTree, trees=5)\n(checks=50)", C["geometry"]),
        ("flann.knnMatch(k=2)\n每个查询关键点找最近 2 个匹配", C["geometry"]),
        ("Lowe's Ratio Test\nm.distance < 0.75×n.distance\n筛选 good matches", C["geometry"]),
        ("good matches ≥ 10?\n否 → geo_score=0.0", C["geometry"]),
        ("提取匹配点对坐标\nsrc_pts, dst_pts", C["geometry"]),
        ("cv2.findHomography\n(RANSAC, threshold=5.0)\n计算单应矩阵 M + mask", C["geometry"]),
        ("M≠None 且 inliers≥8?\n否 → geo_score=0.0", C["geometry"]),
        ("inlier_ratio = inliers/len(good)\nmatch_ratio = len(good)/min(len(q_kp),len(c_kp))", C["geometry"]),
        ("geo_score =\n0.7×inlier_ratio + 0.3×match_ratio", C["geometry"]),
    ]
    geo_last = gv_y
    for i, (text, color) in enumerate(geo_steps):
        y = gv_y - 0.5 - i * 0.75
        geo_last = y
        box(ax, 28.7, y, 5.4, 0.58, text, color, fontsize=5.3)
        arrow(ax, 28.7, y + 0.29, 28.7, y - 0.29)

    arrow(ax, 28.7, fuse_y - 2.8, 28.7, gv_y - 0.2)

    # ── Final Score ──
    final_y = geo_last - 0.7
    ax.text(28.7, final_y + 0.15, "最终排序与输出", fontsize=9, fontweight="bold",
            ha="center", color="#37474F")

    final_steps = [
        ("已验证候选: final_score = 0.6×fusion + 0.4×geo", C["score"]),
        ("未验证候选: final_score = fusion × 0.75 (惩罚因子)", C["score"]),
        ("sorted(reverse=True) → 取 top-k (k=5)", C["score"]),
        ("返回 JSON\n{results, query_url, elapsed_ms}", C["web"]),
    ]
    final_last = final_y
    for i, (text, color) in enumerate(final_steps):
        y = final_y - 0.5 - i * 0.7
        final_last = y
        box(ax, 28.7, y, 5.5, 0.55, text, color, fontsize=6)
        arrow(ax, 28.7, y + 0.28, 28.7, y - 0.28)

    arrow(ax, 28.7, geo_last - 0.29, 28.7, final_y - 0.15)

    # ── Load index arrow (offline → online) ──
    ax.annotate("加载索引\npickle.load + faiss.deserialize_index",
                xy=(21.7, 10), xytext=(19.5, 10),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
                fontsize=8, fontweight="bold", ha="center", va="center", color="#1565C0",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1565C0", alpha=0.9))

    # ── Web layer ──
    web_x, web_y = 28.7, final_last - 1.2
    ax.text(web_x, web_y + 0.1, " Web 服务层 (Flask)", fontsize=9, fontweight="bold",
            ha="center", color="#37474F")

    web_items = [
        "Flask(__name__)  app.run(host='0.0.0.0', port=5000, debug=True)",
        "GET / → render_template('index.html')  前端界面",
        "GET /status → 索引状态 JSON",
        "POST /search → 图片检索 (max 32MB)  uuid.uuid4().hex 唯一命名",
        "POST /build → 触发索引构建",
        "GET /database-images → 列出库内图片  os.path.relpath",
        "GET /images/database/<name> → send_from_directory",
        "GET /images/uploads/<name> → send_from_directory",
        "os.makedirs(..., exist_ok=True) 自动创建目录",
    ]
    for i, item in enumerate(web_items):
        y = web_y - 0.45 - i * 0.55
        box(ax, web_x, y, 8.0, 0.45, item, C["web"], fontsize=5.5)

    # =====================================================================
    # LEGEND
    # =====================================================================
    ly = 1.5
    legend_items = [
        ("[存储] 数据/存储", C["storage"]),
        ("[预处理] 预处理", C["preprocess"]),
        ("[特征] 特征提取", C["feature"]),
        ("[聚类] 聚类/编码", C["clustering"]),
        ("[索引] FAISS索引", C["index"]),
        ("[检索] 检索/匹配", C["search"]),
        ("[几何] 几何验证", C["geometry"]),
        ("[融合] 分数融合", C["score"]),
        ("[Web] Web/API", C["web"]),
    ]
    ax.text(11.0, ly + 0.6, "图例", fontsize=9, fontweight="bold", ha="center")
    for i, (label, color) in enumerate(legend_items):
        x = 3.0 + (i % 5) * 4.2
        y = ly - (i // 5) * 0.5
        box(ax, x, y, 3.5, 0.4, label, color, fontsize=6.5)

    # =====================================================================
    # Title
    # =====================================================================
    ax.text(18.0, 23.8, "大规模图像检索系统 — 完整模型流程与结构图 (SIFT + BoW + ResNet50 + FAISS + 几何验证)",
            fontsize=14, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    # =====================================================================
    # Save
    # =====================================================================
    out_path = "C:/Users/sun/Desktop/测试/image_retrieval/model_architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
