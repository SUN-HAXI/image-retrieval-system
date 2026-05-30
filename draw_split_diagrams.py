"""Split the architecture diagram into 5 clear, readable parts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.size"] = 9

# ── Shared color palette ──
C = {
    "storage":    "#C8E6C9",
    "preprocess": "#BBDEFB",
    "feature":    "#FFCDD2",
    "clustering": "#FFE0B2",
    "index":      "#E1BEE7",
    "search":     "#B2EBF2",
    "geometry":   "#FFCCBC",
    "score":      "#FFF9C4",
    "web":        "#CFD8DC",
    "border":     "#37474F",
    "arrow":      "#546E7A",
    "bg":         "#FAFAFA",
    "title_off":  "#1565C0",
    "title_on":   "#E65100",
    "green":      "#2E7D32",
    "orange":     "#E65100",
    "purple":     "#6A1B9A",
    "teal":       "#006064",
    "red":        "#C62828",
    "gold":       "#F57F17",
    "deepred":    "#BF360C",
}


def box(ax, x, y, w, h, text, color, fontsize=8, bold=False, ha="center", va="center", alpha=1.0):
    fbb = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12", facecolor=color,
        edgecolor=C["border"], linewidth=0.8, zorder=2, alpha=alpha,
    )
    ax.add_patch(fbb)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=0.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)


def section_label(ax, x, y, text, color, fontsize=11):
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold", ha="center", va="center", color=color)


def dashed_box(ax, x, y, w, h, color, alpha=0.08, label="", fontsize=9):
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor=color,
                      alpha=alpha, linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h - 0.25, label, fontsize=fontsize, fontweight="bold",
                ha="center", va="top", color=color)


def new_figure(w=20, h=14):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    return fig, ax


def legend_bar(ax, x, y, items):
    for i, (label, color) in enumerate(items):
        bx = x + i * 2.3
        box(ax, bx, y, 2.0, 0.4, label, color, fontsize=6.5)
    return bx + 1.5


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1 — 系统总体架构                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def part1_overview():
    fig, ax = new_figure(24, 16)
    w, h = 24, 16

    ax.text(w/2, h - 0.6, "大规模图像检索系统 — 总体架构", fontsize=16, fontweight="bold",
            ha="center", va="center")

    # ── Offline phase ──
    dashed_box(ax, 0.3, 0.5, 12.0, 14.3, C["title_off"], label="离线阶段：索引构建")
    # ── Online phase ──
    dashed_box(ax, 12.6, 0.5, 11.1, 14.3, C["title_on"], label="在线阶段：查询检索")

    # Offline blocks
    off_blocks = [
        (6.3, 13.6, 6.0, 0.9, "图像数据库\n(.jpg/.png/.bmp/...)", C["storage"], True),
        (6.3, 12.0, 6.0, 0.9, "SIFT 关键点 + 描述符提取\ncv2.SIFT_create() + detectAndCompute()", C["feature"], False),
        (6.3, 10.3, 6.0, 0.9, "视觉词汇表构建\nMiniBatchKMeans (k=1000)", C["clustering"], False),
        (6.3, 8.6, 6.0, 0.9, "BoW + TF-IDF 编码\nnp.bincount + IDF加权 + L2归一化", C["clustering"], False),
        (3.5, 6.8, 4.5, 0.9, "FAISS IndexFlatL2\n(BoW 索引)", C["index"], False),
        (9.1, 6.8, 4.5, 0.9, "FAISS IndexFlatIP\n(Deep 索引)", C["index"], False),
        (6.3, 5.0, 6.0, 0.9, "ResNet50 深度特征\n2048-d L2 归一化向量", C["feature"], False),
        (6.3, 3.2, 6.0, 0.9, "索引持久化\npickle + faiss.serialize_index", C["storage"], False),
        (6.3, 1.6, 6.0, 0.9, "bow_data.pkl / bow.faiss / deep.faiss", C["storage"], False),
    ]
    for x, y, bw, bh, text, color, bold in off_blocks:
        box(ax, x, y, bw, bh, text, color, fontsize=7.5, bold=bold)

    # Arrows for offline
    for i in range(len(off_blocks) - 1):
        y1 = off_blocks[i][1] - off_blocks[i][3]/2
        y2 = off_blocks[i+1][1] + off_blocks[i+1][3]/2
        arrow(ax, 6.3, y1, 6.3, y2)
    # Special arrows: BoW -> FAISS, Deep -> FAISS
    arrow(ax, 6.3, 8.6 - 0.45, 3.5, 6.8 + 0.45)
    arrow(ax, 6.3, 5.0 - 0.45, 9.1, 6.8 + 0.45)


    # Online blocks
    on_blocks = [
        (18.15, 13.6, 6.0, 0.9, "查询图片上传\n(Flask request.files)", C["web"], True),
        (18.15, 12.0, 6.0, 0.9, "SIFT + BoW 查询编码\nkmeans.predict + TF-IDF + L2", C["clustering"], False),
        (15.2, 10.3, 4.5, 0.9, "FAISS IndexFlatL2\nBoW L2 距离搜索", C["search"], False),
        (21.1, 10.3, 4.5, 0.9, "FAISS IndexFlatIP\nDeep 内积搜索 (原图+翻转)", C["search"], False),
        (18.15, 8.6, 6.0, 0.9, "ResNet50 深度特征\n(原图 + FLIP_LEFT_RIGHT)", C["feature"], False),
        (18.15, 6.9, 6.0, 0.9, "多模态分数融合\n0.35*BoW + 0.65*Deep", C["score"], False),
        (18.15, 5.2, 6.0, 0.9, "几何空间验证 (Top-15)\nFLANN + Lowe's test + RANSAC", C["geometry"], False),
        (18.15, 3.5, 6.0, 0.9, "最终评分排序\n0.6*fusion + 0.4*geo → Top-K", C["score"], False),
        (18.15, 1.8, 6.0, 0.9, "返回 JSON 结果\n{results, query_url, elapsed_ms}", C["web"], False),
    ]
    for x, y, bw, bh, text, color, bold in on_blocks:
        box(ax, x, y, bw, bh, text, color, fontsize=7.5, bold=bold)

    for i in range(len(on_blocks) - 1):
        y1 = on_blocks[i][1] - on_blocks[i][3]/2
        y2 = on_blocks[i+1][1] + on_blocks[i+1][3]/2
        arrow(ax, 18.15, y1, 18.15, y2)

    # Cross-phase arrow
    ax.annotate("pickle.load\nfaiss.deserialize_index",
                xy=(13.5, 6.0), xytext=(10.5, 6.0),
                arrowprops=dict(arrowstyle="->", color=C["title_off"], lw=1.5),
                fontsize=8, fontweight="bold", ha="center", va="center", color=C["title_off"],
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=C["title_off"]))

    legend_bar(ax, 4, h - 1.6, [
        ("存储", C["storage"]), ("预处理", C["preprocess"]), ("特征提取", C["feature"]),
        ("聚类/编码", C["clustering"]), ("FAISS索引", C["index"]), ("检索匹配", C["search"]),
        ("几何验证", C["geometry"]), ("分数融合", C["score"]), ("Web/API", C["web"]),
    ])

    fig.savefig("C:/Users/sun/Desktop/测试/image_retrieval/part1_overview.png",
                dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("Part 1 saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2 — SIFT + BoW 特征提取管线                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def part2_sift_bow():
    fig, ax = new_figure(22, 18)
    w, h = 22, 18

    ax.text(w/2, h - 0.6, "SIFT + BoW 特征提取与编码管线", fontsize=15, fontweight="bold", ha="center")

    # ── Left: SIFT feature extraction ──
    dashed_box(ax, 0.3, 8.5, 10.0, 8.5, C["red"], label="阶段 1: SIFT 局部特征提取")
    dashed_box(ax, 0.3, 0.5, 10.0, 7.5, C["orange"], label="阶段 2: 视觉词汇表构建")

    # SIFT steps (vertical chain)
    sift_y = 16.2
    sift_steps = [
        ("读取图片\ncv2.imdecode + np.frombuffer\n(Unicode 安全, Windows 兼容)", C["preprocess"]),
        ("BGR -> 灰度\ncv2.cvtColor(img, COLOR_BGR2GRAY)", C["preprocess"]),
        ("初始化 SIFT 检测器\ncv2.SIFT_create()\n(专利已过期, OpenCV >= 4.4)", C["feature"]),
        ("关键点检测 + 描述符提取\nsift.detectAndCompute(gray, None)\n每个关键点 -> 128维 SIFT 向量", C["feature"]),
        ("收集所有描述符\n将全部图片的 SIFT 描述符合并\n(为聚类做准备)", C["feature"]),
    ]
    for i, (text, color) in enumerate(sift_steps):
        y = sift_y - i * 1.5
        box(ax, 5.3, y, 6.5, 1.05, text, color, fontsize=7)
        if i < len(sift_steps) - 1:
            arrow(ax, 5.3, y - 0.53, 5.3, y - 1.0)

    # ── Right: Vocabulary building ──
    dashed_box(ax, 10.8, 8.5, 11.0, 8.5, C["orange"], alpha=0.06, label="")
    dashed_box(ax, 10.8, 0.5, 11.0, 7.5, C["purple"], label="阶段 3: BoW 编码 (每张图片)")

    vocab_y = 16.2
    vocab_steps = [
        ("堆叠描述符矩阵\nnp.vstack(all_descriptors)\n→ 大型矩阵 (N_des × 128)", C["preprocess"]),
        ("随机采样 (控制规模)\nnp.random.choice\n≤ 150,000 个描述符", C["preprocess"]),
        ("Mini-Batch K-Means 聚类\nsklearn.cluster.MiniBatchKMeans\nn_clusters = vocab_size = 1000\nbatch_size = 2000, n_init = 3\nrandom_state = 42", C["clustering"]),
        ("训练聚类中心\nkmeans.fit(sampled_descriptors)\n→ 1000 个视觉词 (Visual Words)\n每个视觉词 = 128维聚类中心", C["clustering"]),
        ("描述符 -> 视觉词分配\nkmeans.predict(descriptors)\n每个描述符归入最近的视觉词", C["clustering"]),
    ]
    for i, (text, color) in enumerate(vocab_steps):
        y = vocab_y - i * 1.5
        box(ax, 16.3, y, 6.5, 1.05, text, color, fontsize=7)
        if i < len(vocab_steps) - 1:
            arrow(ax, 16.3, y - 0.53, 16.3, y - 1.0)

    # Connector between SIFT and vocab
    arrow(ax, 8.5, 9.0, 13.0, 16.2)

    # ── BoW encoding (below) ──
    bow_y = 7.0
    bow_steps = [
        ("构建词频直方图\nwords = kmeans.predict(des)\nhist = np.bincount(words, minlength=1000)\n→ 1000 维稀疏向量", C["clustering"]),
        ("TF (Term Frequency) 归一化\ntf = hist / (sum(hist) + 1e-8)\n消除图片关键点数量差异的影响", C["clustering"]),
        ("IDF (Inverse Document Frequency)\ndf = (histograms > 0).sum(axis=0)\nidf = log((N+1)/(df+1)) + 1.0\n抑制常见视觉词, 增强稀有词权重", C["clustering"]),
        ("TF × IDF 加权\ntfidf = tf × idf\n结合词频与逆文档频率", C["clustering"]),
        ("L2 归一化\nsklearn.preprocessing.normalize(tfidf, norm='l2')\n使向量位于单位超球面上", C["clustering"]),
    ]
    for i, (text, color) in enumerate(bow_steps):
        y = bow_y - i * 1.1
        box(ax, 16.3, y, 7.5, 0.82, text, color, fontsize=6.5)
        if i < len(bow_steps) - 1:
            arrow(ax, 16.3, y - 0.41, 16.3, y - 0.69)

    # ── Right side: BoW → FAISS ──
    faiss_y = bow_y - len(bow_steps) * 1.1 - 0.4
    box(ax, 16.3, faiss_y, 7.0, 0.85,
        "FAISS 索引构建\nfaiss.IndexFlatL2 (L2 精确搜索)\nbow_index.add(bow_vectors)\n→ 写入所有 L2 归一化的 BoW 向量",
        C["index"], fontsize=7, bold=True)
    arrow(ax, 16.3, bow_y - len(bow_steps) * 1.1 + 0.41, 16.3, faiss_y + 0.42)

    # ── Arrow from vocab to BoW ──
    arrow(ax, 16.3, vocab_y - len(vocab_steps) * 1.5 - 0.25, 16.3, bow_y + 0.42)

    legend_bar(ax, 3, h - 1.5, [
        ("预处理", C["preprocess"]), ("特征提取", C["feature"]),
        ("聚类/编码", C["clustering"]), ("FAISS索引", C["index"]),
    ])

    fig.savefig("C:/Users/sun/Desktop/测试/image_retrieval/part2_sift_bow.png",
                dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("Part 2 saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3 — ResNet50 深度特征管线                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def part3_deep():
    fig, ax = new_figure(20, 17)
    w, h = 20, 17

    ax.text(w/2, h - 0.6, "ResNet50 深度语义特征提取管线", fontsize=15, fontweight="bold", ha="center")

    # Left: Preprocessing
    dashed_box(ax, 0.3, 0.5, 9.2, 15.0, C["title_off"], label="图像预处理 (torchvision.transforms)")

    pp_y = 15.5
    pp_steps = [
        ("读取图片\nPIL.Image.open(path).convert('RGB')\n确保 3 通道 RGB 格式", C["preprocess"]),
        ("Resize 缩放\nT.Resize(256)\n将短边缩放到 256 像素\n保持宽高比", C["preprocess"]),
        ("CenterCrop 中心裁剪\nT.CenterCrop(224)\n从中心裁出 224×224\n标准 ImageNet 输入尺寸", C["preprocess"]),
        ("ToTensor 转张量\nT.ToTensor()\nPIL Image [0,255] -> Tensor [0,1]\n(H, W, C) -> (C, H, W)", C["preprocess"]),
        ("Normalize 标准化\nT.Normalize(\n  mean=[0.485, 0.456, 0.406]\n  std=[0.229, 0.224, 0.225])\nImageNet 训练集的统计值", C["preprocess"]),
    ]
    for i, (text, color) in enumerate(pp_steps):
        y = pp_y - i * 2.0
        box(ax, 5.0, y, 7.0, 1.5, text, color, fontsize=7)
        if i < len(pp_steps) - 1:
            arrow(ax, 5.0, y - 0.75, 5.0, y - 1.25)

    # Right: ResNet50
    dashed_box(ax, 9.8, 0.5, 10.0, 15.0, C["red"], label="ResNet50 特征提取器")

    rn_y = 15.5
    rn_steps = [
        ("模型加载\nmodels.resnet50(\n  weights=ResNet50_Weights.IMAGENET1K_V2)\n预训练权重 (ImageNet 1000类)", C["feature"]),
        ("去除全连接层\ntorch.nn.Sequential(\n  *list(backbone.children())[:-1])\n保留 Conv1~AvgPool\n输出: 2048-d 特征图", C["feature"]),
        ("设备部署\nmodel.to(device)\nCUDA (GPU) 或 CPU\nmodel.eval() 推理模式", C["feature"]),
        ("前向推理\nwith torch.no_grad():\n  feat = model(tensor)\n禁用梯度计算, 节省显存", C["feature"]),
        ("L2 归一化\nfeat / (||feat|| + 1e-8)\n单位化到超球面\n→ 2048维特征向量", C["feature"]),
    ]
    for i, (text, color) in enumerate(rn_steps):
        y = rn_y - i * 2.0
        box(ax, 15.0, y, 7.0, 1.5, text, color, fontsize=7)
        if i < len(rn_steps) - 1:
            arrow(ax, 15.0, y - 0.75, 15.0, y - 1.25)

    # Connector
    arrow(ax, 8.5, pp_y - len(pp_steps)*2.0 + 1.0, 11.5, rn_y)

    # FAISS
    fi_y = pp_y - len(pp_steps) * 2.0 - 1.0
    box(ax, 10.0, fi_y, 8.5, 0.9,
        "FAISS 索引构建\nfaiss.IndexFlatIP (内积相似度搜索)\ndeep_index.add(deep_features)\n→ 内积相似度等价于 L2 归一化后的余弦相似度",
        C["index"], fontsize=7.5, bold=True)
    arrow(ax, 15.0, rn_y - len(rn_steps)*2.0 + 0.45, 10.0, fi_y + 0.45)

    # ── Additional note: flip augmentation ──
    box(ax, 10.0, fi_y - 1.2, 9.0, 0.65,
        "查询增强: PIL.Image.FLIP_LEFT_RIGHT 水平翻转\n生成第二组深度特征, 提高翻转不变性",
        C["preprocess"], fontsize=7, bold=True)

    legend_bar(ax, 2.5, h - 1.5, [
        ("预处理", C["preprocess"]), ("特征提取", C["feature"]), ("FAISS索引", C["index"]),
    ])

    fig.savefig("C:/Users/sun/Desktop/测试/image_retrieval/part3_deep.png",
                dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("Part 3 saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 4 — 在线查询检索流程                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def part4_search():
    fig, ax = new_figure(22, 18)
    w, h = 22, 18

    ax.text(w/2, h - 0.6, "在线查询检索流程 — 双路 FAISS 检索 + 分数融合", fontsize=15, fontweight="bold", ha="center")

    # ── Query input ──
    qy = 16.5
    box(ax, w/2, qy, 8.0, 0.8, "查询图片输入\n上传 (request.files) / 库内图片 (db_image)", C["web"], fontsize=8, bold=True)

    # ── Split into two query processing paths ──
    arrow(ax, w/2, qy - 0.4, 7.0, 15.0)
    arrow(ax, w/2, qy - 0.4, 15.0, 15.0)

    # LEFT: Query SIFT
    dashed_box(ax, 0.3, 6.0, 10.5, 8.5, C["orange"], label="查询 SIFT + BoW 编码")
    qs_y = 14.3
    qs_steps = [
        ("读取图片 (BGR)\ncv2.imdecode", C["preprocess"]),
        ("BGR -> GRAY\ncv2.cvtColor", C["preprocess"]),
        ("SIFT 关键点 + 描述符\nsift.detectAndCompute", C["feature"]),
        ("视觉词分配\nkmeans.predict", C["clustering"]),
        ("词频直方图\nnp.bincount", C["clustering"]),
        ("TF归一化 + IDF加权\nhist/sum × idf", C["clustering"]),
        ("L2 归一化\nnp.linalg.norm", C["clustering"]),
    ]
    qs_last = qs_y
    for i, (text, color) in enumerate(qs_steps):
        y = qs_y - i * 1.0
        qs_last = y
        box(ax, 5.5, y, 5.0, 0.7, text, color, fontsize=7)
        if i < len(qs_steps) - 1:
            arrow(ax, 5.5, y - 0.35, 5.5, y - 0.65)

    arrow(ax, 7.0, 15.0, 5.5, qs_y + 0.35)

    # RIGHT: Query Deep
    dashed_box(ax, 11.3, 7.0, 10.5, 10.0, C["red"], label="查询 ResNet50 深度特征 + 水平翻转增强")

    qd_y = 16.0
    qd_steps_main = [
        ("PIL.Image.open(RGB)\nT.Resize(256) -> CenterCrop(224)", C["preprocess"]),
        ("T.ToTensor()\nT.Normalize(ImageNet stats)", C["preprocess"]),
        ("ResNet50 (no FC)\nmodel.eval() + torch.no_grad()", C["feature"]),
        ("L2 归一化\n2048-d 向量 (原始)", C["feature"]),
    ]
    qd_last_main = qd_y
    for i, (text, color) in enumerate(qd_steps_main):
        y = qd_y - i * 1.2
        qd_last_main = y
        box(ax, 16.5, y, 5.5, 0.85, text, color, fontsize=7)
        if i < len(qd_steps_main) - 1:
            arrow(ax, 16.5, y - 0.43, 16.5, y - 0.78)

    arrow(ax, 15.0, 15.0, 16.5, qd_y + 0.43)

    # Flip branch
    flip_y = qd_last_main - 0.7
    box(ax, 16.5, flip_y, 4.5, 0.55, "PIL.Image.FLIP_LEFT_RIGHT\n水平翻转", C["preprocess"], fontsize=7, bold=True)
    arrow(ax, 16.5, qd_last_main - 0.43, 16.5, flip_y - 0.28)

    flip_sub = [
        ("预处理 + ResNet50 + L2归一化\n→ 2048-d 向量 (翻转)", C["feature"]),
    ]
    flip_last = flip_y
    for i, (text, color) in enumerate(flip_sub):
        y = flip_y - 0.6 - i * 0.8
        flip_last = y
        box(ax, 16.5, y, 5.5, 0.6, text, color, fontsize=7)
        arrow(ax, 16.5, y + 0.3, 16.5, y - 0.3)

    # ── FAISS Search ──
    fs_y = 5.5
    dashed_box(ax, 0.3, 0.5, 21.5, 5.5, C["teal"], label="FAISS 双索引检索 + 分数融合")

    box(ax, 5.5, fs_y - 1.0, 6.0, 0.7,
        "faiss.IndexFlatL2.search(hist, k*20)\nBoW L2 距离搜索 → bow_d, bow_idx", C["search"], fontsize=7)
    arrow(ax, 5.5, qs_last - 0.35, 5.5, fs_y - 0.5)

    box(ax, 16.5, fs_y - 1.0, 6.0, 0.7,
        "faiss.IndexFlatIP.search(feat, k*20)\nDeep 内积搜索 (原始图)", C["search"], fontsize=7)
    arrow(ax, 16.5, qd_last_main - 0.43, 16.5, fs_y - 0.5)

    box(ax, 16.5, fs_y - 2.2, 6.0, 0.7,
        "faiss.IndexFlatIP.search(feat_flip, k*20)\nDeep 内积搜索 (翻转图)", C["search"], fontsize=7)
    arrow(ax, 16.5, flip_last - 0.3, 16.5, fs_y - 1.85)

    # Score normalization
    box(ax, 11.0, fs_y - 2.2, 5.5, 0.7,
        "np.maximum(deep_d, deep_d_flip)\n取原图/翻转 max 相似度", C["score"], fontsize=7)
    arrow(ax, 16.5, fs_y - 1.35, 11.0, fs_y - 2.55)

    box(ax, 11.0, fs_y - 3.2, 5.5, 0.7,
        "Min-Max归一化\ndeep_sim = (d-d_min)/(d_max-d_min)", C["score"], fontsize=7)
    arrow(ax, 11.0, fs_y - 2.55, 11.0, fs_y - 3.55)

    box(ax, 5.5, fs_y - 3.2, 5.5, 0.7,
        "L2距离 → 相似度\nbow_sim = 1/(1+bow_d)", C["score"], fontsize=7)
    arrow(ax, 5.5, fs_y - 1.35, 5.5, fs_y - 2.85)

    # Fusion
    box(ax, 11.0, fs_y - 4.2, 8.0, 0.7,
        "分数融合: 0.35 * bow_sim + 0.65 * deep_sim\nsorted(reverse=True) -> 选前 15 进入几何验证",
        C["score"], fontsize=7.5, bold=True)
    arrow(ax, 5.5, fs_y - 3.55, 11.0, fs_y - 4.55)
    arrow(ax, 11.0, fs_y - 3.55, 11.0, fs_y - 4.55)

    legend_bar(ax, 3, h - 1.5, [
        ("预处理", C["preprocess"]), ("特征提取", C["feature"]), ("聚类/编码", C["clustering"]),
        ("FAISS检索", C["search"]), ("分数处理", C["score"]), ("Web", C["web"]),
    ])

    fig.savefig("C:/Users/sun/Desktop/测试/image_retrieval/part4_search.png",
                dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("Part 4 saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5 — 几何验证 + 最终排序 + Web 层                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def part5_geo_web():
    fig, ax = new_figure(22, 17)
    w, h = 22, 17

    ax.text(w/2, h - 0.6, "几何空间验证 + 最终排序 + Web 服务层", fontsize=15, fontweight="bold", ha="center")

    # ── Left: Geometric Verification ──
    dashed_box(ax, 0.3, 0.5, 14.0, 15.5, C["deepred"], label="几何空间验证 (对 Top-15 候选逐一执行)")

    gv_y = 15.5
    gv_steps = [
        ("输入: 融合分数 Top-15 候选\nsorted(candidates, key=score, reverse=True)", C["score"]),
        ("读取候选图片\ncv2.imdecode (BGR 格式)", C["preprocess"]),
        ("转灰度 (x2)\ncv2.cvtColor(BGR->GRAY)\n查询图 & 候选图", C["preprocess"]),
        ("SIFT 特征提取 (x2)\nsift.detectAndCompute()\n查询关键点+描述符 / 候选关键点+描述符", C["feature"]),
        ("FLANN 匹配器初始化\ncv2.FlannBasedMatcher(\n  dict(algorithm=1, trees=5),\n  dict(checks=50))\nKD-Tree 索引, 50次检查", C["geometry"]),
        ("KNN 匹配\nflann.knnMatch(q_des, c_des, k=2)\n每个查询关键点找最近2个匹配", C["geometry"]),
        ("Lowe's Ratio Test\nfor m, n in matches:\n  if m.distance < 0.75 * n.distance:\n    good.append(m)\n筛选可靠的匹配对", C["geometry"]),
        ("判断: good matches >= 10?\n否 -> geo_score = 0.0 (跳过)", C["geometry"]),
        ("提取匹配点坐标\nsrc_pts = [q_kp[m.queryIdx].pt]\ndst_pts = [c_kp[m.trainIdx].pt]", C["geometry"]),
        ("计算单应矩阵\ncv2.findHomography(\n  src_pts, dst_pts,\n  cv2.RANSAC, 5.0)\nRANSAC 鲁棒估计, 阈值 5px", C["geometry"]),
        ("判断: M != None && inliers >= 8?\n否 -> geo_score = 0.0 (跳过)", C["geometry"]),
        ("计算几何评分\ninlier_ratio = inliers / len(good)\nmatch_ratio = len(good) / min(q_kp, c_kp)\ngeo_score = 0.7*inlier_ratio + 0.3*match_ratio", C["geometry"]),
    ]
    gv_last = gv_y
    for i, (text, color) in enumerate(gv_steps):
        y = gv_y - i * 1.0
        gv_last = y
        box(ax, 7.3, y, 8.0, 0.72, text, color, fontsize=6)
        if i < len(gv_steps) - 1:
            arrow(ax, 7.3, y - 0.36, 7.3, y - 0.64)

    # ── Right: Final scoring ──
    dashed_box(ax, 14.6, 6.5, 7.1, 7.5, C["gold"], label="最终评分排序")

    fin_y = 13.0
    fin_steps = [
        ("已验证候选 (前15):\nfinal = 0.6*fusion + 0.4*geo", C["score"]),
        ("未验证候选 (其余):\nfinal = fusion * 0.75\n(未验证惩罚因子)", C["score"]),
        ("合并排序\nsorted(all, key=score, reverse=True)\n取 Top-K (K=5)", C["score"]),
    ]
    for i, (text, color) in enumerate(fin_steps):
        y = fin_y - i * 1.5
        box(ax, 18.2, y, 5.5, 1.0, text, color, fontsize=7)
        if i < len(fin_steps) - 1:
            arrow(ax, 18.2, y - 0.5, 18.2, y - 1.0)

    arrow(ax, 11.3, gv_last - 0.3, 18.2, fin_y + 0.5)

    # ── Web Layer (bottom) ──
    dashed_box(ax, 14.6, 0.5, 7.1, 5.5, C["web"], label="Web 服务层")

    web_y = 5.0
    web_steps = [
        "Flask(__name__) app.run(0.0.0.0:5000, debug=True)",
        "GET / -> render_template('index.html')",
        "POST /search -> 检索 API (max 32MB)",
        "POST /build -> 触发索引构建",
        "GET /status -> {indexed, ready, loaded}",
        "GET /database-images -> 库内图片列表",
        "GET /images/*/* -> send_from_directory",
        "uuid.uuid4().hex 上传文件唯一命名",
        "os.makedirs(exist_ok=True) 自动建目录",
    ]
    for i, text in enumerate(web_steps):
        y = web_y - i * 0.5
        box(ax, 18.2, y, 6.2, 0.4, text, C["web"], fontsize=5.8)

    arrow(ax, 18.2, fin_y - len(fin_steps)*1.5 + 0.5, 18.2, web_y + 0.3)

    # ── Summary boxes on the left bottom ──
    dashed_box(ax, 0.3, 0.5, 14.0, 2.5, C["purple"], alpha=0.05, label="")
    box(ax, 7.3, 2.5, 10.0, 0.5, "关键技术参数总览", C["bg"], fontsize=8, bold=True, alpha=0.9)

    params = [
        ("视觉词表大小: 1000  |  Bow权重: 0.35  |  Deep权重: 0.65",
         "SIFT: 128-d  |  ResNet50: 2048-d  |  BoW: 1000-d",
         "Lowe's ratio: 0.75  |  RANSAC阈值: 5.0px  |  几何权重: 0.4",
         "FAISS: IndexFlatL2 + IndexFlatIP  |  候选数: k*20  |  重排: Top-15"),
    ]
    for i, text in enumerate(params):
        box(ax, 7.3, 2.0 - i * 0.45, 11.0, 0.35, text, C["bg"], fontsize=5.8, alpha=0.8)

    legend_bar(ax, 1.5, h - 1.5, [
        ("预处理", C["preprocess"]), ("特征提取", C["feature"]),
        ("几何匹配", C["geometry"]), ("分数融合", C["score"]), ("Web/API", C["web"]),
    ])

    fig.savefig("C:/Users/sun/Desktop/测试/image_retrieval/part5_geo_web.png",
                dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("Part 5 saved.")


if __name__ == "__main__":
    part1_overview()
    part2_sift_bow()
    part3_deep()
    part4_search()
    part5_geo_web()
    print("\nAll 5 parts generated in image_retrieval/")
