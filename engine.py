import os
import pickle
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")

# ── Deep learning (ResNet50) ──────────────────────────────────────────
import torch
import torchvision.transforms as T
from torchvision import models

# ── Clustering & FAISS ────────────────────────────────────────────────
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import faiss
from PIL import Image


def _imread(path: str) -> np.ndarray | None:
    """cv2.imread replacement that handles non-ASCII paths on Windows."""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class ImageRetrievalEngine:
    """Large-scale image retrieval: SIFT + BoW + ResNet50 + FAISS + geometric verification.

    Two complementary FAISS indexes are built:
      - BoW index  (TF-IDF weighted, L2)      – local texture / structure
      - Deep index (ResNet50, inner-product)  – semantic similarity

    At query time the two score lists are fused, and the top candidates are
    re-ranked with a SIFT keypoint-matching + homography check to handle
    cropping, flipping, and rotation.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        device: str | None = None,
        bow_weight: float = 0.35,
        deep_weight: float = 0.65,
    ):
        self.vocab_size = vocab_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bow_weight = bow_weight
        self.deep_weight = deep_weight

        # SIFT
        self.sift = cv2.SIFT_create()

        # BoW
        self.kmeans: MiniBatchKMeans | None = None
        self.idf: np.ndarray | None = None

        # Deep model
        self.deep_model: torch.nn.Module | None = None
        self.transform: T.Compose | None = None

        # FAISS indexes
        self.bow_index: faiss.Index | None = None
        self.deep_index: faiss.Index | None = None

        # FLANN matcher (for geometric verification)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )

        # Metadata
        self.image_paths: list[str] = []

        self._init_deep_model()

    # ── Deep model ────────────────────────────────────────────────────

    def _init_deep_model(self):
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.deep_model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.deep_model.to(self.device)
        self.deep_model.eval()

        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_deep_features(self, image) -> np.ndarray:
        """Return L2-normalised 2048-d feature vector."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.deep_model(t).squeeze().cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-8)

    # ── SIFT + BoW ────────────────────────────────────────────────────

    def extract_sift_descriptors(self, image) -> np.ndarray | None:
        if isinstance(image, str):
            image = _imread(image)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, des = self.sift.detectAndCompute(gray, None)
        return des

    def build_vocabulary(self, image_paths: list[str]):
        all_descriptors = []
        for p in image_paths:
            des = self.extract_sift_descriptors(p)
            if des is not None and len(des) > 0:
                all_descriptors.append(des)

        if not all_descriptors:
            raise ValueError("No SIFT descriptors extracted – check your images.")

        all_des = np.vstack(all_descriptors).astype(np.float32)
        n_samples = min(150_000, len(all_des))
        idx = np.random.choice(len(all_des), n_samples, replace=False)
        sampled = all_des[idx]

        print(f"   Clustering {n_samples} descriptors → {self.vocab_size} words …")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size, batch_size=2000, random_state=42, n_init=3
        )
        self.kmeans.fit(sampled)
        print("   Vocabulary ready.")

    def bow_histogram(self, descriptors: np.ndarray | None) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        words = self.kmeans.predict(descriptors.astype(np.float32))
        return np.bincount(words, minlength=self.vocab_size).astype(np.float32)

    def compute_tfidf(self, histograms: np.ndarray) -> np.ndarray:
        tf = histograms / (histograms.sum(axis=1, keepdims=True) + 1e-8)
        df = (histograms > 0).sum(axis=0)
        N = len(histograms)
        self.idf = np.log((N + 1) / (df + 1)) + 1.0
        tfidf = tf * self.idf
        return normalize(tfidf, norm="l2").astype(np.float32)

    # ── Index build ───────────────────────────────────────────────────

    def build_index(self, image_dir: str):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
        self.image_paths = []
        for root, _dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(exts):
                    self.image_paths.append(os.path.join(root, f))
        self.image_paths.sort()
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        print(f"[*] {len(self.image_paths)} images found.")

        # 1. Vocabulary
        print("[1/4] Building visual vocabulary …")
        self.build_vocabulary(self.image_paths)

        # 2. BoW histograms + deep features
        print("[2/4] Computing BoW + deep features …")
        bow_hists, deep_feats = [], []
        for i, p in enumerate(self.image_paths):
            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{len(self.image_paths)}")
            des = self.extract_sift_descriptors(p)
            bow_hists.append(self.bow_histogram(des))
            try:
                deep_feats.append(self.extract_deep_features(p))
            except Exception:
                deep_feats.append(np.zeros(2048, dtype=np.float32))

        bow_hists = self.compute_tfidf(np.array(bow_hists))
        deep_feats = np.array(deep_feats, dtype=np.float32)

        # 3. FAISS indexes
        print("[3/4] Building FAISS indexes …")
        self.bow_index = faiss.IndexFlatL2(bow_hists.shape[1])
        self.bow_index.add(bow_hists)

        self.deep_index = faiss.IndexFlatIP(deep_feats.shape[1])
        self.deep_index.add(deep_feats)

        print(f"[4/4] Done – {len(self.image_paths)} images indexed.")

    # ── Geometric verification (image matching model) ─────────────────

    def _geometric_verify(self, query_img: np.ndarray, candidate_img: np.ndarray) -> float:
        """Return a score in [0, 1] based on SIFT matches + homography inliers."""
        q_kp, q_des = self.sift.detectAndCompute(
            cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY), None
        )
        c_kp, c_des = self.sift.detectAndCompute(
            cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY), None
        )

        if q_des is None or c_des is None or len(q_des) < 4 or len(c_des) < 4:
            return 0.0

        # Lowe's ratio test
        matches = self.flann.knnMatch(q_des, c_des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            return 0.0

        src_pts = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([c_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return 0.0

        inliers = int(mask.sum())
        if inliers < 8:
            return 0.0

        # Score: combination of inlier ratio and match-ratio
        inlier_ratio = inliers / len(good)
        match_ratio = len(good) / min(len(q_kp), len(c_kp))
        return float(0.7 * inlier_ratio + 0.3 * match_ratio)

    # ── Search ────────────────────────────────────────────────────────

    def search(self, query_image, k: int = 5) -> list[dict]:
        """Return top-k results.  query_image can be a file path or a numpy array (BGR)."""
        if isinstance(query_image, str):
            img_cv = _imread(query_image)
            img_pil = Image.open(query_image).convert("RGB")
        else:
            img_cv = query_image
            img_pil = Image.fromarray(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))

        if img_cv is None:
            raise ValueError("Cannot read query image.")

        # ── 1.  Query features ──
        des = self.extract_sift_descriptors(img_cv)
        hist = self.bow_histogram(des)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        if self.idf is not None:
            hist = hist * self.idf
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        hist = hist.reshape(1, -1).astype(np.float32)

        deep_feat = self.extract_deep_features(img_pil).reshape(1, -1).astype(np.float32)

        # flipped query (handle horizontal flips)
        deep_feat_flip = (
            self.extract_deep_features(img_pil.transpose(Image.FLIP_LEFT_RIGHT))
            .reshape(1, -1)
            .astype(np.float32)
        )

        # ── 2.  FAISS search ──
        n_candidates = min(k * 20, len(self.image_paths))
        bow_d, bow_idx = self.bow_index.search(hist, n_candidates)
        deep_d, deep_idx = self.deep_index.search(deep_feat, n_candidates)
        deep_d_flip, deep_idx_flip = self.deep_index.search(
            deep_feat_flip, n_candidates
        )

        # ── 3.  Score fusion ──
        bow_sim = 1.0 / (1.0 + bow_d[0])  # L2 → similarity

        deep_sim = np.maximum(deep_d[0], deep_d_flip[0])
        # min-max normalise deep scores
        d_min, d_max = deep_sim.min(), deep_sim.max()
        if d_max - d_min > 1e-8:
            deep_sim = (deep_sim - d_min) / (d_max - d_min)

        candidates: dict[int, float] = {}
        for i, s in zip(bow_idx[0], bow_sim):
            candidates[int(i)] = candidates.get(int(i), 0.0) + self.bow_weight * s
        for i, s in zip(deep_idx[0], deep_sim):
            candidates[int(i)] = candidates.get(int(i), 0.0) + self.deep_weight * s

        # ── 4.  Re-rank top with geometric verification ──
        sorted_cand = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_for_geo = sorted_cand[: min(15, len(sorted_cand))]

        verified: list[tuple[int, float]] = []
        for idx, fusion_score in top_for_geo:
            cand_img = _imread(self.image_paths[idx])
            if cand_img is None:
                continue
            geo_score = self._geometric_verify(img_cv, cand_img)
            # fuse geometric score with fusion score
            final_score = 0.6 * fusion_score + 0.4 * geo_score
            verified.append((idx, final_score))

        # Merge remaining candidates
        verified_set = {v[0] for v in verified}
        for idx, fusion_score in sorted_cand:
            if idx not in verified_set:
                verified.append((idx, fusion_score * 0.75))  # slight penalty for not verified

        verified.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in verified[:k]:
            results.append(
                {"path": self.image_paths[idx], "score": round(float(score), 4)}
            )
        return results

    # ── Persistence ───────────────────────────────────────────────────

    def save_index(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        bow_data = {
            "kmeans": self.kmeans,
            "idf": self.idf,
            "vocab_size": self.vocab_size,
            "image_paths": self.image_paths,
        }
        with open(os.path.join(index_dir, "bow_data.pkl"), "wb") as f:
            pickle.dump(bow_data, f)
        if self.bow_index is not None:
            data = faiss.serialize_index(self.bow_index)
            with open(os.path.join(index_dir, "bow.faiss"), "wb") as f:
                f.write(data)
        if self.deep_index is not None:
            data = faiss.serialize_index(self.deep_index)
            with open(os.path.join(index_dir, "deep.faiss"), "wb") as f:
                f.write(data)
        print(f"[OK] Index saved to {index_dir}")

    def load_index(self, index_dir: str) -> bool:
        bow_pkl = os.path.join(index_dir, "bow_data.pkl")
        bow_faiss = os.path.join(index_dir, "bow.faiss")
        deep_faiss = os.path.join(index_dir, "deep.faiss")
        if not all(os.path.exists(f) for f in [bow_pkl, bow_faiss, deep_faiss]):
            return False
        with open(bow_pkl, "rb") as f:
            data = pickle.load(f)
        self.kmeans = data["kmeans"]
        self.idf = data["idf"]
        self.vocab_size = data["vocab_size"]
        self.image_paths = data["image_paths"]
        with open(bow_faiss, "rb") as f:
            self.bow_index = faiss.deserialize_index(
                np.frombuffer(f.read(), dtype=np.uint8)
            )
        with open(deep_faiss, "rb") as f:
            self.deep_index = faiss.deserialize_index(
                np.frombuffer(f.read(), dtype=np.uint8)
            )
        print(f"[OK] Index loaded - {len(self.image_paths)} images.")
        return True
