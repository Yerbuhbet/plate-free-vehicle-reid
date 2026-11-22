from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import faiss
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, match_pair
from lightglue.utils import load_image


@dataclass
class VehicleRecord:
    vehicle_id: str
    plate: str
    image_path: str
    brand: Optional[str] = None
    axles: Optional[int] = None


class VehicleIndex:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.records: List[VehicleRecord] = []

    def add(self, embedding: np.ndarray, record: VehicleRecord):
        vec = embedding.astype(np.float32).reshape(1, self.dim)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.records.append(record)

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        brand: Optional[str] = None,
        axles: Optional[int] = None,
    ) -> List[Dict]:
        vec = query_emb.astype(np.float32).reshape(1, self.dim)
        faiss.normalize_L2(vec)

        # Ищем с запасом, чтобы применить фильтр метаданных (марка, оси)
        fetch_k = min(len(self.records), max(top_k * 5, 50))
        if fetch_k == 0:
            return []
        scores, indices = self.index.search(vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rec = self.records[idx]
            if brand and rec.brand and rec.brand.lower() != brand.lower():
                continue
            if axles is not None and rec.axles is not None and rec.axles != axles:
                continue
            results.append({"record": rec, "cosine_sim": float(score)})
            if len(results) >= top_k:
                break
        return results


class LocalDetailVerifier:
    """Сверка локальных деталей (царапины, стыки, наклейки) через SuperPoint + LightGlue."""
    def __init__(self, device: Optional[str] = None, inlier_threshold: int = 15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inlier_threshold = inlier_threshold
        self.extractor = SuperPoint(max_num_keypoints=512).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    @torch.inference_mode()
    def verify(self, query_path: str, candidate_path: str) -> Dict:
        img0 = load_image(query_path).to(self.device)
        img1 = load_image(candidate_path).to(self.device)
        f0, f1, m01 = match_pair(self.extractor, self.matcher, img0, img1)

        matches = m01["matches"]
        k0 = f0["keypoints"][matches[..., 0]].cpu().numpy()
        k1 = f1["keypoints"][matches[..., 1]].cpu().numpy()

        n_inliers = len(k0)
        if len(k0) >= 8:
            _, mask = cv2.findFundamentalMat(k0, k1, cv2.FM_RANSAC, 3.0)
            if mask is not None:
                n_inliers = int(mask.sum())

        return {
            "raw_matches": int(len(k0)),
            "ransac_inliers": n_inliers,
            "verified_for_operator": n_inliers >= self.inlier_threshold,
        }
