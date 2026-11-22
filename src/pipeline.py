import argparse
from pathlib import Path
from src.data.mask_plates import VehiclePreprocessor
from src.models.embedder import DINOv2Embedder
from src.retrieval.matcher import LocalDetailVerifier, VehicleIndex, VehicleRecord


def run_demo(query_img: str, gallery_dir: str, weights_path: str, top_k: int = 3):
    preprocessor = VehiclePreprocessor()
    embedder = DINOv2Embedder(weights_path=weights_path)
    index = VehicleIndex(dim=768)
    verifier = LocalDetailVerifier(inlier_threshold=15)

    gallery_paths = sorted(Path(gallery_dir).glob("*.jpg"))
    print(f"Индексация галереи ({len(gallery_paths)} кадров)...")
    for p in gallery_paths:
        vid = p.stem.split("_")[0]
        img = preprocessor.process(p, crop_vehicle=False)
        emb = embedder.encode_image(img)
        index.add(emb, VehicleRecord(vehicle_id=vid, plate=f"PLATE-{vid}", image_path=str(p)))

    print(f"\nПоиск нарушителя без номера: {query_img}")
    q_img = preprocessor.process(query_img, crop_vehicle=False)
    q_emb = embedder.encode_image(q_img)
    candidates = index.search(q_emb, top_k=top_k)

    print("\n--- Топ кандидатов для проверки оператором (Human-in-the-loop) ---")
    for rank, item in enumerate(candidates, start=1):
        rec = item["record"]
        ver = verifier.verify(query_img, rec.image_path)
        status = "HIGH CONFIDENCE" if ver["verified_for_operator"] else "MANUAL CHECK"
        print(
            f"Rank-{rank} | Plate: {rec.plate} | Cosine: {item['cosine_sim']:.3f} | "
            f"LightGlue Inliers: {ver['ransac_inliers']:3d} | Status: {status} | File: {rec.image_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plate-Free Vehicle Re-ID Inference Pipeline")
    parser.add_argument("--query", required=True, help="Путь к фото нарушителя без номера")
    parser.add_argument("--gallery", required=True, help="Папка с базой фото with_plate")
    parser.add_argument("--weights", default="weights/dinov2_lora_fp16_merged.pth", help="Путь к весам")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run_demo(args.query, args.gallery, args.weights, args.top_k)
