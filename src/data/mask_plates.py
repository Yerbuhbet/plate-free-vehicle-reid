from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Классы COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}


class VehiclePreprocessor:
    def __init__(
        self,
        vehicle_model_path: str = "yolo11n.pt",
        plate_model_path: Optional[str] = None,
        conf: float = 0.35,
    ):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path) if plate_model_path else None
        self.conf = conf

    @staticmethod
    def mask_box(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(box, fill=(0, 0, 0))
        return img_copy

    def process(self, image_path: str | Path, crop_vehicle: bool = True) -> Image.Image:
        img = Image.open(image_path).convert("RGB")

        # 1. Маскируем госномер, если подключен кастомный детектор номеров
        if self.plate_model is not None:
            plate_res = self.plate_model.predict(img, conf=self.conf, verbose=False)[0]
            for box in plate_res.boxes.xyxy.cpu().numpy():
                img = self.mask_box(img, tuple(box.tolist()))

        # 2. Вырезаем самый крупный автомобиль / грузовик в кадре
        if crop_vehicle:
            res = self.vehicle_model.predict(img, conf=self.conf, verbose=False)[0]
            best_box, best_area = None, 0.0
            for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if int(cls_id) in VEHICLE_CLASSES:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > best_area:
                        best_area = area
                        best_box = box
            if best_box is not None:
                img = img.crop(tuple(best_box.tolist()))

        return img
