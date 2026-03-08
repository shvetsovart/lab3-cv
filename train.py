#!/usr/bin/env python3
"""
Train YOLOv11n-seg on Russian Road Signs dataset.
Run: python3 train.py
"""
import os, json, shutil, glob
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'data', 'sign_dataset')
YOLO_DIR = os.path.join(BASE_DIR, 'yolo_dataset')
BEST_MODEL_SAVE = os.path.join(BASE_DIR, 'best.pt')

SIGN_CLASSES = {
    1: 'prohibitory',
    2: 'mandatory',
    3: 'warning',
    4: 'priority',
    5: 'informational',
    6: 'special',
    7: 'service',
    8: 'additional',
}
CLASS_NAMES = list(SIGN_CLASSES.values())
CLASS_ID_TO_IDX = {cid: i for i, cid in enumerate(SIGN_CLASSES.keys())}


def mask_roi_to_polygon(small_mask, roi, img_h, img_w):
    y1, x1, y2, x2 = roi
    roi_h = max(y2 - y1, 1)
    roi_w = max(x2 - x1, 1)
    mask_uint8 = (np.array(small_mask) > 0).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_uint8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(float)
    pts[:, 0] += x1
    pts[:, 1] += y1
    normalized = []
    for px, py in pts:
        normalized.extend([
            max(0.0, min(1.0, round(px / img_w, 6))),
            max(0.0, min(1.0, round(py / img_h, 6))),
        ])
    return normalized


def convert_split(split_path, out_dir):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    all_files = sorted(os.listdir(split_path))
    img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"  Images: {len(img_files)}")
    count, skipped = 0, 0
    for img_name in img_files:
        json_path = os.path.join(split_path, img_name + '_coco.json')
        if not os.path.exists(json_path):
            skipped += 1
            continue
        with open(json_path, 'r') as f:
            ann = json.load(f)
        masks_3d = np.array(ann.get('masks', []))
        class_ids = ann.get('class_ids', [])
        rois = ann.get('rois', [])
        if len(class_ids) == 0:
            skipped += 1
            continue
        img_data = cv2.imread(os.path.join(split_path, img_name))
        if img_data is None:
            skipped += 1
            continue
        img_h, img_w = img_data.shape[:2]
        label_lines = []
        for i in range(len(class_ids)):
            cls_idx = CLASS_ID_TO_IDX.get(class_ids[i], max(0, min(class_ids[i] - 1, len(CLASS_NAMES) - 1)))
            if masks_3d.ndim == 3:
                poly = mask_roi_to_polygon(masks_3d[:, :, i], rois[i], img_h, img_w)
                if poly and len(poly) >= 6:
                    coords = ' '.join(f'{c:.6f}' for c in poly)
                    label_lines.append(f'{cls_idx} {coords}')
        if label_lines:
            shutil.copy2(os.path.join(split_path, img_name), os.path.join(out_dir, 'images', img_name))
            with open(os.path.join(out_dir, 'labels', Path(img_name).stem + '.txt'), 'w') as f:
                f.write('\n'.join(label_lines))
            count += 1
    print(f"  Converted: {count}, skipped: {skipped}")
    return count


def main():
    # Step 1: Convert dataset
    print("=" * 50)
    print("Step 1: Converting dataset to YOLO-seg format")
    print("=" * 50)

    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)

    for split in ['train', 'val']:
        print(f"\n--- {split} ---")
        convert_split(os.path.join(DATASET_ROOT, split), os.path.join(YOLO_DIR, split))

    yaml_content = f"""path: {YOLO_DIR}
train: train/images
val: val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(os.path.join(YOLO_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"\ndata.yaml created")

    # Step 2: Train
    print("\n" + "=" * 50)
    print("Step 2: Training YOLOv11n-seg")
    print("=" * 50)

    from ultralytics import YOLO
    model = YOLO('yolo11n-seg.pt')
    model.train(
        data=os.path.join(YOLO_DIR, 'data.yaml'),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        name='road_signs_seg',
        save=True,
        plots=True,
        workers=4,
    )

    # Step 3: Save best model to project root
    trained_best = os.path.join('runs', 'segment', 'road_signs_seg', 'weights', 'best.pt')
    if not os.path.exists(trained_best):
        candidates = glob.glob('runs/segment/road_signs_seg*/weights/best.pt')
        if candidates:
            trained_best = candidates[0]

    if os.path.exists(trained_best):
        shutil.copy2(trained_best, BEST_MODEL_SAVE)
        print(f"\n{'=' * 50}")
        print(f"best.pt saved to: {BEST_MODEL_SAVE}")
        print(f"Size: {os.path.getsize(BEST_MODEL_SAVE) / 1e6:.1f} MB")
        print(f"{'=' * 50}")
    else:
        print(f"\nWARNING: best.pt not found at {trained_best}")
        print("Check runs/segment/ for trained weights.")


if __name__ == '__main__':
    main()
