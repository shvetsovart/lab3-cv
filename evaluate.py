#!/usr/bin/env python3
"""
Evaluate trained model: metrics on val set, video inference, tracking.
Run: python3 evaluate.py
"""
import os, glob, json, gc
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(BASE_DIR, 'yolo_dataset')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'video_results')
TRACKING_DIR = os.path.join(BASE_DIR, 'tracking_results')

SIGN_CLASSES = {
    1: 'prohibitory', 2: 'mandatory', 3: 'warning', 4: 'priority',
    5: 'informational', 6: 'special', 7: 'service', 8: 'additional',
}
CLASS_NAMES = list(SIGN_CLASSES.values())

# ============================================================
# Pixel-level metrics
# ============================================================

def get_boundary(mask, thickness=1):
    kernel = np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return (mask.astype(np.uint8) - eroded).astype(bool)

def compute_pixel_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0
    tp = intersection
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    pred_boundary = get_boundary(pred)
    gt_boundary = get_boundary(gt)
    if pred_boundary.sum() > 0 and gt_boundary.sum() > 0:
        dt_gt = distance_transform_edt(~gt_boundary)
        dt_pred = distance_transform_edt(~pred_boundary)
        l2 = (dt_gt[pred_boundary].mean() + dt_pred[gt_boundary].mean()) / 2.0
    else:
        l2 = float('inf')
    return {'iou': iou, 'precision': precision, 'recall': recall, 'l2': l2}

def polygon_to_mask(polygon, img_w, img_h):
    pts = np.array(polygon).reshape(-1, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    pts = pts.astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def parse_yolo_label(label_path):
    entries = []
    if not os.path.exists(label_path):
        return entries
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                entries.append((int(parts[0]), [float(x) for x in parts[1:]]))
    return entries

# ============================================================
# Tracking helpers
# ============================================================

def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def count_id_switches(frame_tracks, iou_thresh=0.3):
    switches = 0
    prev_tracks = {}
    for curr_tracks in frame_tracks:
        if not prev_tracks or not curr_tracks:
            prev_tracks = curr_tracks
            continue
        matched_prev, matched_curr = set(), set()
        pairs = []
        for pid in prev_tracks:
            for cid in curr_tracks:
                iou = compute_box_iou(prev_tracks[pid], curr_tracks[cid])
                if iou > iou_thresh:
                    pairs.append((iou, pid, cid))
        pairs.sort(reverse=True)
        for iou, pid, cid in pairs:
            if pid in matched_prev or cid in matched_curr:
                continue
            matched_prev.add(pid)
            matched_curr.add(cid)
            if pid != cid:
                switches += 1
        prev_tracks = curr_tracks
    return switches

# ============================================================
# Main
# ============================================================

def main():
    from ultralytics import YOLO
    import pandas as pd

    assert os.path.exists(BEST_MODEL_PATH), f"best.pt not found: {BEST_MODEL_PATH}"
    print(f"Model: {BEST_MODEL_PATH} ({os.path.getsize(BEST_MODEL_PATH)/1e6:.1f} MB)")

    model = YOLO(BEST_MODEL_PATH)

    # ---- STEP 1: YOLO validation ----
    print("\n" + "=" * 60)
    print("  STEP 1: YOLO Validation Metrics")
    print("=" * 60)

    data_yaml = os.path.join(YOLO_DIR, 'data.yaml')
    if os.path.exists(data_yaml):
        metrics = model.val(data=data_yaml, split='val')
        print(f"\n  Mask Precision: {metrics.seg.mp:.4f}")
        print(f"  Mask Recall:    {metrics.seg.mr:.4f}")
        print(f"  Mask mAP@50:    {metrics.seg.map50:.4f}")
        print(f"  Mask mAP@50-95: {metrics.seg.map:.4f}")
    else:
        print(f"  data.yaml not found, skipping YOLO val")

    # ---- STEP 2: Per-pixel metrics ----
    print("\n" + "=" * 60)
    print("  STEP 2: Per-Pixel IoU / Precision / Recall / L2")
    print("=" * 60)

    val_img_dir = os.path.join(YOLO_DIR, 'val', 'images')
    val_lbl_dir = os.path.join(YOLO_DIR, 'val', 'labels')

    if os.path.exists(val_img_dir):
        val_images = sorted(glob.glob(os.path.join(val_img_dir, '*.*')))
        print(f"  Evaluating {len(val_images)} images...")
        all_metrics = []
        for idx, img_path in enumerate(val_images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem
            gt_entries = parse_yolo_label(os.path.join(val_lbl_dir, stem + '.txt'))
            if not gt_entries:
                continue
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            for _, coords in gt_entries:
                gt_mask = np.maximum(gt_mask, polygon_to_mask(coords, w, h))
            results = model.predict(img_path, verbose=False)
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            if results and results[0].masks is not None:
                for seg_mask in results[0].masks.data:
                    m = seg_mask.cpu().numpy()
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_mask = np.maximum(pred_mask, (m_resized > 0.5).astype(np.uint8) * 255)
            m = compute_pixel_metrics(pred_mask, gt_mask)
            all_metrics.append(m)
            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(val_images)} done...")

        if all_metrics:
            df = pd.DataFrame(all_metrics)
            mean_iou = df['iou'].mean()
            mean_prec = df['precision'].mean()
            mean_rec = df['recall'].mean()
            finite_l2 = df['l2'][df['l2'] != float('inf')]
            mean_l2 = finite_l2.mean() if len(finite_l2) > 0 else float('inf')
            pct_05 = (df['iou'] >= 0.5).mean() * 100
            pct_075 = (df['iou'] >= 0.75).mean() * 100
            pct_09 = (df['iou'] >= 0.9).mean() * 100
            print(f"\n  {'='*55}")
            print(f"  {'Метрика':<25} {'Значение':>10}")
            print(f"  {'-'*55}")
            print(f"  {'Mean IoU':<25} {mean_iou:>10.4f}")
            print(f"  {'Mean Precision':<25} {mean_prec:>10.4f}")
            print(f"  {'Mean Recall':<25} {mean_rec:>10.4f}")
            print(f"  {'Mean L2 (boundary)':<25} {mean_l2:>10.4f}")
            print(f"  {'-'*55}")
            print(f"  {'% IoU >= 0.50':<25} {pct_05:>9.1f}%")
            print(f"  {'% IoU >= 0.75':<25} {pct_075:>9.1f}%")
            print(f"  {'% IoU >= 0.90':<25} {pct_09:>9.1f}%")
            print(f"  {'='*55}")
    else:
        print("  Val images not found, skipping per-pixel eval")

    del model
    gc.collect()

    # ---- STEP 3: Video inference ----
    print("\n" + "=" * 60)
    print("  STEP 3: Video Inference")
    print("=" * 60)

    video_files = sorted([
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]) if os.path.exists(VIDEO_DIR) else []

    if not video_files:
        print(f"  No videos in {VIDEO_DIR}")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        seg_model = YOLO(BEST_MODEL_PATH)
        for vf in video_files:
            vp = os.path.join(VIDEO_DIR, vf)
            cap = cv2.VideoCapture(vp)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = frames / fps if fps > 0 else 0
            cap.release()
            print(f"\n  {vf}: {dur:.0f}s, {frames} frames, {fps:.0f} fps")
            results = seg_model.predict(
                source=vp, save=True, stream=True, conf=0.25,
                project=OUTPUT_DIR, name=Path(vf).stem,
                show_labels=True, show_conf=True, vid_stride=2,
            )
            count = sum(1 for _ in results)
            print(f"    Processed: {count} frames")
            gc.collect()

        del seg_model
        gc.collect()

        out_vids = glob.glob(os.path.join(OUTPUT_DIR, '**', '*.avi'), recursive=True)
        out_vids += glob.glob(os.path.join(OUTPUT_DIR, '**', '*.mp4'), recursive=True)
        print(f"\n  Saved videos:")
        for ov in out_vids:
            print(f"    {ov}")

    # ---- STEP 4: Tracking ----
    print("\n" + "=" * 60)
    print("  STEP 4: Tracking (ByteTrack + BoT-SORT)")
    print("=" * 60)

    if not video_files:
        print("  No videos, skipping tracking")
        return

    os.makedirs(TRACKING_DIR, exist_ok=True)
    all_tracking_results = {}

    for vf in video_files:
        vp = os.path.join(VIDEO_DIR, vf)
        print(f"\n  {'='*50}")
        print(f"  Video: {vf}")

        for tracker_name, tracker_yaml in [('ByteTrack', 'bytetrack.yaml'), ('BoT-SORT', 'botsort.yaml')]:
            print(f"\n  --- {tracker_name} ---")
            trk_model = YOLO(BEST_MODEL_PATH)
            results = trk_model.track(
                source=vp, tracker=tracker_yaml, save=True, stream=True,
                conf=0.25, project=TRACKING_DIR,
                name=f'{Path(vf).stem}_{tracker_name.lower().replace("-","")}',
                vid_stride=2,
            )
            frame_tracks = []
            for r in results:
                tracks = {}
                if r.boxes is not None and r.boxes.id is not None:
                    for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
                        tracks[int(tid)] = box.tolist()
                frame_tracks.append(tracks)

            switches = count_id_switches(frame_tracks)
            unique_ids = len(set(tid for ft in frame_tracks for tid in ft.keys()))
            print(f"    Unique tracks: {unique_ids}")
            print(f"    ID Switches:   {switches}")
            print(f"    Frames:        {len(frame_tracks)}")

            all_tracking_results.setdefault(vf, {})[tracker_name] = {
                'id_switches': switches, 'unique_ids': unique_ids, 'frames': len(frame_tracks)
            }
            del trk_model
            gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRACKING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Video':<22} {'Tracker':<12} {'ID Switches':>12} {'Tracks':>8} {'Frames':>8}")
    print(f"  {'-'*62}")
    for vf, res in all_tracking_results.items():
        name = Path(vf).stem[:20]
        for tname, data in res.items():
            print(f"  {name:<22} {tname:<12} {data['id_switches']:>12} {data['unique_ids']:>8} {data['frames']:>8}")
    print(f"{'='*70}")

    total_bt = sum(r.get('ByteTrack', {}).get('id_switches', 0) for r in all_tracking_results.values())
    total_bs = sum(r.get('BoT-SORT', {}).get('id_switches', 0) for r in all_tracking_results.values())
    print(f"\n  Total ID Switches: ByteTrack={total_bt}, BoT-SORT={total_bs}")

    print("\n  DONE!")


if __name__ == '__main__':
    main()
