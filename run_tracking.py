#!/usr/bin/env python3
"""Run tracking only (ByteTrack + BoT-SORT) on videos."""
import os, gc, glob
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
TRACKING_DIR = os.path.join(BASE_DIR, 'tracking_results')

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

os.makedirs(TRACKING_DIR, exist_ok=True)

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
])

print(f"Videos: {video_files}")
all_results = {}

for vf in video_files:
    vp = os.path.join(VIDEO_DIR, vf)
    print(f"\n{'='*50}")
    print(f"Video: {vf}")

    for tracker_name, tracker_yaml in [('ByteTrack', 'bytetrack.yaml'), ('BoT-SORT', 'botsort.yaml')]:
        print(f"\n--- {tracker_name} ---")
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
        print(f"  Unique tracks: {unique_ids}")
        print(f"  ID Switches:   {switches}")
        print(f"  Frames:        {len(frame_tracks)}")

        all_results.setdefault(vf, {})[tracker_name] = {
            'id_switches': switches, 'unique_ids': unique_ids, 'frames': len(frame_tracks)
        }
        del trk_model
        gc.collect()

print(f"\n{'='*70}")
print(f"  TRACKING SUMMARY")
print(f"{'='*70}")
print(f"  {'Video':<22} {'Tracker':<12} {'ID Switches':>12} {'Tracks':>8} {'Frames':>8}")
print(f"  {'-'*62}")
for vf, res in all_results.items():
    name = Path(vf).stem[:20]
    for tname, data in res.items():
        print(f"  {name:<22} {tname:<12} {data['id_switches']:>12} {data['unique_ids']:>8} {data['frames']:>8}")
print(f"{'='*70}")

total_bt = sum(r.get('ByteTrack', {}).get('id_switches', 0) for r in all_results.values())
total_bs = sum(r.get('BoT-SORT', {}).get('id_switches', 0) for r in all_results.values())
print(f"\nTotal ID Switches: ByteTrack={total_bt}, BoT-SORT={total_bs}")
print("\nDONE!")
