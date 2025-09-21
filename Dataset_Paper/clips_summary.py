# 📊 SCRIPT 1: Analyze all clips — get real FPS, duration, detection rates
import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from glob import glob

def get_video_info(video_path):
    """Extract fps, duration, frame count, resolution from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0
    return {
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "frames": frame_count,
        "resolution": f"{width}x{height}"
    }

def parse_cvat_xml(annot_path):
    """Parse CVAT XML and return dict: {frame_num: [objects]}"""
    if not os.path.exists(annot_path):
        print(f"❌ Annotation not found: {annot_path}")
        return None
    tree = ET.parse(annot_path)
    root = tree.getroot()
    annotations = {}
    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            frame_num = int(box.get("frame"))
            obj = (
                label,
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr"))
            )
            if frame_num not in annotations:
                annotations[frame_num] = []
            annotations[frame_num].append(obj)
    return annotations

def analyze_clip(video_path, annot_path):
    """Analyze one clip: video + annotations"""
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🔍 Processing {clip_name}...")

    # Get video metadata
    video_info = get_video_info(video_path)
    if not video_info:
        return None

    # Parse annotations
    annotations = parse_cvat_xml(annot_path)
    if not annotations:
        return None

    total_frames = video_info["frames"]
    player_frames = 0
    ball_frames = 0

    for frame_idx in range(total_frames):
        objects = annotations.get(frame_idx, [])
        players = sum(1 for o in objects if o[0].startswith("Player"))
        ball = sum(1 for o in objects if o[0] == "Ball")

        player_rate = players / 4.0
        ball_rate = 1.0 if ball > 0 else 0.0

        player_frames += player_rate
        ball_frames += ball_rate

    avg_player_pct = (player_frames / total_frames) * 100
    avg_ball_pct = (ball_frames / total_frames) * 100

    return {
        "clip_name": clip_name,
        "fps": video_info["fps"],
        "duration": video_info["duration"],
        "resolution": video_info["resolution"],
        "frames": total_frames,
        "player_detection_rate": round(avg_player_pct, 2),
        "ball_detection_rate": round(avg_ball_pct, 2)
    }

# 🔍 Main loop over all groups
base_dir = "/content/dataset"
video_base = os.path.join(base_dir, "new_dataset")
annot_base = os.path.join(base_dir, "new_dataset_ann")

results = []

for groupe in ["groupe_1", "groupe_2", "groupe_3"]:
    video_folder = os.path.join(video_base, groupe)
    annot_folder = os.path.join(annot_base, groupe)

    if not os.path.exists(video_folder) or not os.path.exists(annot_folder):
        print(f"⚠️ Missing folder: {video_folder} or {annot_folder}")
        continue

    for video_file in glob(os.path.join(video_folder, "*.mp4")):
        clip_name = os.path.splitext(os.path.basename(video_file))[0]
        annot_file = os.path.join(annot_folder, f"{clip_name}.xml")

        result = analyze_clip(video_file, annot_file)
        if result is not None:  # ✅ Fixed: complete condition
            results.append(result)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("dataset_summary.csv", index=False)
print(f"✅ Analysis complete! Saved {len(df)} clips to dataset_summary.csv")
display(df.head())