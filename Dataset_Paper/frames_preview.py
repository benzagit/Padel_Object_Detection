# 🖼️ SCRIPT 2: Show 4 random clips, 6 annotated frames each
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from glob import glob
import numpy as np

# 🔍 Paths
base_dir = "/content/dataset"
video_base = os.path.join(base_dir, "new_dataset")
annot_base = os.path.join(base_dir, "new_dataset_ann")

# 🎯 Choose 4 random groups + clips
selected_clips = []

for groupe in ["groupe_1", "groupe_2", "groupe_3"]:
    video_folder = os.path.join(video_base, groupe)
    if not os.path.exists(video_folder):
        continue
    video_files = glob(os.path.join(video_folder, "*.mp4"))
    selected_clips.extend([
        (groupe, os.path.splitext(os.path.basename(v))[0]) for v in video_files
    ])

if len(selected_clips) == 0:
    print("❌ No clips found!")
else:
    # Pick 4 random clips
    chosen_clips = random.sample(selected_clips, min(4, len(selected_clips)))

    # Plot: 4 rows (clips), 6 columns (frames)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle("Annotated Frames from 4 Random Clips", fontsize=16, y=0.95)

    for row, (groupe, clip_name) in enumerate(chosen_clips):
        print(f"🎲 Processing {clip_name} ({groupe})")

        video_path = os.path.join(video_base, groupe, f"{clip_name}.mp4")
        annot_path = os.path.join(annot_base, groupe, f"{clip_name}.xml")

        # Parse annotations
        if not os.path.exists(annot_path):
            print(f"❌ Missing annotation: {annot_path}")
            continue

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

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = sorted(random.sample(range(total_frames), min(6, total_frames)))

        for col, frame_idx in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            objects = annotations.get(frame_idx, [])

            ax = axes[row, col]
            ax.imshow(frame_rgb)
            for label, xmin, ymin, xmax, ymax in objects:
                width = xmax - xmin
                height = ymax - ymin
                color = "red" if label.startswith("Player") else "yellow" if label == "Ball" else "blue"
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                ax.text(xmin, ymin - 5, label[:10], color=color, fontsize=8, weight="bold")
            ax.set_title(f"Frame {frame_idx}", fontsize=9)
            ax.axis("off")

        cap.release()

    # Turn off empty subplots if <4 clips
    for row in range(len(chosen_clips), 4):
        for col in range(6):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()