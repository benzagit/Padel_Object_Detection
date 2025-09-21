# Padel Sports Vision: Annotation Pipeline & Model Evaluation

> A technical record of my internship at Blockward — building the first structured dataset for Padel sports using CVAT, FairMOT, Trackernet, and VLMs. This repository documents the full pipeline: from data collection and annotation to model evaluation and figure generation for an upcoming dataset paper.

---

## 📘 Abstract

Padel is a fast-growing racquet sport with no public computer vision benchmarks. During my two-month internship, I led the creation of a custom dataset for multi-player tracking, ball detection, and court keypoint localization. The challenge was threefold: (1) annotate 4 players with persistent IDs in 2v2 matches, (2) detect a small, fast-moving ball under motion blur, and (3) do so efficiently across hundreds of frames. 

I designed and implemented an annotation pipeline using CVAT with serverless YOLOv7 for pre-labeling, evaluated SOTA models including FairMOT for player tracking and TrackernetV3 for ball detection, and experimented with Vision-Language Models (VLMs) like OWLv2 for prompt-based annotation. Despite limitations in off-the-shelf models, the pipeline achieved high-quality labeling suitable for future model training. I also contributed two sections and three figures to an upcoming dataset paper.

This repository reconstructs the technical workflow without proprietary data, serving as both a portfolio piece and a reference for sports vision researchers.

---

## 🔍 Problem Statement

Standard object detection models fail in Padel due to:
- **Lack of identity persistence**: YOLO detects "person" but cannot distinguish between 4 players.
- **Small, fast objects**: The ball (~4 cm) moves at >100 km/h — often blurred or occluded.
- **Structured geometry**: 12 fixed court keypoints require precise labeling.
- **Annotation cost**: Manual labeling of 4500+ frames is time-prohibitive.

Without a dedicated dataset, training robust models for real-time analysis is impossible.

---

## 🛠️ Technical Approach

The solution was not a single model, but a **pipeline combining human-in-the-loop annotation with model-assisted labeling**:

1. **Data Collection**: High-resolution YouTube match clips (60fps, 1920x1080) → split into 5-second segments.
2. **Label Design**: In CVAT, defined `player` (with `player_id=1..4`), `ball`, and `court_keypoint` (with `point_id=1..12`).
3. **Pre-annotation**: Used CVAT’s built-in serverless YOLOv7 via `deploy_cpu.sh` to auto-label frames.
4. **Manual Correction**: Assigned unique IDs, corrected missed balls, used interpolation every 30 frames.
5. **Model Evaluation**:
   - **FairMOT**: Fine-tuned for reID; reduced ID switches to <5%.
   - **TrackernetV3**: Didn't perform well with detecting padel ball on test clips.
   - **OWLv2**: Tested zero-shot prompts; failed on ID persistence and speed.
6. **Paper Contribution**: Wrote “Related Work” and “Dataset Description” sections; generated 3 figures via Python scripts.

The result: a high-quality, structured dataset ready for model training.

---

## 🧪 Key Experiments & Results

### 1. FairMOT for Player Tracking
Adapted FairMOT to track 4 players with unique IDs by formatting labels as `class_id, track_id, x, y, w, h`.

```bash
python src/train.py mot \
  --load_model ../models/fairmot_dla34.pth \
  --gpus 0 \
  --data_cfg ../src/lib/cfg/padel.json
  --load_model ../models/fairmot_dla34.pth \
  --gpus -1 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-5 \
  --num_epochs 1 \
  --num_iters 2 \
  --conf_thres 0.4 \
  --arch dla_34 \
  --input_h 608 \
  --input_w 1088 \
  --fix_res \
  --save_all
```
  Result: 92% mAP, <5% ID switches — solved player tracking.

2. TrackernetV3 for Ball Detection

Tested public implementation on Padel clips:

```bash
%cd /content/TrackNetV3

!python predict.py \
  --video_file /content/your_video.mp4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir prediction \
  --output_video \
  --batch_size 1
```

3. OWLv2 for Prompt-Based Annotation

Tested zero-shot detection with natural language:

```bash
from transformers import Owlv2Processor, Owlv2ForObjectDetection
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
```

Result: ~70% accuracy, but no persistent IDs and slow (2 FPS) — not viable for production.

📚 What Was Learned

+ CVAT’s serverless module enables rapid integration of ONNX/TensorRT models — but requires careful label mapping.
+ FairMOT’s reID head is essential for sports tracking — default settings need tuning (reid_dim, track_buffer).
+ Trackernet excels at small-object detection but lacks generalization across sports.
+ VLMs are promising but not yet production-ready for structured annotation tasks.
+ Dataset quality > model complexity — a well-labeled dataset is the foundation of any vision system.


📝 Acknowledgements

This work was conducted during my internship at Blockward. I thank the team for the opportunity and mentorship. The full dataset and internal models remain property of the company.


