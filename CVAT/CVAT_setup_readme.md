1. CVAT_SETUP/README.md

# CVAT Setup for Padel Sports Annotation

> How I configured CVAT during my internship at Blockward to build a custom dataset for multi-player tracking and ball detection in Padel matches.

---

## 🧰 Purpose

To accelerate manual labeling using semi-automated tools while maintaining high-quality annotations for future model training.

The challenge was threefold:
- Track **4 unique players** (2v2) with persistent IDs
- Detect a **small, fast-moving ball**
- Label **12 static court keypoints**

Standard object detectors failed on ID persistence and ball detection — so we used CVAT as the foundation of our pipeline.

---

## 🛠️ Setup Process

### 1. Local Installation via Docker

Used official CVAT with serverless module:

```bash
git clone https://github.com/openvinotoolkit/cvat.git
cd cvat
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Accessed at http://localhost:8080.

Allocated 8GB RAM in Docker settings for stable video handling.


2. Serverless YOLOv7 Integration
Ran pre-trained YOLOv7 for auto-labeling:

```bash
./serverless/deploy_cpu.sh serverless/onnx/WongKinYiu/yolov7
```
Enabled under “TF Detection API” in CVAT UI.

Result:

Detected "person" → mapped to player
But assigned same player_id=1 to all players → required manual correction
Ball detection was poor due to size and motion blur.

3. Label Schema Design
Defined labels carefully to support downstream tracking:
    player → Attribute: player_id (number: 1, 2, 3, 4) ← critical for identity tracking
    ball → no attributes
    court_keypoint → Attribute: point_id (1 to 12)
Without proper label design, even SOTA models fail.

4. Semi-Automation Workflow
Annotated keyframes every 10 frames
Used Interpolation (Propagation) to fill intermediate boxes
Saved ~40% time vs full manual labeling
Reviewed output frame-by-frame for quality control

📝 Key Insight
Automation doesn’t eliminate work — it shifts it.

Instead of drawing boxes, I spent time designing workflows, correcting IDs, and validating outputs. That’s where real engineering value lies.

This setup allowed us to annotate over 4500 frames — forming the basis of a novel Padel dataset.