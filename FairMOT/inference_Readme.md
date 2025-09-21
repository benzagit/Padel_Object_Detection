# FairMOT Inference Pipeline for Player Tracking

A clean, inference pipeline for **FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**.



> Running FairMOT inference on local Padel videos using a pre-trained baseline model (fairmot_dla34.pth) to evaluate detection and tracking performance before fine-tuning.

---

## 🧰 Purpose

To assess the out-of-the-box capabilities of FairMOT — a state-of-the-art multi-object tracking model — on real-world Padel match footage. The goal was to determine whether it could:
- Detect all players consistently
- Maintain persistent IDs across frames
- Handle fast motion and occlusion

This formed the **baseline evaluation** prior to fine-tuning on our custom annotated dataset.

### Environment
- OS: WSL2 (Ubuntu 20.04)
- Python: 3.8
- Dependencies: install from requirements.txt

🔗 Official Repo: https://github.com/ifzhang/FairMOT

### 1. Clone the Official Repository

```bash
git clone https://github.com/ifzhang/FairMOT.git
cd FairMOT
```

### 2. Create Conda Environment

```bash
conda create -n FairMOT
conda activate FairMOT
```

### 3. Install the required libraries

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${FAIRMOT_ROOT}
```

### 4. Install the requirements

```bash
pip install cython
pip install -r requirements.txt
```

### 5. Clone the DCNv2 repository

```bash
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

### 6. Download and move your FairMOT model

🔗 [fairmot_dla34.pth](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?spm=a2ty_o01.29997173.0.0.192f5171d364Xo&usp=sharing)

(or other models from the FairMOT repo)

```bash
mkdir -p models
cp /path/to/your/fairmot_dla34.pth models/
```

### 7. Set PYTHONPATH and create results dir

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/lib"
mkdir -p results
TIMESTAMP=$(date +%Y%m%d_%H%M%S) # To differentiate between multiple inference runs, skip if unwanted 
```

### 8. Move input video to FairMOT dir

```bash
cp /path/to/your/video.mp4 ./input_video.mp4
```

### 9. Fix numpy issues

Run the fix_numpy_errors.py script in the fairmot dir

### 10. Run inference

```bash
python src/demo.py mot \
  --input-video input_video.mp4 \
  --output-root results/run_$TIMESTAMP/ \ # If TIMESTAMP is skipped in 7. replace by --output-root results/ \
  --load_model models/fairmot_dla34.pth \
  --conf_thres 0.4 \ # Adjust threshold 0.3 to 0.7
  --gpus -1 \ # For CPU, GPU usage --> -- gpus 0 
```

# Key Observations

✅ Tracks 4 players with almost no ID switches — excellent reID capability
⚠️ Bounding boxes often cut off hands or feet, especially during swings
⚠️ Occasionally loses track during fast rallies or partial occlusion


📝 Conclusion
The baseline FairMOT model is already highly effective for multi-player tracking in Padel matches. Its re-identification head performs robustly even when players cross paths.

However, limitations in body coverage and rare ID switches suggest that fine-tuning on domain-specific data (our Padel dataset) will further improve accuracy and stability.

This inference test validated our decision to use FairMOT as the foundation of our tracking pipeline.