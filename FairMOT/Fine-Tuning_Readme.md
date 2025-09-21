# Fine-Tuning FairMOT on Custom Padel Dataset

> Adapting FairMOT to better detect and track players in Padel matches by fine-tuning on our internally annotated dataset — improving body coverage, reducing ID switches, and enhancing domain alignment.

## 🧰 Why Fine-Tune?

While the baseline FairMOT model performed well, we observed:
- Bounding boxes often excluded **hands and limbs**
- Occasional **ID switches** during fast motion

These issues stem from domain mismatch: the original model was trained on general crowd scenes (CrowdHuman) and generic sports (MOT17), not racquet sports like Padel.

Fine-tuning allows the model to:
- Learn padel-specific appearance patterns
- Improve full-body localization
- Reduce false ID switches
- Adapt to indoor court conditions

## 🗂️ Dataset Preparation

Our dataset consisted of:
- 4,500+ annotated frames from YouTube Padel matches
- Labels exported from CVAT in .xml format
- Converted to FairMOT-compatible structure ; .xml to gt.txt using the script convert_xml.py

**MOT Challenge format** (`labels_with_ids/train/*.txt`) with one line per object:
<class_id> <track_id> <x_center> <y_center> <width> <height>

Our raw annotations came from **CVAT** in Pascal VOC `.xml` format — one file per frame. We had to:
- Parse all `.xml` files across multiple clips
- Assign consistent `track_id` per player (1–4)
- Normalize coordinates
- Export into MOT-compatible `.txt` files

---

### 🗂️ Final Directory Structure

datasets/
└── padel_mot/
├── images/
│ └── train/
│ ├── clip_001/
│ │ ├── img_000001.jpg
│ │ └── ...
│ ├── clip_002/
│ └── ...
└── labels_with_ids/
└── train/
├── clip_001.txt
├── clip_002.txt
└── ...

## 🛠️ Training Configuration

Modified `src/lib/cfg/padel.json`:

```json
{
  "num_classes": 1,
  "reid_dim": 128,
  "lr": 1.25e-4,
  "lr_step": [45, 60],
  "batch_size": 8,
  "max_frame_num": 1500,
  "input_h": 608,
  "input_w": 1088,
  "down_ratio": 4
}
```

### Training the model

+ Using the same model we used for the inference part fairmot_dla34.pth
+ Resizing frames is necessary to match fairmot backbone resolution (resize to 1088x608 preferably)
+ Run the training/fine tuning command

```bash
cd /FairMOT/src

python train.py mot \
  --exp_id test_save_model \
  --dataset new_dataset \
  --data_cfg '/FairMOT/src/lib/cfg/data.json' \
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

📝 Conclusion

Fine-tuning FairMOT on our custom Padel dataset significantly improved tracking stability and detection quality. The model learned to better localize players’ full bodies and maintain identity through complex motions.

This demonstrates the importance of domain-specific training data in computer vision pipelines — even for SOTA models.

Future work includes integrating ball detection (via TrackernetV3) and deploying the full pipeline on edge devices for real-time analysis.