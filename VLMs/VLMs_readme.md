# VLM Experiments: Zero-Shot Detection for Padel Annotation

> Testing Vision-Language Models (VLMs) like OWL-v2 for semi-automated annotation of players and ball in Padel matches — can natural language replace manual labeling?

---

## 🧰 Purpose

To evaluate whether zero-shot object detection models could accelerate the annotation process by detecting:
- **4 unique players** (2v2) using prompts like `"player"` or `"player closest to the left"`
- The **small, fast-moving white padel ball**
- Without requiring training data or fine-tuning

This was part of my internship work, where we explored alternatives to traditional object detectors (like YOLO) that failed on ID persistence and small-object detection.

---

## 🛠️ Model Used: OWL-ViT (Google)

We tested **OWL-ViT v2**, a vision-language model that combines CLIP-style image-text embeddings with object detection.

- **Model**: [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble)
- **Framework**: Hugging Face Transformers
- **Input**: Image + list of text queries (e.g., `["player", "padel_ball"]`)
- **Output**: Bounding boxes with confidence scores

### Installation & Setup

```bash
!pip install torch torchvision transformers pillow requests matplotlib scipy
```

Load Model
```bash
from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
```

🎯 Detection Pipeline
Step 1: Extract Frames from Video
We uploaded a Padel match clip and extracted 5 random frames for testing:

```bash
import cv2
import os
import numpy as np
from google.colab import files

# Upload video
uploaded = files.upload()
video_file = list(uploaded.keys())[0]

# Save random frames
cap = cv2.VideoCapture(video_file)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_indices = np.random.choice(frame_count, size=5, replace=False)

os.makedirs('/content/frames', exist_ok=True)
for idx in random_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'/content/frames/frame_{idx}.jpg', frame)
cap.release()
```

Step 2: Run Zero-Shot Detection
For each frame, we ran inference with custom queries:

```bash
def detect_objects(image, queries, processor, model, threshold=0.3):
    inputs = processor(text=queries, images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )
    return results
```

Example Query

```bash
queries = ["player", "padel_ball"]
results = detect_objects(image, queries, processor, model, threshold=0.3)
```

Step 3: Visualize Results

```bash
import matplotlib.pyplot as plt
import random

def display_image_with_boxes(image, results, queries, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    res = results[0]
    boxes, scores, labels = res["boxes"], res["scores"], res["labels"]

    colors = [(random.random(), random.random(), random.random()) for _ in range(len(boxes))]

    for (box, score, label), color in zip(zip(boxes, scores, labels), colors):
        xmin, ymin, xmax, ymax = box.tolist()
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                     edgecolor=color, facecolor='none', linewidth=2))
        text = queries[0][label]
        ax.text(xmin, ymin-5, f"{text}: {score:.2f}", fontsize=12,
                backgroundcolor=color, color='white')

    plt.axis('off')
    plt.show()
```

📊 Results

+ Player detection: Reliable across majority of test frames
+ Ball detection: Failed consistently — likely due to:
            - Small size 
            - Motion blur

+ No persistent IDs: All players labeled generically → useless for tracking

🖼️ Sample Output

![Random frame from a padel video](Owlv2.png)

📝 Conclusion

While OWL-ViT shows strong zero-shot capability for general object detection, it fails to meet our core requirements for Padel annotation:
    ❌ Cannot assign unique IDs to players
    ❌ Misses fast-moving ball in all test cases

However, it proved valuable as a pre-labeling tool — reducing manual effort by providing initial bounding boxes for players.

This experiment confirmed that while VLMs are promising for exploratory tasks, they are not yet production-ready for structured sports vision pipelines requiring identity persistence and high-speed processing.