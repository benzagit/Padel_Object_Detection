# TrackernetV3 for Padel Ball Detection

> Evaluating TrackNetV3 — a SOTA model for small-object tracking — on real Padel match footage. Despite correct pipeline execution, no ball detections were observed, highlighting challenges in cross-domain generalization.

---

## 🧪 Experimental Setup

The goal was to test whether **TrackNetV3**, pretrained on tennis data, could detect padel balls without fine-tuning.

- **Model**: [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- **Weights**: `TrackNet_best.pt` (pretrained on tennis matches)
- **Input Video**: `clip_343.mp4` (1920×1080, 30 FPS, ~12 seconds)
- **Inference Script**: Official `predict.py` with fixes for Colab compatibility
- **Output**: 
  - `clip_343_ball.csv` → predicted coordinates
  - `clip_343.avi` → video with overlay

Command used:

```bash
python predict.py \
  --video_file /content/clip_343.mp4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir prediction \
  --output_video
```

📝 Conclusion

TrackNetV3 failed to detect padel balls due to significant domain mismatch:
    + Ball appearance: White vs yellow
    + Motion dynamics: Faster serves, more blur
    + Lighting: Indoor arena vs outdoor tennis court
    + Camera angle & scale: Different framing
    
While the model works well on tennis, it cannot generalize to padel without fine-tuning on padel-specific data.
