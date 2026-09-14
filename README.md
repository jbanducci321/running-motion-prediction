# running-motion-prediction

Real-time human motion capture and prediction system comparing two ways of recovering 3D human pose — single-camera learned depth inference vs. multi-camera geometric triangulation — feeding a shared motion prediction model.

CSUMB capstone project (Jacob Banducci, Shannyn Cabi, Daniel Everman; faculty sponsor Dr. Feng).

## What this is

Two pipelines produce a 3D skeleton stream in the same joint format:

- **Single-camera**: pretrained [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) infers depth from learned temporal patterns in one video feed. No calibration needed — also the fallback plan if the multi-camera work runs out of time.
- **Multi-camera**: [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) runs independently on 4 physical webcams, combined through our own camera calibration and triangulation code.

Both feed a single RNN/GRU sequence-to-sequence model, trained on [Human3.6M](http://vision.imar.ro/human3.6m/), that forecasts movement 0.5-2s into the future. Live captures of the team are used only as a generalization test set, never for training.

**Central research question:** does multi-camera triangulation produce meaningfully more accurate depth than single-camera learned inference, or does the single camera hold up fine without the added calibration cost?

## Status

Early scaffolding. The canonical 17-joint skeleton format (shared by Human3.6M and VideoPose3D, with a MediaPipe adapter) is the first real implementation milestone — see `skeleton_adapter/`.

## Repo layout

- `skeleton_adapter/` - converts H36M / VideoPose3D / MediaPipe output into one canonical 17-joint skeleton format (`configs/skeleton.yaml`)
- `single_camera/` - VideoPose3D pipeline (Track 1)
- `multi_camera/` - camera calibration, MediaPipe capture, triangulation (Track 2)
- `prediction/` - H36M preprocessing, RNN/GRU model, training loop (Track 3)
- `configs/` - shared config (skeleton layout, camera calibration, etc.)
- `data/` - gitignored; H36M and capture data live here locally, never committed (H36M's license prohibits redistribution)
- `tests/` - pytest suite

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

Install PyTorch first, matching your platform/GPU setup (see [pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Licensing note

Project code is MIT licensed (see `LICENSE`). Two dependencies carry more restrictive terms worth knowing about: VideoPose3D is CC-BY-NC-4.0 (non-commercial), and Human3.6M's license is research/education-only and prohibits redistributing the dataset. Neither affects this project's own MIT license, and neither should ever be committed to this repo.
