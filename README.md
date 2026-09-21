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

The canonical 17-joint skeleton format (`skeleton_adapter/`) is done and verified against real webcam captures. Both single-frame pose sources - MediaPipe and VideoPose3D - have working, tested integration scripts. Camera calibration/triangulation (multi-camera) and the prediction model itself (needs Human3.6M dataset access, currently pending) haven't been started yet.

## Repo layout

- `skeleton_adapter/` - converts H36M / VideoPose3D / MediaPipe output into one canonical 17-joint skeleton format (`configs/skeleton.yaml`)
- `single_camera/` - VideoPose3D pipeline (Track 1): `videopose3d_model.py` (vendored model architecture) + `inference.py` (2D-to-3D wrapper)
- `multi_camera/` - camera calibration, MediaPipe capture, triangulation (Track 2) - not started yet
- `prediction/` - H36M preprocessing, RNN/GRU model, training loop (Track 3) - not started yet, blocked on H36M dataset access
- `configs/` - shared config (skeleton layout, camera calibration, etc.)
- `scripts/` - runnable test/demo scripts, see below
- `data/` - gitignored; model checkpoints, H36M data, and capture data live here locally, never committed
- `tests/` - pytest suite

## Setup

### 1. Clone and create a virtual environment

Requires Python 3.11+ (tested on 3.14). MediaPipe tends to lag behind on supporting the newest CPython release, so if setup fails on a brand-new Python version, that's the first thing to check.

```bash
git clone https://github.com/jbanducci321/running-motion-prediction.git
cd running-motion-prediction
python -m venv .venv
```

Activate it - on Windows (PowerShell/cmd):

```bash
.venv\Scripts\activate
```

or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install PyTorch

Install this *before* the rest of the requirements, matching your platform/GPU. Check whether you have an NVIDIA GPU first:

```bash
nvidia-smi
```

If that prints a GPU/driver table, install the CUDA build (adjust the CUDA version to what `nvidia-smi` reports if needed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Otherwise (CPU-only, or on the fence - CPU is fine for everything in this repo so far):

```bash
pip install torch torchvision
```

### 3. Install everything else

```bash
pip install -r requirements.txt
```

### 4. Download the model checkpoints

Two pretrained model files aren't in git (they're binary and large) and need to be downloaded once into `data/models/`:

```bash
mkdir -p data/models
curl -L -o data/models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o data/models/pretrained_h36m_detectron_coco.bin https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
```

A third model (YOLOv8n-pose, used as VideoPose3D's 2D keypoint detector) downloads itself automatically into `data/models/` the first time you run the VideoPose3D script below - no manual step needed for that one.

### 5. Verify the install

```bash
pytest
ruff check .
```

Both should pass clean with no camera or dataset needed.

## Scripts

All of these live in `scripts/` and are run from the repo root (e.g. `python scripts/validate_skeleton_adapter.py`).

- **`validate_skeleton_adapter.py`** - builds the same synthetic T-pose in all three raw formats (H36M, VideoPose3D, MediaPipe), runs each through its adapter, and plots all three side by side. No camera needed; good first check that the joint mapping itself is correct. Output: `data/skeleton_adapter_validation.png`.
- **`live_test_mediapipe.py [camera_index] [countdown_seconds]`** - captures one frame from a real webcam, runs MediaPipe, and saves the 2D detection overlay plus the resulting canonical skeleton plot. Stand far enough back that your whole body is in frame. Outputs: `data/live_test_annotated.png`, `data/live_test_skeleton.png`.
- **`live_test_videopose3d.py [camera_index] [countdown_seconds] [num_frames]`** - records a short burst of frames (default 30), runs YOLOv8-pose per frame, lifts the sequence to 3D with VideoPose3D, and saves the same kind of overlay + skeleton plot for the middle frame. Needs a burst rather than one frame because VideoPose3D is a temporal model with a 243-frame receptive field (short bursts get edge-padded up to that, the same way VideoPose3D's own code handles video edges). Outputs: `data/live_test_videopose3d_annotated.png`, `data/live_test_videopose3d_skeleton.png`.
- **`live_demo.py [camera_index]`** - continuous live version of the MediaPipe test: an OpenCV window with the raw 2D overlay next to a real-time VPython 3D rig of the canonical skeleton. Press `q` in the webcam window to quit.

## Known caveats

- **2D detector substitutions**: VideoPose3D's official 2D keypoint detector is Facebook's Detectron2, which doesn't install cleanly on Windows. This project uses Ultralytics YOLOv8-pose instead (same COCO-17 keypoint format the pretrained checkpoint expects, just not the exact detector it was calibrated against). Fine for functional testing; worth reconsidering before final published accuracy numbers.
- **VideoPose3D output units**: empirically verified (not documented upstream) to be in meters, not H36M's native millimeters - already handled in `single_camera/inference.py`, but worth knowing if you touch that code.
- **Single-frame depth noise**: both pose sources estimate depth from monocular input, so a single frame/burst can show implausible bone-length asymmetry (e.g. one arm looking much longer than the other) purely from 2D detector noise on that one capture, not a bug in the adapter. This is expected, and is part of the motivation for the multi-camera triangulation arm of this project.

## Licensing note

Project code is MIT licensed (see `LICENSE`), with one exception: `single_camera/videopose3d_model.py` is vendored directly from VideoPose3D and is covered by VideoPose3D's own CC-BY-NC-4.0 license (non-commercial use only), not this project's MIT license. Human3.6M's license is research/education-only and prohibits redistributing the dataset. Nothing under `data/` should ever be committed to this repo.
