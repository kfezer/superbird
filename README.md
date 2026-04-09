# superbird buddy 🐦

Tiered bird species classifier. Local [MobileNetV3](https://pytorch.org/vision/stable/models/mobilenet_v3_small.html) runs first — fast, free, offline. Images the local model isn't confident about escalate to a [LLaVA-1.5](https://llava-vl.github.io/) vision model deployed on [Baseten](https://www.baseten.co/).

Built as a DevRel demo for Baseten, but has pratical use and I've wanted to build it for a while. The real pattern: edge-first inference with cloud fallback, automatic logging of hard cases for future fine-tuning.

```
photo.jpg → MobileNetV3 (local, CPU)
                │
         confidence ≥ 70%? ──yes──► "Painted Bunting" ✓
                │
               no
                │
                └──────────────────► Baseten (LLaVA-1.5-7B, A10G)
                                           │
                                     logs/fallback_log.csv ← hard cases
                                     (training data for next round)
```

---

## Project structure

```
superbird_buddy_bastenfallback/
├── app.py                       # CLI: classify images, view fallback stats
├── pyproject.toml               # dependencies (source of truth)
├── .python-version              # pins Python 3.11 for uv
│
├── classifier/
│   ├── local_model.py           # MobileNetV3-Small, 200 CUB-200 species
│   ├── baseten_model.py         # Baseten REST client (LLaVA)
│   └── router.py                # confidence threshold + fallback logger
│
├── truss_fallback/              # inference Truss — always-on, low latency
│   ├── config.yaml              # A10G, LLaVA-1.5-7B (YAML — serving models use this format)
│   └── model/model.py           # LLaVA-1.5 Truss wrapper
│
├── truss_train/                 # training Truss — on-demand, GPU fine-tuning
│   ├── config.py                # TrainingProject definition: A10G, cache, checkpointing
│   ├── train.sh                 # Runtime entrypoint: pip install remaining deps + python train.py
│   └── train.py                 # downloads CUB-200 from Caltech, trains, prints weights as base64
│
├── scripts/
│   ├── prepare_cub200.py        # raw CUB-200-2011 download → ImageFolder split (local only)
│   ├── train_local.py           # fine-tune on your own machine (CPU/MPS/GPU)
│   └── train_baseten.py         # submit async training job to Baseten + poll
│
└── logs/
    └── fallback_log.csv         # auto-created; one row per Baseten inference call
```

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — install it once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Install

```bash
# Clone and enter the project
git clone <repo-url>
cd superbird_buddy_bastenfallback

# Create .venv and install all runtime dependencies
uv sync
```

`uv sync` reads `.python-version` (Python 3.11) and `pyproject.toml`, creates `.venv/`, and generates `uv.lock`. Commit `uv.lock` — it's what makes the environment reproducible.

### Install the Baseten CLI

The `baseten` deploy SDK has a dependency conflict with PyTorch (it pins `numpy==1.23.5` via `truss`), so it lives in its own isolated environment managed by uv — it never touches your project's venv:

```bash
uv tool install baseten

# Add uv tools to your PATH (only needed once)
uv tool update-shell
# then restart your shell, or: source ~/.profile
```

---

## Running the classifier

```bash
# Set your credentials
export BASETEN_API_KEY=your_key
export BASETEN_MODEL_ID=your_deployed_llava_model_id

# Classify a single image (default threshold: 70%)
uv run python app.py photo.jpg

# Classify multiple images with a custom threshold
uv run python app.py *.jpg --threshold 0.65

# Show fallback log stats (which images triggered Baseten, agreement rate, etc.)
uv run python app.py --log-stats
```

### Example: local model confused, Baseten corrects it

```bash
uv run python app.py examples/eastern_yellow_robin.jpg
```

```
[local] Loaded weights from weights/mobilenet_cub200.pt

eastern_yellow_robin.jpg
  Species:    Yellow Warbler
  Confidence: 95% [baseten fallback]
  Local was:  Tropical Kingbird (22%)
```

The local model guessed "Tropical Kingbird" at 22% confidence — below the 70% threshold.
Baseten's LLaVA model correctly identified it as a Yellow Warbler at 95%.

---

## Training the local model on Baseten (A10G GPU, ~15 min)

The training job downloads CUB-200-2011 directly from
Caltech (~1.1 GB) and returns the finished weights as part of the API response.

```bash
# 1. Deploy the training Truss (only needed once)
baseten deploy --truss ./truss_train
export BASETEN_TRAINER_ID=<model-id-from-deploy-output>

# 2. Submit the job — Baseten downloads CUB-200, trains, sends weights back
uv run python scripts/train_baseten.py --epochs 20
# Polls every 30s; saves weights/mobilenet_cub200.pt when done.
```

Hyperparameters:
```bash
uv run python scripts/train_baseten.py --epochs 30 --batch-size 32 --lr 5e-4
```

How it works under the hood:
- `config.py` defines the `TrainingProject`: base image, A10G compute, `train.sh` as the entrypoint
- `train.sh` pip installs the remaining deps (torch is pre-installed in the base image) and calls `train.py`
- `train.py` downloads CUB-200 from Caltech, fine-tunes on the A10G, and prints weights as base64 to stdout
- `train_baseten.py` submits the job, polls the async API, decodes the base64 and saves the `.pt` file locally

---

## Deploying the inference model on Baseten

```bash
# Deploy LLaVA-1.5-7B (used for fallback inference)
baseten deploy --truss ./truss_fallback

# Copy the model ID from the deploy output, then:
export BASETEN_MODEL_ID=<model-id>
```

The inference Truss (`truss_fallback/`) stays warm. The training Truss (`truss_train/`) only runs when you submit a job — no idle GPU cost between training rounds.

---

## Day-to-day commands

```bash
uv sync                                   # install / update after a git pull
uv run python app.py photo.jpg            # classify with the managed venv
uv run python scripts/train_baseten.py --help

# Regenerate requirements.txt (for tools that need a flat pip file)
uv export --no-hashes > requirements.txt
```

---

## How the fallback loop works

Every image that falls below the confidence threshold gets logged to `logs/fallback_log.csv`:

```
timestamp, image_path, local_label, local_confidence, remote_label, remote_confidence
2026-04-07T12:00:00, warbler.jpg, House Sparrow, 0.4821, Yellow Warbler, 0.9300
```

Run `python app.py --log-stats` to see agreement rates and the lowest-confidence images — those are your best candidates for the next fine-tuning batch. Train a new round on Baseten, download the weights, and the local model gets better. Fewer fallbacks. Lower cost over time.
