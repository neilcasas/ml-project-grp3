# Comparative Analysis of CNN and Efficient Vision Transformers for Gleason Grade Classification

**Course:** CSELEC2C Machine Learning (2025–2026)  
**Dataset:** SICAPv2 — 18,783 prostate histopathology patches, 4-class (NC / G3 / G4 / G5)  
**Models:** ResNet-50 · TinyViT-11M · FastViT-T8

This project compares CNN, hybrid CNN-Transformer, and reparameterized efficient ViT architectures for patch-level Gleason grade classification. The goal is to identify which architectural design sits on the Pareto frontier of accuracy vs. parameter efficiency vs. inference speed for real-world pathology deployment.

---

## Dataset

**SICAPv2** must be downloaded manually from Mendeley Data:  
https://data.mendeley.com/datasets/9xxm58dvs3/

After downloading, extract so the layout is:

```
<project_root>/SICAPv2/
    images/          # 18,783 .jpg patches (512×512, 10X magnification)
    masks/           # pixel-level annotation masks
    partition/
        Test/        # Test.xlsx, Train.xlsx
        Validation/
            Val1/ Val2/ Val3/ Val4/
    readme.txt
    wsi_labels.xlsx
```

**Label scheme:** `NC` (non-cancerous), `G3`, `G4`, `G5` — taken from one-hot columns in the partition xlsx files. The `G4C` cribriform flag is present but unused as a class.

**Splits used:**
- Train: `partition/Validation/Val1/Train.xlsx`
- Val:   `partition/Validation/Val1/Test.xlsx`
- Test:  `partition/Test/Test.xlsx`

Verify the layout:
```bash
python data/download_sicapv2.py
```

---

## Environment Setup

Python 3.10+, CUDA-capable GPU recommended (tested on Colab T4).

```bash
pip install -r requirements.txt
```

---

## Reproducing Results

All commands are run from the project root (`ml_project_implementation/`).

### 1. Verify dataset
```bash
python data/download_sicapv2.py
```

### 2. Train all three models
```bash
python training/train.py --config training/config.yaml --model all
```

Or train individually:
```bash
python training/train.py --config training/config.yaml --model resnet50
python training/train.py --config training/config.yaml --model tiny_vit_11m
python training/train.py --config training/config.yaml --model fastvit_t8
```

### 3. Evaluate on the test set
```bash
python evaluation/metrics.py --config training/config.yaml --model all
```

### 4. Benchmark inference speed
```bash
python evaluation/inference_speed.py --config training/config.yaml
```

### 5. Generate Grad-CAM visualizations
```bash
python evaluation/gradcam.py --config training/config.yaml --model all
```

### One-click (Colab notebook)
Open `notebooks/main.ipynb` in Google Colab. The notebook:
1. Checks for `SICAPv2/` at the project root — raises a clear error if missing
2. Optionally mounts Google Drive for checkpoint saving
3. Runs training, evaluation, speed benchmarking, and Grad-CAM in sequence

---

## Results

| Model | Test Acc | Params | ms/image |
|---|---|---|---|
| ResNet-50 | TBD | 25.6M | TBD |
| TinyViT-11M | TBD | 11M | TBD |
| FastViT-T8 | TBD | 8M | TBD |

Results and checkpoints are saved to `results/`.

---

## Project Structure

```
ml_project_implementation/
├── data/
│   ├── dataset.py              # SICAPv2Dataset (reads partition xlsx, returns image + label)
│   ├── transforms.py           # Train/val augmentation pipelines
│   └── download_sicapv2.py     # Layout verification script
├── models/
│   ├── load_models.py          # timm model loading + head replacement
│   └── classifier_heads.py     # freeze_backbone / unfreeze_all utilities
├── training/
│   ├── train.py                # Main training loop (two-phase fine-tuning)
│   ├── config.yaml             # All hyperparameters (single source of truth)
│   └── utils.py                # Seed, checkpointing, AverageMeter
├── evaluation/
│   ├── metrics.py              # Test-set accuracy + per-class breakdown
│   ├── gradcam.py              # Grad-CAM for all 3 architectures
│   └── inference_speed.py      # ms/image benchmark
├── notebooks/
│   └── main.ipynb              # End-to-end Colab notebook
├── results/
│   ├── checkpoints/            # Saved per epoch + best_*.pt per model
│   ├── figures/                # Grad-CAM heatmaps, training curves, tradeoff plot
│   └── metrics.json            # Final accuracy + speed + params
├── SICAPv2/                    # Dataset (not committed — download separately)
├── requirements.txt
└── README.md
```

---

## Reproducibility

- Random seeds fixed: `torch`, `numpy`, `random`, `cuda` — all set to 42
- All hyperparameters in `training/config.yaml` (single config file)
- Best checkpoints saved per model to `results/checkpoints/`
- `requirements.txt` with pinned versions

---

## LLM Disclosure

*[List here which specific parts of the code and manuscript were generated or significantly assisted by an LLM, per course policy. Failure to disclose results in deductions.]*
