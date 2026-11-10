# Facial Expression Recognition Using Deep Learning: Improving Robustness Under Real-World Conditions

**Course:** CSE 455 Computer Vision, Autumn 2026
**Team:** Hsu Wai Hnin Kyaw, Larrianna Warner, Xuan Nhu Tran, Rina Yuan
**Repo:** https://github.com/hsu01/emotion-detection.git

## Project overview

Facial expressions are a key form of nonverbal communication, and automatic
expression recognition has applications in human–computer interaction,
education, and accessibility. This project builds a deep-learning system that
recognizes a person's facial expression from an image and investigates how
real-world conditions affect recognition performance.

**Input:** a cropped face image (from FER2013, or a webcam frame).
**Output:** one of seven emotions — **angry, disgust, fear, happy, sad,
surprise, neutral** — with per-class confidence scores.

**Research questions**
1. How much do changes in lighting, image rotation, and partial occlusion
   degrade a standard expression classifier, and can targeted data
   augmentation recover the lost accuracy?
2. Are pretrained models inherently more robust to these distortions than
   models trained from scratch, and do they benefit equally from
   augmentation?

## Experiment design (2×2)

| Experiment | Model | Augmentation | Notebook |
|---|---|---|---|
| **A** | CNN trained from scratch | No | `notebooks/01_baseline_cnn.ipynb` |
| **B** | CNN trained from scratch | Yes | (to come) |
| **C** | Pretrained ResNet18 | No | (to come) |
| **D** | Pretrained ResNet18 | Yes | (to come) |

All four experiments use the same train/validation/test splits so results are
directly comparable. Augmentation is applied to training images only — never
to validation or test images.

## Robustness conditions

Each trained model is evaluated on identical distorted versions of the
held-out test set:

| Condition | Transformation |
|---|---|
| Dark lighting | Brightness × 0.5 |
| Bright lighting | Brightness × 1.5 |
| Rotation | Rotate ±15°, ±30° |
| Eye occlusion | Cover the eye region |
| Mouth occlusion | Cover the lower face |

FER2013 is already cropped and aligned, so these artificial distortions
simulate real-world conditions but do not fully represent live webcam
conditions — the demo and error analysis (Nov 30 milestone) cover that gap.

## Folder layout

```
emotion-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # FER2013 dataset (NOT committed — download via Kaggle)
├── notebooks/
│   └── 01_baseline_cnn.ipynb   # Experiment A: baseline CNN, no augmentation
├── src/                   # shared modules: model, dataset, training, evaluation
├── results/               # figures and tables (generated, NOT committed)
└── demo/                  # webcam demo (Nov 30 milestone)
```

## Setup

```bash
git clone https://github.com/hsu01/emotion-detection.git
cd emotion-detection
```

**Python 3.10+**, then:

```bash
pip install -r requirements.txt
```

**Dataset — FER2013 from Kaggle** ([msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013),
~35,887 grayscale 48×48 face images):
1. Create a Kaggle account, go to Settings → **Create New API Token**.
   That downloads `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
3. Never commit `kaggle.json` — it is in `.gitignore`.

**Google Colab (recommended for training):** open the notebook in Colab and
choose Runtime → Change runtime type → **GPU** (T4).

## Running the baseline notebook (Experiment A)

1. Open `notebooks/01_baseline_cnn.ipynb` in Google Colab — upload the file
   (File → Upload notebook), or open it from this repo if it is on GitHub.
2. Set the runtime to GPU: Runtime → Change runtime type → **T4 GPU**.
3. **What to fill in:**
   - Kaggle credentials — either save `KAGGLE_USERNAME` and `KAGGLE_KEY` as
     Colab Secrets (key icon in the left sidebar), or upload `kaggle.json`
     and copy it to `~/.kaggle/` (the notebook explains both).
   - `DATA_ROOT` (default `/content/data`) — only if your FER2013 folders
     land somewhere else.
   - `CHECKPOINT_DIR` (default `./checkpoints`) — point it at a Drive folder
     if you want checkpoints to survive the Colab runtime being recycled.
4. Run the cells top to bottom. The notebook downloads FER2013, plots samples
   and the class distribution, builds a stratified 90/10 train/val split
   (seed 42, official test set untouched), trains a 3-block CNN for 15
   epochs (Adam, lr 0.001, batch 64), and evaluates the best checkpoint.
5. It produces `checkpoints/best_baseline_cnn.pth`,
   `checkpoints/history_baseline.json`, a per-class classification report,
   a normalized confusion matrix, and train/val loss and accuracy curves.

## Milestones

| Date | Milestone |
|---|---|
| Nov 9 | FER2013 preprocessing done, baseline CNN trained, initial accuracy recorded (Experiment A) |
| Nov 16 | Pretrained model fine-tuned, compared with the baseline; data augmentation implemented (Experiments B–D) |
| Nov 23 | Robustness tested (lighting, rotation, occlusion); impact of augmentation evaluated |
| Nov 30 | Webcam demo working in real time; full error analysis completed |
| Dec 10 | **Presentation** |
| Dec 14 | **Final report due** |

## Notes

- `data/` and `checkpoints/` are **not committed** (see `.gitignore`). The
  dataset is downloaded from Kaggle; checkpoints are regenerated by
  training. `results/` holds generated figures and tables and is likewise
  not committed.
- The official FER2013 test set stays untouched until final evaluation.
  Validation is a stratified 90/10 split of the train set (seed 42).
- FER2013 is class-imbalanced (disgust is rare, happy is common), so
  evaluation reports per-class precision/recall/F1 and a confusion matrix,
  not just overall accuracy.
