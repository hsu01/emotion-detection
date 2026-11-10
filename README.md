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

## Current status

As of Nov 10, 2026, the baseline notebook exists and has been partially run
in Colab: the T4 GPU was detected, FER2013 downloaded successfully, and
image counts and sample plots were checked. There are 28,709 official
training images and 7,178 held-out test images. No model has been trained
and no accuracy or F1 results exist yet. Experiments B–D are not implemented.

## Setup — Google Colab (recommended)

You do not need to install Python or download the dataset on your laptop
when using Colab. Have a Google account with Drive access and your own
Kaggle account ready.

### 1. Get Kaggle credentials

1. Sign in at https://www.kaggle.com/settings/api.
2. Under **Legacy API Credentials**, choose **Create Legacy API Key** to
   download `kaggle.json`. This is the credential format used by this notebook.
3. Keep the file private. Never commit it, paste its contents into chat,
   or include it in shared screenshots. Each teammate should use their own key.
4. Visit the [FER2013 dataset page](https://www.kaggle.com/datasets/msambare/fer2013)
   while signed in and complete any access prompts.

### 2. Open a working copy in Colab

1. Go to https://colab.research.google.com/.
2. Choose **Open notebook → GitHub**, enter
   `https://github.com/hsu01/emotion-detection`, and select
   `notebooks/01_baseline_cnn.ipynb`.
3. Click **Copy to Drive** and work in that copy. This does not update GitHub;
   record the source commit when recording an experiment.
4. Choose **Runtime → Change runtime type → T4 GPU**, save, then **Connect**.
5. In the left sidebar, open **Secrets** (key icon). Add `KAGGLE_USERNAME`
   and `KAGGLE_KEY` using the `username` and `key` values from `kaggle.json`,
   without quotation marks. Enable **Notebook access** for both.

### 3. Run setup and check the data

Run cells individually using the **▶** button; do not use **Run all** on
this first run.

1. Run the installation cell. Also add and run a Colab code cell containing:
   ```python
   !pip install -q kaggle
   ```
   The current notebook installation cell does not include Kaggle.
2. Run the GPU-check cell. Confirm `CUDA available: True` and
   `Using device: cuda`. Stop and resolve this if CUDA is unavailable.
3. Run the credentials/download cell. Confirm the download succeeds and
   `/content/data` contains `train/` and `test/`. The message
   `Download finished.` alone does not prove the shell command succeeded.
4. Run the folder/count check and exploration cells. Expected totals:
   **28,709 train**, **7,178 test**, **35,887 combined**. Counting test files
   is a dataset integrity check; do not evaluate models on test images yet.

### 4. Required grayscale fix before preprocessing

The committed notebook currently needs this correction: `ImageFolder`
loads images as RGB by default, while the baseline CNN expects one channel.
In your Colab copy, add `transforms.Grayscale(num_output_channels=1)` before
`Resize` in **both** `train_transform` and `eval_transform`:

```python
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

Run section 3 without changing the split, seed, or batch size. Expected output:

```text
Train: 25838 | Val: 2871 | Test (held out): 7178
Batch: torch.Size([64, 1, 48, 48]) torch.Size([64])
```

Then run the model cell in section 4; its output shape must be `(2, 7)`.
Class index order is `angry, disgust, fear, happy, neutral, sad, surprise`.
Use that order consistently when interpreting predictions.

## Running Experiment A

### Save checkpoints to Drive

Before training, replace the checkpoint-directory cell in your Colab copy with:

```python
from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/emotion-detection/checkpoints/A_smoke'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print('Checkpoints will be saved to:', CHECKPOINT_DIR)
```

Colab runtime files under `/content` can disappear when the runtime is recycled.
Mounting Drive preserves files written to the Drive path, but you still need to
rerun setup and download data in a fresh runtime.

### First: two-epoch smoke test

1. In the training cell, change `NUM_EPOCHS = 15` to `NUM_EPOCHS = 2`.
2. Run the training cell. Keep Adam, learning rate 0.001, batch size 64,
   and `CrossEntropyLoss` unchanged.
3. Confirm both epochs finish, losses are finite, and the checkpoint and
   history files appear in `A_smoke`.
4. Run section 6 to verify that validation metrics and plots work. This is
   a pipeline check, not the final baseline result. Do not run test evaluation.

### Then: fresh 15-epoch baseline

1. Rerun the section 3 preprocessing/split cell to reset the seeds and loaders,
   then rerun the section 4 model cell to create a fresh model. Do not continue
   training the smoke-test model.
2. Change `CHECKPOINT_DIR` to
   `/content/drive/MyDrive/emotion-detection/checkpoints/A_baseline`
   and rerun the checkpoint-directory cell.
3. Set `NUM_EPOCHS = 15` and rerun the training cell. This creates a new Adam
   optimizer and resets the training history. The best checkpoint is selected
   by **validation accuracy**.
4. Run section 6, which evaluates the best checkpoint on **validation only**.

Record the source commit, Colab edits, package versions, split seed, settings,
best epoch, validation accuracy, per-class precision/recall/F1, macro F1,
confusion matrix, and training curves. Training duration is not yet measured.

Files written by training:
- `best_baseline_cnn.pth`
- `history_baseline.json`

The report and plots are displayed in notebook outputs; the current notebook
does not automatically save them as separate files. Download the plots from
Colab and keep them with the run artifacts in Drive. `results/` and checkpoints
are ignored by Git, so share artifacts through Drive separately from code.
Never include Kaggle credentials in shared artifacts or notebook outputs.

## Optional local setup

Colab is the documented training path above. To install dependencies locally:

```bash
git clone https://github.com/hsu01/emotion-detection.git
cd emotion-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebook currently uses Colab-specific paths and credentials; local execution
requires adapting those cells. Installing dependencies locally does not connect
your laptop to a Colab GPU.

## Collaborating

- Use a separate Git branch for changes and open a pull request for review.
- Colab Drive copies do not sync back to GitHub. Save intended code changes
  back to your branch explicitly and remove private outputs before sharing.
- Keep experiment notebooks in `notebooks/` and shared modules in `src/`.
- Preserve the stratified 90/10 train/validation split with seed 42 across A–D.
  Use the same ordered source images and split indices for every experiment.
- Apply augmentation to training images only. Keep validation clean and the
  official test set held out until final evaluation.
- Share checkpoints and generated results separately through Drive, with clear
  experiment names and settings. Do not commit the dataset or credentials.

## Milestones

| Date | Milestone |
|---|---|
| Nov 10–12 | Dataset setup, preprocessing, smoke test, and baseline CNN (Experiment A) |
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
