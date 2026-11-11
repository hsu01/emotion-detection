
# Facial Expression Recognition Using Deep Learning: Improving Robustness Under Real-World Conditions

**Course:** CSE 455 Computer Vision, Autumn 2026  
**Team:** Hsu Wai Hnin Kyaw, Larrianna Warner, Xuan Nhu Tran, Rina Yuan  
**Repository:** https://github.com/hsu01/emotion-detection

## Project Overview

Facial expressions are a key form of nonverbal communication, and automatic expression recognition has applications in human–computer interaction, education, and accessibility.

This project builds a deep-learning system that recognizes facial expressions from images and investigates how real-world conditions affect classification performance.

We compare convolutional neural networks trained from scratch with pretrained ResNet18 models, both with and without data augmentation.

**Input:** A cropped face image from FER2013 or a webcam frame.

**Output:** One of seven facial expression categories — angry, disgust, fear, happy, sad, surprise, or neutral — with per-class confidence scores.

### Research Questions

1. How much do changes in lighting, image rotation, and partial occlusion degrade facial expression recognition accuracy, and can targeted data augmentation improve robustness?

2. Are pretrained models more robust to these distortions than CNNs trained from scratch, and do they benefit equally from augmentation?

## Experiment Design (2×2)

We conduct four experiments to compare model architecture and data augmentation.

| Experiment | Model | Data Augmentation | Status |
|---|---|---|---|
| **A** | CNN trained from scratch | No | Training and validation completed |
| **B** | CNN trained from scratch | Yes | Pending |
| **C** | Pretrained ResNet18 | No | Pending |
| **D** | Pretrained ResNet18 | Yes | Pending |

All experiments use the same training, validation, and test splits to ensure fair comparisons.

Data augmentation is applied only to training images. Validation and test images remain unaugmented during standard accuracy evaluation.

### Experiment Notebooks

- **Experiment A:** `notebooks/01_baseline_cnn.ipynb`
- **Experiment B:** To be added
- **Experiment C:** To be added
- **Experiment D:** To be added

## Dataset — FER2013

We use the Facial Expression Recognition 2013 (FER2013) dataset, which contains 35,887 grayscale facial images across seven emotion categories.

**Dataset:** https://www.kaggle.com/datasets/msambare/fer2013

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 25,838 |
| Validation | 2,871 |
| Test (held out) | 7,178 |
| **Total** | **35,887** |

The official FER2013 training set contains 28,709 images.

We divide this set into training (90%) and validation (10%) using a stratified split with random seed 42.

The official test set remains untouched until final evaluation.

### Emotion Categories

The seven classes, in model index order, are:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

All images are resized to 48 × 48 pixels and converted to single-channel grayscale tensors.

Pixel values are normalized using mean 0.5 and standard deviation 0.5.

## Robustness Evaluation

To investigate performance under real-world conditions, we plan to evaluate trained models on identical distorted versions of the held-out test set.

| Condition | Transformation |
|---|---|
| Dark lighting | Brightness × 0.5 |
| Bright lighting | Brightness × 1.5 |
| Rotation | ±15°, ±30° |
| Eye occlusion | Cover the eye region |
| Mouth occlusion | Cover the lower face |

These transformations simulate challenging conditions such as poor lighting, head rotation, and partially obscured facial features.

Because FER2013 images are already cropped and aligned, these synthetic distortions do not fully represent live webcam conditions.

A webcam demonstration and additional error analysis are planned.

**Status:** Robustness evaluation has not yet been completed.

---

## Experiment A — Baseline CNN (No Augmentation)

**Status: Training and validation evaluation completed — November 10, 2026.**

Experiment A establishes the baseline performance of a convolutional neural network trained from scratch without data augmentation.

This baseline will be used to measure improvements from augmentation and pretrained models in Experiments B, C, and D.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | Custom CNN |
| Training | From scratch |
| Trainable parameters | 1,274,823 |
| Input dimensions | 1 × 48 × 48 |
| Output classes | 7 |
| Data augmentation | None |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 64 |
| Training epochs | 15 |
| Best checkpoint selection | Validation accuracy |
| Train/validation split | Stratified 90/10 |
| Random seed | 42 |

Both training and validation images use identical preprocessing transformations:

```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

No random rotation, flipping, brightness adjustment, or other data augmentation is applied during Experiment A training.

### Training Results

The baseline CNN was trained for 15 epochs using the FER2013 training split.

| Metric | Result |
|---|---:|
| Best validation accuracy | **56.29%** |
| Macro F1-score | **46.45%** |
| Best epoch | **14** |
| Final training accuracy | 57.88% |
| Final validation accuracy | 55.97% |
| Training epochs | 15 |

The best model checkpoint was saved at epoch 14 based on validation accuracy.

Training accuracy increased throughout the training process, while validation accuracy began to plateau toward the later epochs.

Validation loss also increased after reaching its minimum, suggesting some overfitting.

### Per-Class Validation Performance

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Angry | 0.5294 | 0.4060 | 0.4596 |
| Disgust | 0.0000 | 0.0000 | 0.0000 |
| Fear | 0.4071 | 0.2512 | 0.3107 |
| Happy | 0.7593 | 0.8213 | 0.7891 |
| Neutral | 0.5323 | 0.5806 | 0.5554 |
| Sad | 0.3733 | 0.5797 | 0.4542 |
| Surprise | 0.7917 | 0.5994 | 0.6822 |

**Overall validation accuracy:** 56.29%

**Macro F1-score:** 46.45%

### Key Observations

**1. Happy expressions are recognized most successfully.**

The model achieves 82.13% recall and an F1-score of 0.7891 for happy expressions.

**2. Disgust is particularly difficult to classify.**

The model achieves 0% recall for disgust. This category is underrepresented in FER2013, which may contribute to poor performance.

**3. Fear is frequently confused with sadness.**

The confusion matrix shows that the CNN struggles to distinguish fear from visually similar expressions.

**4. Some overfitting is observed.**

Training accuracy continues to improve while validation accuracy plateaus around 55–56%.

**5. The baseline provides a reference for future experiments.**

Experiment B will investigate whether data augmentation improves generalization.

Experiments C and D will examine the effect of transfer learning using ResNet18.

### Saved Experiment A Artifacts

The following files have been generated and verified:

| File | Description |
|---|---|
| `best_baseline_cnn.pth` | Best model checkpoint |
| `history_baseline.json` | Training and validation metrics across 15 epochs |
| `classification_report.json` | Per-class precision, recall, F1-score, and accuracy |
| `confusion_matrix.png` | Normalized validation confusion matrix |
| `training_curves.png` | Training and validation loss/accuracy graphs |

These artifacts are stored in Google Drive:

```text
MyDrive/
└── emotion-detection/
    └── checkpoints/
        └── A_baseline/
            ├── best_baseline_cnn.pth
            ├── history_baseline.json
            ├── classification_report.json
            ├── confusion_matrix.png
            └── training_curves.png
```

Model checkpoints and generated results are not committed to GitHub.

### Experiment A Notebook

The completed notebook is available at:

[Experiment A — Baseline CNN](notebooks/01_baseline_cnn.ipynb)

The notebook includes:

- Dataset download and exploration
- Data preprocessing
- Stratified training/validation split
- CNN architecture
- Model training
- Checkpoint saving
- Validation classification report
- Confusion matrix
- Training loss and accuracy visualization

**Important:** All reported Experiment A performance metrics are from the validation set, not the official held-out test set.

---

## Repository Structure

```text
emotion-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── # FER2013 dataset (not committed)
├── notebooks/
│   └── 01_baseline_cnn.ipynb
├── src/
│   └── # Shared model, dataset, and evaluation modules
├── results/
│   └── # Generated figures and tables (not committed)
└── demo/
    └── # Webcam demonstration (planned)
```

## Setup — Google Colab

Google Colab is the recommended environment for running the experiments.

A local Python installation is not required.

### 1. Prepare Kaggle Credentials

1. Sign in to https://www.kaggle.com/settings/api.
2. Under Legacy API Credentials, create a legacy API key and download `kaggle.json`.
3. Keep the credentials private.
4. Open the FER2013 dataset page and complete any required access prompts.

Never upload `kaggle.json` to GitHub or share your Kaggle API key.

### 2. Open the Notebook

1. Visit https://colab.research.google.com/.
2. Select **Open notebook → GitHub**.
3. Enter the repository URL:

   https://github.com/hsu01/emotion-detection

4. Open `notebooks/01_baseline_cnn.ipynb`.
5. Create a working copy in Google Drive.
6. Select **Runtime → Change runtime type → T4 GPU**.
7. Connect to the runtime.

### 3. Configure Kaggle Access

In the Colab Secrets panel, add:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Use the corresponding values from your Kaggle API credentials.

Enable notebook access for both secrets.

Install Kaggle if necessary:

```python
!pip install -q kaggle
```

### 4. Download and Verify the Dataset

Run the notebook's dataset setup cells.

Expected dataset counts:

```text
Official training images: 28709
Official test images: 7178
Total images: 35887
```

After the stratified split:

```text
Train: 25838
Val: 2871
Test (held out): 7178
```

Expected batch shape:

```text
torch.Size([64, 1, 48, 48])
```

The baseline CNN should produce seven output scores per image.

### 5. Run Experiment A

Run the notebook sections in order:

1. Imports and GPU setup
2. Dataset download and exploration
3. Preprocessing and data loaders
4. CNN model initialization
5. Training
6. Validation evaluation and visualization

For the completed baseline configuration:

```python
NUM_EPOCHS = 15

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/emotion-detection/"
    "checkpoints/A_baseline"
)
```

Mount Google Drive before saving checkpoints.

The best checkpoint is selected based on validation accuracy.

### Optional — Two-Epoch Smoke Test

Before a fresh full training run, a two-epoch smoke test can verify that the pipeline works.

Use a separate checkpoint directory:

```python
CHECKPOINT_DIR = (
    "/content/drive/MyDrive/emotion-detection/"
    "checkpoints/A_smoke"
)
```

Set:

```python
NUM_EPOCHS = 2
```

After the smoke test, recreate the model and optimizer before starting a fresh 15-epoch baseline run.

Do not continue training the smoke-test model as if it were a new baseline.

### Reproducing Evaluation

Load the best checkpoint:

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Evaluate using the validation loader and generate:

- Classification report
- Confusion matrix
- Training curves

Save the generated artifacts to the `A_baseline` Google Drive directory.

Do not evaluate on the official test set until the team's final evaluation protocol is ready.

## Optional Local Setup

Clone the repository:

```bash
git clone https://github.com/hsu01/emotion-detection.git
cd emotion-detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current notebook uses Google Colab-specific paths and authentication. Local execution requires adapting those cells.

## Collaboration Guidelines

- Keep experiment notebooks inside `notebooks/`.
- Use separate Git branches and pull requests for collaborative changes.
- Maintain consistent dataset splits across Experiments A–D.
- Preserve the stratified 90/10 training/validation split with seed 42.
- Apply data augmentation only to training images.
- Keep validation preprocessing consistent across experiments.
- Reserve the official test set for final evaluation.
- Record model configurations, hyperparameters, and results.
- Do not commit datasets, checkpoints, or private credentials.
- Share generated artifacts separately through Google Drive.

## Project Milestones

| Date | Milestone |
|---|---|
| Nov 10–12 | Dataset setup, preprocessing, and baseline CNN (Experiment A) |
| Nov 16 | Pretrained model comparison and data augmentation (Experiments B–D) |
| Nov 23 | Robustness evaluation: lighting, rotation, and occlusion |
| Nov 30 | Webcam demonstration and error analysis |
| Dec 10 | **Final presentation** |
| Dec 14 | **Final report due** |

## Current Progress

### Completed

- [x] Repository structure created
- [x] FER2013 dataset downloaded
- [x] Data preprocessing implemented
- [x] Stratified train/validation split created
- [x] Baseline CNN implemented
- [x] Two-epoch smoke test completed
- [x] Full 15-epoch baseline training completed
- [x] Best model checkpoint saved
- [x] Validation classification report generated
- [x] Confusion matrix generated
- [x] Training curves generated
- [x] Experiment A artifacts saved to Google Drive
- [x] Completed Experiment A notebook committed to GitHub

### Remaining

- [ ] Experiment B — CNN with augmentation
- [ ] Experiment C — ResNet18 without augmentation
- [ ] Experiment D — ResNet18 with augmentation
- [ ] Final test-set evaluation
- [ ] Robustness evaluation under lighting changes
- [ ] Robustness evaluation under rotations
- [ ] Robustness evaluation under occlusions
- [ ] Webcam demonstration
- [ ] Final model comparison
- [ ] Presentation and final report

## Notes

FER2013 is class-imbalanced, with relatively few disgust images and many happy images.

For this reason, evaluation includes per-class precision, recall, F1-score, and confusion matrices in addition to overall accuracy.

Experiment A serves as the baseline for investigating whether augmentation and transfer learning improve classification accuracy and robustness under challenging conditions.

The official FER2013 test set remains held out for final evaluation.
