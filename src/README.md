# src/

This folder will hold the Python modules shared by all four experiments,
so the notebooks stay thin and every experiment uses the same pipeline:
model definitions (the baseline CNN and the pretrained ResNet18), dataset
loading and transforms (including the augmentation and robustness
distortions), the training loop, and evaluation (classification report,
confusion matrix, per-class metrics). Nothing here yet — the baseline
notebook (`notebooks/01_baseline_cnn.ipynb`) currently carries the code
inline, and it will be factored into this package as Experiments B, C,
and D are built.
