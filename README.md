# Respiratory Disease Detection from Lung Sounds

A deep learning pipeline that classifies breath cycles from stethoscope recordings into four classes: **normal**, **crackles**, **wheezes**, and **both**. It converts each breath cycle into a Mel spectrogram and fine-tunes a ResNet50V2 pretrained on ImageNet.

The whole project lives in one notebook, [pipeline_local.ipynb](pipeline_local.ipynb), which is saved with its outputs so the plots and results can be read directly on GitHub.

## Dataset

[ICBHI 2017 Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database): 920 recordings from 126 patients, with 6,898 breath cycles annotated by clinicians.

| Class | Breath cycles |
|---|---|
| normal | 3,642 |
| crackles | 1,864 |
| wheezes | 886 |
| both | 506 |

The dataset is not included in this repository. The notebook downloads it (about 4 GB) through `kagglehub` on the first run.

## Pipeline

1. **Parse annotations**: read the start time, end time, and crackle/wheeze flags of every breath cycle.
2. **Filter**: apply a 100–1999 Hz bandpass filter to remove heartbeat and cable noise.
3. **Segment**: cut each cycle, drop those shorter than 0.5 s, and pad or trim to 4 seconds at 4000 Hz.
4. **Extract features**: compute a Mel spectrogram (128 Mel bins, FFT size 1024, hop length 512).
5. **Split**: 60% train / 40% test by patient, so no patient appears in both sets.
6. **Train**: ResNet50V2 with focal loss (gamma 2.0, alpha 0.25) in two phases: first with the base frozen, then with all layers unfrozen at a lower learning rate.
7. **Evaluate**: confusion matrix, classification report, and the ICBHI score.

## Results

Evaluated on 2,304 test cycles from patients the model never saw during training.

| Class | Precision | Recall (sensitivity) | Specificity |
|---|---|---|---|
| normal | 0.61 | 0.73 | 0.38 |
| crackles | 0.34 | 0.32 | 0.78 |
| wheezes | 0.12 | 0.07 | 0.94 |
| both | 0.10 | 0.02 | 0.99 |

- **Accuracy:** 51%
- **ICBHI score:** 52.86% (average of sensitivity and specificity across the four classes)

The model overfits: training accuracy reaches 96% while validation accuracy stays around 50%, and most abnormal cycles are predicted as normal, the majority class. The notebook's conclusion discusses this and lists possible improvements (proper spectrogram resizing, augmentation, class balancing, a lighter model).

## Running it

Requires Python 3.11.

```bash
git clone https://github.com/soukainadaali/RespiratoryDisease.git
cd RespiratoryDisease

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install kagglehub librosa matplotlib pandas numpy tensorflow seaborn scikit-learn scipy ipykernel
```

Then open `pipeline_local.ipynb` in VS Code or Jupyter, select the `.venv` interpreter as the kernel, and run all cells.

What to expect:

- The first run downloads the dataset into the `kagglehub` cache; later runs reuse it.
- The notebook creates an `icbhi_project/` folder next to itself holding the processed cycles, spectrograms, train/test splits, and trained models (about 1.5 GB). It is excluded from git.
- A full run took about 1.5 hours on a laptop CPU, most of it training. A GPU makes it much faster.

## References

- Rocha et al. (2019), "An open access database for the evaluation of respiratory sound classification algorithms". https://doi.org/10.1016/j.pnpbp.2018.09.015
- Gairola et al. (2021), "RespireNet: A Deep Neural Network for Accurately Detecting Abnormal Lung Sounds in Limited Data Setting". https://arxiv.org/abs/2011.00196
- Moummad & Farrugia (2023), "Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning", WASPAA 2023. https://arxiv.org/abs/2210.16192
