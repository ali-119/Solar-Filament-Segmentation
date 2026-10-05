# Solar Filament Segmentation
A **deep learning semantic segmentation project** for detecting **solar filaments** in full-disk H-alpha images of the Sun, built with **TensorFlow / Keras**. The goal is to predict a pixel-wise binary mask marking filaments, comparing a U-Net trained from scratch against **EfficientNetB0 transfer-learning** variants before selecting a final model.

<p align="center">
  <img src="https://img.shields.io/badge/ComputerVision-Segmentation-green">
  <img src="https://img.shields.io/badge/Models-3%20Compared-blue">
  <img src="https://img.shields.io/badge/Dataset-MAGFiLO-purple">
  <img src="https://img.shields.io/badge/Language-Python-yellow">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-red">
</p>

------

# Overview
Solar filaments are dark, elongated structures of cool plasma suspended in the solar atmosphere. They are closely linked to solar eruptions, so detecting them automatically is useful for space-weather research.

The workflow progresses from raw COCO annotations to a tuned transfer-learning U-Net:
- Explore the dataset and its COCO-style annotations
- Convert polygon annotations into binary masks on the fly
- Train a **baseline U-Net from scratch**
- Train a U-Net with a **frozen pretrained EfficientNetB0 encoder**
- **Fine-tune** the top encoder layers and tune loss weights and the prediction threshold
- Evaluate with the **Dice coefficient** and compare all approaches

------

# Dataset
- **Name:** MAGFiLO 1.0 (Kaggle 2026 edition)
- **Source:** Kaggle competition `filament-segmentation-2026`
- **Images:** full-disk H-alpha solar images, **2048 × 2048** px (707 unique image files in the training folder)
- **Annotations:** COCO-format JSON (`images`, `annotations`, `categories`), with a polygon `segmentation`, `bbox`, `area` and a `spine` line for every filament
- **Target:** binary mask (1 = filament, 0 = background)

## Data Characteristics
- Extremely **imbalanced classes**: filaments cover only a tiny fraction of each image, so plain pixel accuracy is misleading (a model that predicts "all background" already scores above 99%)
- Thin, elongated, irregular shapes with a variable number of filaments per image
- Some frames appear more than once in the annotation file under different image IDs

> **Note:** The dataset is **not included** in this repository. It is downloaded automatically through `kagglehub` (you must accept the competition rules on Kaggle first).

<!-- 📷 FIGURE: sample solar image next to its ground-truth mask -->
<p align="center">
  <img width="766" height="790" alt="download" src="https://github.com/user-attachments/assets/bea78670-48d7-4cda-80c3-396f096a7643" />
</p>

------


# Project Workflow

## 1) Data Analysis
`Data Analysis/Data_file.ipynb`
- Downloaded the data and inspected the image folder, image sizes and COCO JSON structure
- Converted `images` and `annotations` into DataFrames to inspect their columns
- Counted filaments per image and visualized sample images

## 2) Data Pipeline
Shared across all notebooks and `Final Model.py`
- `generate_mask_for_image()` rasterizes the polygon annotations into a binary mask with `cv2.fillPoly`
- A custom Keras `Sequence` (`FilamentDataGenerator`) reads, resizes and batches data **on the fly**:
  - Images resized to **512 × 512** (bilinear), masks resized with **nearest neighbour** to keep them binary
  - Batch size **8**
- **75 / 25** train / validation split (`random_state=19`)
- Images are fed as raw 0–255 values, since the Keras EfficientNet already includes its own rescaling layer (dividing by 255 manually was tried first)

## 3) Models Compared
`models/*.ipynb`

| Notebook | Architecture | Encoder | Loss |
|---|---|---|---|
| `Base CNN Model.ipynb` | U-Net from scratch (~1.37 M params) | Trained from scratch | BCE + Dice (1 : 1) |
| `Transform Model.ipynb` | U-Net with EfficientNetB0 encoder | ImageNet weights, **fully frozen** | BCE + Dice (1 : 1) |
| `Fine_Tuning Model.ipynb` | U-Net with EfficientNetB0 encoder | ImageNet weights, **last 40 layers unfrozen** | 0.3 · BCE + 0.7 · Dice |

## 4) Final Model
`Final Model.py`
- **Encoder:** EfficientNetB0 (ImageNet), only the last 40 layers trainable
- **Skip connections:** `block2a`, `block3a`, `block4a` and `block6a` expand activations
- **Decoder:** 3 up-sampling blocks (`Conv2DTranspose` → skip concatenation → 2 × `Conv2D` + `BatchNormalization`) with 256 / 128 / 64 filters, followed by a final up-sampling to full resolution and a 1 × 1 sigmoid output
- **Loss:** `0.3 · BCE + 0.7 · (1 − Dice)`
- **Optimizer:** Adam, `learning_rate = 1e-4`
- **Callbacks:** `ReduceLROnPlateau` and `ModelCheckpoint`, both monitoring `val_dice_coef`
- **Training:** 20 epochs; the best checkpoint is saved as `best_filament_model.keras`
- **Inference:** sigmoid output thresholded at **0.45**

------

# Project Structure
```
Solar-Filament-Segmentation/
├── README.md
├── LICENSE
├── images/
│   ├── sample_image_and_mask.png
│   ├── architecture.png
│   ├── loss_curve.png
│   ├── dice_curve.png
│   └── predictions.png
├── Data Analysis/
│   └── Data_file.ipynb          # Dataset exploration
├── models/
│   ├── Base CNN Model.ipynb     # U-Net from scratch
│   ├── Transform Model.ipynb    # Frozen EfficientNetB0 encoder
│   └── Fine_Tuning Model.ipynb  # Partially unfrozen encoder
└── Final Model.py               # Final training + evaluation script
```

------

# How to Run

## 1) Clone the repository
```bash
git clone https://github.com/ali-119/Solar-Filament-Segmentation.git
cd Solar-Filament-Segmentation
```

## 2) Install dependencies
```bash
pip install tensorflow opencv-python numpy pandas matplotlib plotly scikit-learn kagglehub
```
A GPU is strongly recommended. The notebooks were run on Google Colab.

## 3) Get the data
1. Join the [`filament-segmentation-2026`](https://www.kaggle.com/competitions/filament-segmentation-2026) competition on Kaggle and accept its rules.
2. Add your Kaggle API credentials when `kagglehub.login()` prompts for them.
3. If your download location differs, update `BASE_DIR` at the top of `Final Model.py` and the notebooks. The default is `/root/.cache/kagglehub/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`.

## 4) Explore the data
```bash
jupyter notebook "Data Analysis/Data_file.ipynb"
```

## 5) Compare the models
Run any notebook inside `models/` to reproduce that experiment.

## 6) Train & evaluate the final model
```bash
python "Final Model.py"
```
This builds the EfficientNetB0 U-Net, trains it for 20 epochs, plots the loss and Dice curves, and prints the final Dice score on a random validation sample.

------

# Results Summary

| Model | Best Val Dice (Keras metric) | Dice @ threshold 0.45 | Epochs |
|---|---|---|---|
| U-Net from scratch | 0.627 | not measured | 10 |
| EfficientNetB0 (frozen encoder) | 0.659 | 66.48 (10 val. images) | 20 |
| **EfficientNetB0 (fine-tuned) — Final** | **0.662** | **68.34 (200 val. images)** | 20 |

> The thresholded Dice scores were computed on random validation subsets of different sizes, so they are indicative rather than strictly comparable. The Keras `val_dice_coef` column uses the full validation set and is the fairer comparison.

<!-- 📷 FIGURE: training curves -->
<p align="center">
  <img width="1853" height="525" alt="newplot" src="https://github.com/user-attachments/assets/07052662-7c39-4725-8d47-1b36273135bc" />
</p>
<p align="center">
  <img width="1853" height="525" alt="newplot (1)" src="https://github.com/user-attachments/assets/4f8b94f7-d02a-4a56-a5e5-9c6696ae9611" />
</p>


## Different Settings, Different Results
Many configurations were tried during development, and the results varied noticeably between them. The most important variations were:
- **Training from scratch vs. transfer learning:** pretrained ImageNet features gave a clear boost and much faster convergence
- **Frozen vs. fine-tuned encoder:** unfreezing the last 40 encoder layers gave the best validation loss and Dice
- **Loss weighting:** a plain `BCE + Dice` loss was compared with `0.3 · BCE + 0.7 · Dice`, which emphasizes overlap on such a sparse target
- **Learning rate:** a lower learning rate caused underfitting; `1e-4` with `ReduceLROnPlateau` worked best
- **Input scaling:** raw 0–255 inputs (the EfficientNet default) were compared with 0–1 normalization
- **Prediction threshold:** several thresholds were tested and **0.45** gave the best Dice

The final model is the configuration in `Final Model.py`.

------

# Key Takeaways
- Pretrained encoders helped a lot on a small, highly imbalanced segmentation dataset
- A Dice-weighted loss is essential here, since plain accuracy is dominated by the background class
- Fine-tuning only the top encoder layers gave the best balance between adapting to solar images and keeping the pretrained features
- Training Dice kept rising to about 0.78 while validation Dice plateaued around 0.66, so the models show some **overfitting**
- Tuning the output threshold gave a small but real gain at inference time

------

# Limitations
- The annotation file lists some frames more than once under different IDs, and the split is made per entry, so the same frame can appear in both train and validation. This may make validation scores optimistic
- Images were downscaled from 2048 × 2048 to 512 × 512, which can erase thin filaments
- No data augmentation was used
- Thresholded Dice was measured on random validation subsets of different sizes, and the from-scratch U-Net has no thresholded score
- No separate held-out test evaluation is documented, only validation results
- Only EfficientNetB0 was explored as an encoder

------

# Future Work
- Group-aware train/validation splitting by image file
- Data augmentation (flips, rotations, brightness) and higher input resolution or patch-based training
- Other encoders (ResNet, EfficientNetV2) and attention-based or transformer-based decoders
- Post-processing of masks (e.g. removing small blobs)

------

# Libraries Used
- `tensorflow` / `keras`
- `opencv-python`
- `numpy` / `pandas`
- `scikit-learn`
- `matplotlib` / `plotly`
- `kagglehub`

------

# License
This project is released under the [MIT License](LICENSE).

------

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)
