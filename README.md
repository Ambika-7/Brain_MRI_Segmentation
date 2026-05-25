# 🧠 Brain MRI Tumor Segmentation

An end-to-end, lightweight, attention-enhanced deep learning framework for multi-class Brain MRI segmentation using the **BraTS (Brain Tumor Segmentation)** dataset. 

This project features a custom **Attention U-Net** architecture (with a MobileNetV2 encoder, ASPP bottleneck, and channel-wise attention gates) and includes a professional, real-time medical imaging dashboard built with **Streamlit**.

---

## 🚀 How to Run the Project

Follow these steps to set up, preprocess, train, and run the project locally on your machine.

### 1. Installation & Environment Setup
Open your terminal inside the project directory (`Brain-MRI-Segmentation-main`) and activate the pre-configured virtual environment:

#### On Windows Command Prompt (cmd):
```cmd
venv\Scripts\activate
```

#### On PowerShell:
```powershell
.\venv\Scripts\activate
```

#### On Linux / macOS:
```bash
source venv/bin/activate
```

---

### 2. Preprocess the Dataset
The raw 3D NIfTI scans must be converted into normalized 2D NumPy slices (`.npy` files) for training and inference. 

#### Option A: Quick Run / Testing (Recommended first step)
To test the project quickly without waiting hours, preprocess a sample of the first 5 patients:
```bash
python scripts/preprocess_sample.py
```
This script automatically copies 5 patient folders to `data/raw_sample/` and outputs the preprocessed slices to `data/processed_sample/` within seconds.

#### Option B: Full Preprocessing
To preprocess the entire dataset of 336 patients:
```bash
python scripts/preprocess_local.py --input_dir data/BraTS2021_Training_Data --output_dir data/processed --batch_size 50
```

---

### 3. Train the Model
You can train a model locally. Since the virtual environment uses CPU-only PyTorch by default, use the following commands:

#### Option A: Fast CPU Training (on the 5-patient sample dataset)
This will take **under 5 minutes** and produce a model checkpoint file so you can run the Streamlit app:
```bash
python scripts/train_local.py --data_dir data/processed_sample --output_dir outputs/ablation/full --epochs 2 --batch_size 8 --device cpu --num_workers 0 --model_name ablation_full
```

#### Option B: Full Local Training
```bash
python scripts/train_local.py --data_dir data/processed --output_dir outputs/ablation/full --epochs 50 --batch_size 16 --device cpu --num_workers 0 --model_name ablation_full
```
*(If you have an NVIDIA GPU set up on your machine, you can change `--device cpu` to `--device gpu`)*

---

### 4. Launch the Streamlit Web Dashboard
Launch the interactive web-based visualization dashboard using the command corresponding to your terminal:

#### On Windows Command Prompt (cmd):
```cmd
# For the quick sample run:
set BRAIN_MRI_DATA_DIR=data/processed_sample
streamlit run app/frontend.py
```

#### On PowerShell:
```powershell
# For the quick sample run:
$env:BRAIN_MRI_DATA_DIR="data/processed_sample"
streamlit run app/frontend.py
```

Streamlit will print a URL (usually `http://localhost:8501`). Open it in your web browser to upload slices, visualize ground-truth vs. predicted tumor masks, and navigate slice by slice.

---

## 📂 Project Structure

```
Brain-MRI-Segmentation/
│
├── 📁 app/
│   ├── frontend.py                 # Streamlit web application (Medical Dashboard)
│   └── styles.css                  # Custom styling for dashboard
│
├── 📁 models/
│   ├── architecture.py             # AttentionUNet model definition (MobileNetV2 + ASPP)
│   ├── losses.py                   # Combined Dice + CE loss
│   ├── metrics.py                  # SegmentationMetrics (Dice, IoU, recall)
│   └── model_registry.py           # Registry mapping model keys to classes
│
├── 📁 scripts/
│   ├── preprocess_local.py         # Batch preprocessing script
│   ├── preprocess_sample.py        # Helper to preprocess a 5-patient sample
│   ├── train_local.py              # Main training script (CPU/GPU support)
│   └── validate_pipeline.py        # Pipeline shape and metric verification script
│
├── 📁 utils/
│   ├── dataset_loader.py           # PyTorch Dataset & DataLoader classes
│   ├── transforms.py               # Geometric and intensity augmentations
│   └── visualization.py            # Result visualization utilities
│
└── requirements.txt                # Project dependencies
```

---

## 🧠 Model Architecture & Methodology

Our segmentation model, **Lightweight Attention-Enhanced U-Net**, combines three key components to balance high accuracy with low parameter overhead:

1. **MobileNetV2 Encoder**: Serves as a lightweight feature extractor, reducing parameter count compared to classic ResNet or VGG encoders.
2. **ASPP Bottleneck**: Atrous Spatial Pyramid Pooling extracts multi-scale context without increasing resolution, which is vital for varying brain tumor sizes.
3. **Attention Gates**: Integrated into skip-connections, they filter out irrelevant features and focus gradients on the tumor regions.

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| Encoder (MobileNetV2) | 2,224,160 | 65.0% |
| Decoder | 760,584 | 22.2% |
| Bottleneck (LightASPP) | 415,104 | 12.1% |
| Attention Gates | 23,332 | 0.7% |
| Output Layer | 68 | 0.0% |
| **Total Model** | **3.42M** | **100%** |

---

## 📊 Core Features

* **Multi-Modal Integration**: Combines four standard MRI sequences (**T1**, **T1CE**, **T2**, and **FLAIR**) into a 4-channel input.
* **4-Class Semantic Labels**: Segments scans into:
  - Background
  - Necrotic Tumor Core
  - Peritumoral Edema
  - Enhancing Tumor
* **Class Imbalance Resolution**: Built-in weighted losses (Dice + CrossEntropy) to prevent the model from ignoring minority tumor classes in favor of background tissue.
* **Windows & CPU Friendly**: Configured with compatibility fixes for Windows console encoding and optimized CPU memory allocation.