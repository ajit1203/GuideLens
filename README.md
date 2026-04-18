# Trustworthy Assistive Visual Question Answering for Blind and Low-Vision Users

This project focuses on building a trustworthy assistive Visual Question Answering (VQA) system for blind and low-vision users. The system takes an image and a natural language question as input and predicts an answer, while also emphasizing cautious behavior when the image is unclear or the question cannot be answered reliably.

The project currently includes:
- a working **baseline multimodal classifier** using **ResNet-18 + DistilBERT**
- a **Streamlit interface** for interactive inference
- an exploratory **Qwen2.5-VL + MLX** evaluation path on Apple Silicon
- training logs, plots, and sample predictions for early evaluation

---

## Project Overview

Blind and low-vision users often capture images that are blurry, poorly framed, too dark, or otherwise difficult to interpret. Standard VQA systems may still generate answers even when they are uncertain, which can reduce trust and create risk in assistive settings.

This project addresses that problem by combining:
- visual understanding
- question understanding
- answer prediction
- answerability-aware behavior

The current dataset for development and evaluation is the **VizWiz VQA** dataset.

---

## Repository Structure

```text
trustworthy-assistive-vqa/
├── checkpoints/                  # Saved model weights and adapter files
├── configs/                      # Configuration files
├── data/
│   ├── processed/                # Processed CSV/JSONL files
│   └── raw/
│       └── vizwiz/               # Local VizWiz dataset storage
├── docs/                         # UI screenshots and project visuals
├── figures/                      # Report figures such as architecture and plots
├── notebooks/
│   ├── setup.ipynb               # Dataset setup and exploratory analysis
│   └── qwen_finetune_eval.ipynb  # Qwen/MLX evaluation notebook
├── results/
│   ├── baseline/
│   │   ├── metrics/              # Baseline metrics
│   │   ├── plots/                # Baseline plots
│   │   └── predictions/          # Baseline sample predictions
│   └── mlx_qwen/
│       ├── logs/                 # MLX/Qwen training logs
│       ├── metrics/              # MLX/Qwen evaluation metrics
│       ├── plots/                # MLX/Qwen loss curves
│       └── predictions/          # MLX/Qwen sample predictions
├── scripts/                      # Utility, training, and evaluation scripts
├── src/
│   ├── data/                     # Dataset code
│   ├── models/                   # Baseline and Qwen model wrappers
│   └── training/                 # Training loop code
├── ui/                           # Streamlit application
├── environment.yml               # Conda environment file
├── requirements.txt              # Python dependency list
├── README.md                     # Project documentation
└── setup.py                      # Package setup file
```
## Dataset Information

This project uses the **VizWiz VQA** dataset.

The dataset contains:
- images captured by blind users
- natural language questions about those images
- crowdsourced answers
- answerability information for many samples

For local development, the dataset should be placed in the following structure:

```text
data/raw/vizwiz/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── train/
├── val/
└── test/
```

## Dataset Download and Usage

This project uses the **VizWiz VQA** dataset, which is designed for visual question answering in assistive settings. The official dataset page provides the train, validation, and test image files along with the annotation files. 

### Where to download the dataset

Download the following files from the official VizWiz VQA page:

- Train images: `train.zip`
- Validation images: `val.zip`
- Test images: `test.zip`
- VQA annotations: `Annotations.zip` 

Official dataset page:  
VizWiz VQA Dataset Page 

Direct download files:
- Train images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip`
- Validation images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip`
- Test images: `https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip`
- Annotations: `https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip`

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ajit1203/GuideLens.git
cd GuideLens
```

### 2. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate assistive_vqa
```

If needed, you can also install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Notebook

To launch Jupyter Notebook:
```bash
jupyter notebook
```
Then open:
```bash
notebooks/setup.ipynb
```

This notebook verifies:

- environment setup
- dataset loading
- annotation inspection
- summary statistics
- sample image visualization

Notebook 2: Qwen / MLX evaluation
```bash
notebooks/qwen_finetune_eval.ipynb
```
This notebook shows:

- processed Qwen-format data inspection
- MLX training log summary
- Qwen training loss visualization
- evaluation metrics
- sample predictions
- qualitative examples

## How to Run the Baseline Pipeline
1. Prepare processed data
```bash
python scripts/prepare_data.py
```
2. Train the baseline model
```bash
python scripts/train_baseline.py
```

This generates:

- baseline metrics
- loss curves
- sample predictions
- saved checkpoints
  
3. Review baseline outputs
Typical outputs are stored under:
```text
results/baseline/
```

## How to Run the Qwen / MLX Path

1. Prepare MLX/Qwen JSONL files
```bash
python scripts/prepare_mlx_qwen_data.py
```
2. Run MLX/Qwen evaluation
```bash
python scripts/eval_mlx_qwen.py
```
Typical outputs are stored under:
```text
results/mlx_qwen/
```
This includes:

- training logs
- evaluation metrics
- sample predictions
- training loss plots

## How to Launch the Interface
Run the Streamlit app:
```bash
python -m streamlit run ui/app.py
```
The interface currently supports:

- image upload
- question input
- answer display
- cautious response behavior for uncertain cases

## Current Results
### Baseline model

The baseline system uses:

- ResNet-18 for image encoding
- DistilBERT for question encoding
- concatenation-based fusion
- dual heads for:
  - answer prediction
  - answerability prediction

Current baseline results from the main run:

- training subset: 1000
- validation subset: 200
- epochs: 8
- final validation answer accuracy: 0.6617
- final validation answerability accuracy: 0.7883

### Qwen / MLX path

The project also includes an exploratory generative model direction using Qwen2.5-VL + MLX on Apple Silicon.

Current status:

- MLX training pipeline runs successfully
- training loss curves are available
- preliminary base-Qwen evaluation has been completed
- fine-tuned adapter remains under refinement

## Current Project Status

At the current stage, the repository includes:

- dataset setup
- preprocessing pipeline
- baseline training pipeline
- evaluation outputs
- Streamlit interface
- exploratory Qwen / MLX workflow
- report figures and visual evidence

Next steps include:

- refining answerability calibration
- improving failure analysis
- polishing the interface
- improving the generative Qwen-based path

## Planned System Components

The planned system will include:

- a vision encoder for image representation
- a text encoder for question understanding
- a fusion module for multimodal learning
- an answer prediction head
- a future confidence or answerability component
- a user-facing interface for interactive inference

## Author

**Ajit Reddy Lingannagaru**

GitHub: https://github.com/ajit1203  
LinkedIn: https://www.linkedin.com/in/ajitreddylingannagaru/
