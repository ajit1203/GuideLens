<<<<<<< HEAD
# Trustworthy Assistive Visual Question Answering for Blind and Low-Vision Users

This project focuses on building a trustworthy assistive Visual Question Answering (VQA) system for blind and low-vision users. The goal is to develop a multimodal system that takes an image and a natural language question as input, predicts an answer, and later extends to provide confidence-aware or safe-response behavior when the image is unclear or the model is uncertain.

## Project Overview

Blind and low-vision users often capture images that are blurry, poorly framed, too dark, or otherwise difficult to interpret. Standard VQA systems may still generate answers even when they are uncertain, which can reduce trust and create risk in assistive settings. This project aims to address that problem by combining visual and textual understanding with a trustworthiness-focused prediction pipeline.

The project currently uses the VizWiz VQA dataset for initial exploration and baseline development.

## Repository Structure

```text
trustworthy-assistive-vqa/
├── checkpoints/              # Saved model weights
├── configs/                  # Configuration files
├── data/
│   ├── processed/            # Processed metadata files
│   └── raw/
│       └── vizwiz/           # Local VizWiz dataset storage
├── docs/                     # Architecture diagrams and UI sketches
├── notebooks/                # Jupyter notebooks for setup and exploration
├── outputs/
│   ├── logs/                 # Training and evaluation logs
│   └── predictions/          # Model prediction outputs
├── results/                  # Exploratory results and visual outputs
├── scripts/                  # Utility and execution scripts
├── src/                      # Source code for data, models, and training
├── ui/                       # Placeholder for future interface
├── environment.yml           # Conda environment file
├── requirements.txt          # Python dependency list
├── README.md                 # Project documentation
└── setup.py                  # Package setup file
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
git clone <your-github-repo-link>
cd trustworthy-assistive-vqa
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

## Current Project Status

At the current stage, the repository includes:

- initial project structure
- local dataset setup
- environment configuration
- an exploratory notebook for dataset verification and understanding

The next step is to build the preprocessing pipeline and generate smaller processed subsets for baseline training.

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
=======
# GuideLens
Trustworthy Assistive Visual Question Answering for Blind and Low-Vision Users
>>>>>>> 4e1b7610a617d03dd7c2d0068b8bde249fc246ae
