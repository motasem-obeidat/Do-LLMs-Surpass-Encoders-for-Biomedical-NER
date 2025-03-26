# Do LLMs Surpass Encoders for Biomedical NER?
This repository contains the official codebase and dataset for our paper, **"Do LLMs Surpass Encoders for Biomedical NER?"**, accepted at the **IEEE International Conference on Healthcare Informatics (ICHI) 2025**.

In this repository, we provide scripts for training and evaluating both **encoder-based** and **decoder-based** Named Entity Recognition (NER) models. The **encoder model** leverages transformer-based architectures for token classification, while the **decoder model** is designed for autoregressive generation tasks.

The evaluation is conducted using the International Workshop on Semantic Evaluation (SemEval) framework. For more details on named entity evaluation, you can refer to this [blog post by David Batista](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/).

---

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training & Evaluating the Encoder Model](#training--evaluating-the-encoder-model)
  - [Training & Evaluating the Decoder Model](#training--evaluating-the-decoder-model)

---

## Project Overview
This project implements Named Entity Recognition (NER) using two different modeling approaches:

- Encoder-based Models: Uses transformer-based token classification (e.g., BERT) for sequence labeling.
- Decoder-based Models: Uses autoregressive transformer models (e.g., Mistral-7B) for sequence generation.

The models can be trained on various datasets, such as:
- JNLPBA
- BioRED
- ChemProt
- BC5CDR
- Reddit_Impacts

**Notes:** 
- The ChemProt dataset is sourced from [End-to-End Models for Chemical–Protein Interaction Extraction](https://github.com/bionlproc/end-to-end-ChemProt).
- The Reddit-Impacts dataset has not been publicly released. To access it, please contact the authors of the paper: [Reddit-Impacts: A Named Entity Recognition Dataset for Analyzing Clinical and Social Effects of Substance Use Derived from Social Media](https://arxiv.org/pdf/2405.06145).
---

## File Structure
```
.
├── datasets/
│   ├── BC5CDR/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │
│   ├── BioRED/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │
│   ├── JNLPBA/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │
│   ├── ChemProt/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │
│   ├── Reddit_Impacts/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│
├── encoders/
│   ├── data_preprocessing.py       # Prepares datasets for encoder model
│   ├── model_prepare.py            # Prepares encoder model for training
│   ├── metrics.py                  # Evaluation metrics for encoder model
│   ├── ner_SemEval.py              # SemEval evaluation
│   ├── run_encoder_train.py        # Trains the encoder model
│   ├── run_encoder_inference.py    # Runs inference on trained encoder model
│   ├── inference.py                # Core inference script
│   ├── main.py                     # Main script for encoder training/evaluation
│
├── decoders/
│   ├── data_preprocessing.py       # Prepares datasets for decoder model
│   ├── model_prepare.py            # Prepares decoder model for training
│   ├── metrics.py                  # Evaluation metrics for decoder model
│   ├── ner_SemEval.py              # SemEval evaluation
│   ├── run_decoder_train.py        # Trains the decoder model
│   ├── run_decoder_inference.py    # Runs inference on trained decoder model
│   ├── inference.py                # Core inference script
│   ├── main.py                     # Main script for decoder training/evaluation
│
├── encoder_requirements.txt       # Required dependencies for encoder model
├── decoders_requirements.txt       # Required dependencies for decoder model
└── README.md                       # Documentation
```
---

## Installation
To install the necessary dependencies for the encoder model, run:

```bash
pip install -r encoder_requirements.txt
```

To install the necessary dependencies for the decoder model, run:

```bash
pip install -r decoder_requirements.txt
```

---

## Usage
### Training & Evaluating the Encoder Model
To train the encoder model, use:

```bash
python run_encoder_train.py
```

This script executes the following command internally:

```bash
python /src/main.py \
    --output_dir /results/ \
    --datasets_path /datasets/ \
    --num_train_epochs 20 \
    --model_name google-bert/bert-large-uncased \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --dataset_name JNLPBA
```

To perform inference using the trained encoder model, run:

```bash
python run_encoder_inference.py
```

Which executes:

```bash
python /src/inference.py \
    --model_dir /results/ \
    --datasets_path /datasets/ \
    --model_name google-bert/bert-large-uncased \
    --dataset_name JNLPBA
```

### Training & Evaluating the Decoder Model
To train the decoder model, run:
```bash
python run_decoder_train.py
```

This will execute the following command:

```bash
python /src/main.py \
    --output_dir /results/ \
    --datasets_path /datasets/ \
    --num_train_epochs 20 \
    --learning_rate 4e-5 \
    --optim paged_adamw_8bit \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --generation_batch_size 16 \
    --max_new_tokens 3000 \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --isQLoRA True \
    --dataset_name JNLPBA \
    --hf_token <YOUR_HF_TOKEN>
```

To generate predictions using the decoder model, run:

```bash
python run_decoder_inference.py
```

This will internally execute:

```bash
python /src/inference.py \
    --model_dir /results/ \
    --datasets_path /datasets/ \
    --generation_batch_size 16 \
    --max_new_tokens 3000 \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --isQLoRA True \
    --dataset_name JNLPBA \
    --trained_model_checkpoint_number <CHECKPOINT_NUMBER> \
    --hf_token <YOUR_HF_TOKEN>
```

**Notes:**
- `--hf_token <YOUR_HF_TOKEN>`: Replace `<YOUR_HF_TOKEN>` with your Hugging Face API token to access pre-trained models.
- `--trained_model_checkpoint_number <checkpoint_number>`: This specifies the checkpoint number of the trained decoder model. Adjust this to the correct checkpoint number.

