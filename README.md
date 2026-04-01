# AI Text Assistant: Engagement Metric Engine ✍️🤖

## Overview
This repository contains the core training pipeline for an AI-driven text analysis tool, designed to function similarly to Grammarly's engagement scoring. The model evaluates written text to predict user engagement levels, utilizing advanced Natural Language Processing (NLP) techniques.

## Model & Architecture
* **Base Model:** `microsoft/deberta-v3-xsmall`
* **Architecture:** Sequence Classification
* **Dataset:** IMDB Dataset (repurposed for binary engagement/sentiment classification)

## Training & Optimization Highlights
This pipeline was engineered with a strong focus on memory efficiency, hardware optimization, and stable convergence during fine-tuning:

* **Hardware Acceleration:** Implemented `bfloat16` (`bf16=True`) precision to maximize training throughput and leverage Tensor Cores on modern RTX GPUs.
* **VRAM Management:** Utilized `DataCollatorWithPadding` for dynamic sequence padding, significantly reducing wasted computation and memory overhead compared to static padding.
* **Effective Batching:** Employed gradient accumulation (`gradient_accumulation_steps=4`) to achieve an effective batch size of 32 without exceeding local VRAM limits.
* **Stable Convergence:** Implemented a cosine learning rate scheduler with a 10% warmup ratio (`warmup_ratio=0.1`) and weight decay (`0.01`) to prevent catastrophic forgetting and ensure a smooth optimization trajectory for the DeBERTa architecture.

## Tech Stack & Libraries
* **Language:** Python
* **Machine Learning:** PyTorch
* **NLP Framework:** Hugging Face (`transformers`, `datasets`)
* **Metrics:** Scikit-learn (`accuracy_score`)
