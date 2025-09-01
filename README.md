# Large Language Model Implementation

A comprehensive implementation of neural network fundamentals through advanced LLM training techniques, demonstrating the complete pipeline from basic backpropagation to preference optimization.

## Architecture Overview

This repository contains five progressive projects that build toward modern LLM capabilities:

**Foundation**: Neural network fundamentals with manual gradient computation
**Core**: Transformer architecture implementation from scratch  
**Alignment**: Self-supervised instruction tuning via backtranslation
**Optimization**: Direct Preference Optimization (DPO) for model alignment
**Application**: Multi-tool research agent with web integration

## Project Details

### 1. Neural Network Fundamentals
Implementation of basic neural networks with manual backpropagation for educational purposes. Covers gradient computation, activation functions (LeakyReLU, Sigmoid), and loss functions (L1, L2).

### 2. GPT Transformer Implementation
Complete transformer architecture built from scratch including:
- Self-attention mechanisms (single and multi-head)
- Positional embeddings and layer normalization  
- Autoregressive text generation with sampling strategies
- Training on TinyShakespeare dataset with character-level tokenization

### 3. Self-Alignment via Instruction Backtranslation
Advanced training technique implementing the "Self-Alignment with Instruction Backtranslation" paper:
- Backward model training for instruction generation
- Self-augmentation using LIMA dataset
- Quality filtering and curation pipeline
- Forward model fine-tuning on synthetic data

### 4. Direct Preference Optimization
Preference-based model alignment using DPO methodology:
- Multi-response generation using LLaMA-2-Chat
- Response ranking with PairRM model
- PEFT-based fine-tuning with 4-bit quantization
- Comparative evaluation framework

### 5. Research Agent System
Multi-tool agent for automated research workflows:
- Topic decomposition and query expansion
- Web search integration (You.com API)
- Content summarization and self-critique
- Iterative improvement cycles

## Technical Stack

**Core**: PyTorch, Transformers, PEFT, TRL  
**APIs**: HuggingFace Hub, Together.AI, You.com  
**Tools**: LLM-Blender, Datasets, Accelerate

## Model Artifacts

All trained models and datasets are available on HuggingFace Hub under `AnushaKulkarni` namespace:

- Instruction generation model: `q1`
- Curated training dataset: `filtered_dataset` 
- Fine-tuned instruction model: `q4`
- Preference dataset: `preferred_dataset2`
- DPO-optimized model: `peft_model_dpo`

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for Projects 2-4)
- 8GB+ RAM
- HuggingFace Hub token
- API keys for external services

## Usage

Each notebook is self-contained and can be run independently. Start with Project 1 for foundational concepts, then progress through the numbered sequence for increasing complexity.

## Research Impact

These implementations demonstrate key techniques behind state-of-the-art language models, providing educational insight into modern AI training methodologies and practical experience with production-scale tools and frameworks.
