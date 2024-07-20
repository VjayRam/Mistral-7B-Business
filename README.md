---
base_model: mistralai/Mistral-7B-Instruct-v0.1
datasets:
- theoldmandthesea/17k_business_book
library_name: peft
license: apache-2.0
tags:
- trl
- sft
- generated_from_trainer
model-index:
- name: business_mistral-v1.0
  results: []
language:
- en
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/vjayram-enag-pes-university/huggingface/runs/u8906alp)
# Mistral 7B Fine-Tuned for Business

This model is a fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) on the [theoldmandthesea/17k_business_book](https://huggingface.co/datasets/theoldmandthesea/17k_business_book) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5729

## Model description

- Base Model: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- Quantization: 4 Bit
- Tokenizer: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

## Intended uses & limitations

Mainly intended to be used at the backend of a chatbot application meant to assist children and teenager with knowledge of business and finances.

## Training and evaluation data

- Total number of samples: 17480
- 
- Number of Training Samples: 13984
- Number of Validation / Test Samples: 3496

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- training_steps: 100

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6197        | 0.1869 | 20   | 0.6099          |
| 0.5927        | 0.3738 | 40   | 0.5905          |
| 0.5664        | 0.5607 | 60   | 0.5824          |
| 0.572         | 0.7477 | 80   | 0.5767          |
| 0.5817        | 0.9346 | 100  | 0.5729          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.42.3
- Pytorch 2.1.2
- Datasets 2.20.0
- Tokenizers 0.19.1