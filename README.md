# Lightweight Fine-Tuning with PEFT (LoRA & QLoRA) on AG News

This project explores **parameter-efficient fine-tuning (PEFT)** techniques using Hugging Face’s `peft` library.  
Instead of updating hundreds of millions of parameters in a large model, we adapt only small **LoRA adapters** (and in QLoRA, on a 4-bit quantized backbone).  

The result: **competitive accuracy with <2% of parameters trainable**, making fine-tuning feasible on consumer GPUs like NVIDIA T4.

---

## Project Goal
- Demonstrate how **LoRA** and **QLoRA** can drastically reduce training cost and memory usage.  
- Compare multiple PEFT configurations (different rank `r`, scaling `alpha`, dropout) on the **AG News** dataset.  
- Evaluate improvements against the pre-trained baseline using **Accuracy** and **Macro-F1**.  
- Show how lightweight adapters can be saved, reloaded, and reused for inference.

---

## Dataset
**AG News Topic Classification**  
- 120,000 training samples  
- 7,600 test samples  
- 4 balanced categories: *World, Sports, Business, Sci/Tech*

---

## Methodology
1. **Baseline**: Pre-trained BERT (`bert-base-uncased`) evaluated without fine-tuning.  
2. **LoRA / QLoRA Fine-Tuning**: Insert low-rank adapters into BERT attention layers.  
3. **PEFT Grid Search**: Compare multiple configs:  
   - `r=8, alpha=16, dropout=0.05`  
   - `r=16, alpha=32, dropout=0.10`  
   - `r=32, alpha=32, dropout=0.10`  
4. **Evaluation**: Use Accuracy + Macro-F1 on validation set; test set used only for final reporting.  
5. **Efficiency**: Count trainable vs total parameters to highlight savings.  
6. **Robustness**: If QLoRA (4-bit) is unavailable, fallback to 8-bit or FP32.

---

## Results
| Run                | r  | α  | Dropout | Val Acc | Val F1 | Test Acc | Test F1 | Trainable Params |
|--------------------|----|----|---------|---------|--------|----------|---------|------------------|
| LoRA r=8 α=16 d=0.05 | 8  | 16 | 0.05    | 0.89    | 0.89   | 0.90     | 0.90    | ~1.1M (1.7%)     |
| LoRA r=16 α=32 d=0.10| 16 | 32 | 0.10    | **0.91**| **0.91**| **0.92** | **0.92**| ~2.2M (3.3%)     |
| LoRA r=32 α=32 d=0.10| 32 | 32 | 0.10    | 0.90    | 0.90   | 0.91     | 0.91    | ~4.3M (6.3%)     |

- **Baseline (pre-finetune)**: ~25% accuracy (random guess level)  
- **Best LoRA config (r=16)** achieves >92% accuracy with only **~3% of parameters trainable**.

---

## Tech Stack
- [Transformers](https://huggingface.co/transformers/)  
- [PEFT](https://huggingface.co/docs/peft/index) (LoRA / QLoRA)  
- [Datasets](https://huggingface.co/docs/datasets/)  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (for quantization)  
- PyTorch, Accelerate

---

## How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/lightweight-finetuning.git
cd lightweight-finetuning

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook "Lightweight Fine Tuning.ipynb"
