# LoRA Fine-Tuning Lab: DistilBERT Sentiment Analysis

This lab demonstrates two different approaches to fine-tuning DistilBERT for sentiment analysis, providing a comprehensive comparison between traditional full fine-tuning and parameter-efficient LoRA methods using the PEFT library.

## 🎯 Lab Objectives

- **Understand the trade-offs** between full fine-tuning and LoRA approaches
- **Compare implementation complexity** of full fine-tuning vs. library-based LoRA
- **Evaluate performance differences** between the two methods
- **Learn practical fine-tuning** techniques for transformer models

## 🏗️ Project Structure

```
lora_ft/
├── src/svlearn_lora_ft/
│   ├── bert_full_ft_trainer.py      # Full fine-tuning (baseline)
│   └── bert_lora_peft_trainer.py   # PEFT library LoRA implementation
├── docs/notebooks/
│   ├── bert_encoder_full_ft.ipynb      # Full fine-tuning evaluation
│   └── bert_encoder_lora_peft.ipynb   # PEFT LoRA evaluation
└── config.yaml                        # Configuration file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync
```

## 📚 Two Fine-Tuning Approaches

### 1. **Full Fine-Tuning** (`bert_full_ft_trainer.py`)
- **What it does**: Updates ALL parameters of DistilBERT (~66M parameters)
- **Pros**: Best performance potential, no architectural modifications
- **Cons**: High memory usage, longer training time, risk of catastrophic forgetting
- **Use case**: When you have sufficient computational resources and want maximum performance

### 2. **PEFT Library LoRA** (`bert_lora_peft_trainer.py`)
- **What it does**: Uses HuggingFace's PEFT library for LoRA implementation
- **Pros**: Clean, production-ready code (~50 lines), automatic parameter management, efficient training
- **Cons**: Less educational value, dependent on external library
- **Use case**: Production deployments, quick prototyping, standard workflows

## 🏃‍♂️ Running the Trainers

### Step 1: Configure Environment

Extract the subject chunks zip from the website

Update `config.yaml` and `.env` file with the correct paths

### Step 2: Run Full Fine-Tuning (Baseline)

```bash
uv run src/svlearn_lora_ft/bert_full_ft_trainer.py
```

**Expected Output:**
- Training progress with all ~66M parameters being updated
- Longer training time compared to LoRA methods
- Larger final model size

### Step 3: Run PEFT Library LoRA

```bash
uv run src/svlearn_lora_ft/bert_lora_peft_trainer.py
```

**Expected Output:**
- Training progress with only ~1-5M parameters being updated
- Fastest training time
- Smallest model size
- Automatic parameter management

## 📊 Evaluation and Analysis

### Using the Evaluation Notebooks

After training each model, use the corresponding notebook to evaluate performance:

#### 1. Full Fine-Tuning Evaluation
```bash
bert_encoder_full_ft.ipynb
```

#### 2. PEFT LoRA Evaluation
```bash
bert_encoder_lora_peft.ipynb
```

### What to Look For

- **Training Time**: Compare training duration between the two methods
- **Memory Usage**: Monitor GPU memory consumption during training
- **Model Performance**: Compare accuracy, precision, recall, and F1 scores
- **Model Size**: Check final saved model sizes
- **Parameter Count**: Verify trainable vs. total parameters

## 🔍 Key Comparisons

| Aspect | Full Fine-Tuning | PEFT LoRA |
|--------|------------------|-----------|
| **Trainable Parameters** | ~66M | ~1-5M |
| **Training Time** | Longest | Fastest |
| **Memory Usage** | Highest | Lowest |
| **Code Complexity** | Low | Low |
| **Performance** | Best | Good |
| **Model Size** | Largest | Smallest |
| **Educational Value** | Low | Medium |

## 🧪 Experimentation Tips

### 1. **Reduced Dataset Testing**
For faster experimentation, uncomment this line in each trainer:
```python
#train_dataset, test_dataset = train_dataset.select(range(100)), test_dataset.select(range(100))
```

### 2. **Hyperparameter Tuning**
- **Learning Rate**: Try different values (1e-5 to 5e-5)
- **LoRA Rank**: Experiment with ranks 4, 8, 16, 32
- **Batch Size**: Adjust based on available memory

### 3. **Monitoring Training**
- Watch GPU memory usage: `watch -n 1 nvidia-smi`
- Monitor training loss and validation metrics
- Check parameter count differences

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Switch to CPU training (slower but more memory-efficient)

2. **Training Too Slow**
   - Use smaller LoRA rank
   - Reduce dataset size for testing
   - Enable mixed precision training

3. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Check data preprocessing

### Getting Help

- Check the detailed comments in each trainer file
- Review the evaluation notebooks for usage examples
- Ensure all dependencies are properly installed

## 🎓 Learning Outcomes

After completing this lab, you should understand:

1. **Parameter-Efficient Fine-Tuning**: How LoRA reduces trainable parameters
2. **Implementation Trade-offs**: Full fine-tuning vs. library-based approaches
3. **Performance vs. Efficiency**: When to use each method
4. **Practical Fine-Tuning**: Real-world implementation considerations
5. **Model Evaluation**: How to assess and compare different approaches

## 🔗 Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

