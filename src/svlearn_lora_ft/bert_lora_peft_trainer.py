#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#

"""
DistilBERT Fine-tuning with LoRA using HuggingFace PEFT Library
(Custom Dataset: History / Physics / Biology)

This module demonstrates LoRA fine-tuning for DistilBERT using HuggingFace's Parameter-Efficient
Fine-Tuning (PEFT) library, applied to a custom subject classification dataset made of 3 JSON files.

KEY DIFFERENCES FROM FULL FINE-TUNING (bert_full_ft_trainer_subjects.py):

1. **Parameter Efficiency**: Updates only ~1-5M parameters instead of all ~66M parameters
   of the base DistilBERT model, significantly reducing memory usage and training time.

2. **LoRA Architecture**: Uses Low-Rank Adaptation to add trainable rank decomposition
   matrices to specific attention layers, allowing efficient fine-tuning.

3. **Automatic Parameter Management**: PEFT handles which parameters to freeze and which
   to make trainable, eliminating the need for manual parameter management.

4. **Better Memory Efficiency**: Significantly lower GPU memory requirements compared
   to full fine-tuning, making it suitable for resource-constrained environments.

Advantages of PEFT LoRA over Full Fine-Tuning:
- Much faster training time
- Lower memory usage
- Smaller final model size
- Reduced risk of catastrophic forgetting
- Easier to deploy and maintain

Usage:
    python bert_lora_peft_trainer_subjects.py
"""

# Standard library imports
import os
import time
import random
import pandas as pd

# Third-party imports
import numpy as np
import torch
from datasets import Dataset
from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

# PEFT (Parameter-Efficient Fine-Tuning) imports
from peft import LoraConfig, TaskType, PeftModel

# Local imports
from svlearn_lora_ft import config
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# =============================================================================
# REPRODUCIBILITY SETUP
# =============================================================================
seed = 100                               
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =============================================================================
# DATASET FUNCTIONS
# =============================================================================

def get_subject_dataset(history_file, physics_file, biology_file):
    """
    Load and prepare the subject classification dataset (History / Physics / Biology).

    Reads three JSON files, merges them into a single dataset, maps textual labels
    ("Biology", "Physics", "History") to numeric IDs, and splits into training
    and test sets.

    Args:
        history_file (str): Path to History JSON file.
        physics_file (str): Path to Physics JSON file.
        biology_file (str): Path to Biology JSON file.

    Returns:
        tuple:
            - train_dataset (Dataset): Hugging Face Dataset split for training.
            - test_dataset (Dataset): Hugging Face Dataset split for evaluation.
            - label2id (dict): Mapping of labels to IDs (e.g., {"Biology":0, "Physics":1, "History":2}).
    """
    df_bio = pd.read_json(biology_file, lines=True)
    df_phy = pd.read_json(physics_file, lines=True)
    df_hist = pd.read_json(history_file, lines=True)

    df = pd.concat([df_bio, df_phy, df_hist], ignore_index=True)

    label2id = {"Biology": 0, "Physics": 1, "History": 2}
    df["label"] = df["label"].map(label2id)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    return dataset["train"], dataset["test"], label2id


def prepare_dataset(tokenizer, train_dataset, test_dataset):  
    """
    Tokenize and format the subject dataset for model input.

    Applies the tokenizer to raw text, renames the "label" column to "labels"
    (required by Hugging Face Trainer), and sets the dataset format to PyTorch
    tensors for training and evaluation.

    Args:
        tokenizer: DistilBERT tokenizer for preprocessing text.
        train_dataset (Dataset): Training split.
        test_dataset (Dataset): Test split.

    Returns:
        tuple:
            - tokenized_train (Dataset): Tokenized training set (PyTorch ready).
            - tokenized_test (Dataset): Tokenized test set (PyTorch ready).
    """
    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_test


def compute_metrics(pred):
    """
    Compute evaluation metrics (Accuracy, Precision, Recall, F1).

    Converts raw model predictions (logits) into class predictions and compares
    them against the true labels. Metrics are weighted across classes to handle
    imbalance.

    Args:
        pred (EvalPrediction): Contains model logits and true labels.

    Returns:
        dict: Dictionary with "accuracy", "precision", "recall", "f1".
    """
    logits, labels = pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ =="__main__":
    """
    Main training script for DistilBERT with LoRA fine-tuning using PEFT
    on the custom subject dataset (History, Physics, Biology).

    Steps performed:
    1. Detects available device (CUDA, MPS, or CPU).
    2. Loads subject dataset from JSON files and prepares train/test splits.
    3. Initializes base DistilBERT model, tokenizer, and collator.
    4. Configures and applies LoRA using Hugging Face PEFT.
    5. Defines Hugging Face TrainingArguments.
    6. Sets up the Trainer and runs training.
    7. Saves best model checkpoint.
    8. Reloads best model and runs final evaluation.
    9. Prints summary of PEFT advantages over full fine-tuning.
    """

    # =============================================================================
    # ENVIRONMENT AND DEVICE SETUP
    # =============================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  
    
    # Output directory for model checkpoints
    output_dir = config["final-ft-model-paths"]["subject_model_lora_peft"]
    os.makedirs(output_dir, exist_ok=True)
    
    # =============================================================================
    # MODEL AND DATASET SETUP
    # =============================================================================
    # Dataset paths
    history_file = config["dataset-paths"]["history_file"]
    physics_file = config["dataset-paths"]["physics_file"]
    biology_file = config["dataset-paths"]["biology_file"]
    
    # Load dataset
    train_dataset, test_dataset, label2id = get_subject_dataset(history_file, physics_file, biology_file)

    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {v: k for k, v in label2id.items()}
    base_model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    tokenized_train, tokenized_test = prepare_dataset(tokenizer, train_dataset, test_dataset)

    # =============================================================================
    # LoRA CONFIGURATION USING PEFT
    # =============================================================================
    use_cpu, no_cuda = False, False
    if device == "cpu":
        use_cpu, no_cuda = True, True

    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        target_modules=["attention.q_lin", "attention.v_lin"],  # Target layers in DistilBERT
        bias="none",  # Biases remain frozen
        task_type=TaskType.SEQ_CLS,  # Task type: sequence classification
        modules_to_save=["classifier", "pre_classifier"],  # Keep classifier layers trainable
    )

    # Wrap model with PEFT
    peft_model = PeftModel(base_model, lora_config)
    peft_model.print_trainable_parameters()
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    training_args = TrainingArguments(
        output_dir=output_dir,
        no_cuda=no_cuda,
        use_cpu=use_cpu,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        label_names=["labels"],
        seed=seed,
    )

    # =============================================================================
    # TRAINER SETUP AND TRAINING
    # =============================================================================
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    print("Starting training with PEFT LoRA for subject classification...")
    trainer.train()
    end_time = time.time()

    # =============================================================================
    # MODEL SAVING AND EVALUATION
    # =============================================================================
    trainer.model.save_pretrained(f'{output_dir}/best_model')
    print(f"Best PEFT model saved to: {output_dir}/best_model")

    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training started at: {time.ctime(start_time)}")
    print(f"Training ended at: {time.ctime(end_time)}")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")

    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================
    print("Loading best PEFT model for final evaluation...")
    model1 = PeftModel.from_pretrained(base_model, f'{output_dir}/best_model')

    trainer1 = Trainer(
        model=model1,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Evaluating final PEFT model...")
    final_results = trainer1.evaluate()
    print("Final evaluation results:")
    print(final_results)
    
    # =============================================================================
    # SUMMARY OF PEFT ADVANTAGES OVER FULL FINE-TUNING
    # =============================================================================
    print("\n" + "="*80)
    print("PEFT IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Key advantages of using PEFT LoRA over full fine-tuning:")
    print("1. Parameter efficiency: ~1-5M trainable vs ~66M trainable parameters")
    print("2. Faster training: Significantly reduced training time")
    print("3. Lower memory usage: Reduced GPU memory requirements")
    print("4. Smaller model size: Only LoRA weights need to be saved")
    print("5. Reduced overfitting: Lower risk of catastrophic forgetting")
    print("6. Easier deployment: Smaller model files for production")
    print("="*80)
