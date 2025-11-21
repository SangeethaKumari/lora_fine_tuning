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
DistilBERT Full Fine-Tuning Implementation for Subject Classification (History / Physics / Biology)

This module implements traditional full fine-tuning for DistilBERT, where ALL parameters
of the pre-trained model are updated during training. It uses your custom dataset
made up of 3 JSON files (History, Physics, Biology).

Dataset Format:
Each JSON should contain entries like:
{
    "label": "Biology",
    "title": "PrinciplesOfBiochemistry.pdf",
    "chunk_index": 0,
    "text": "Some text chunk ..."
}

Comparison with LoRA:
- Full FT: all ~66M parameters updated
- LoRA: only a few million updated
"""

# Standard library imports
import os
import time
import random
import pandas as pd

# Third-party imports
import numpy as np
import torch
from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import TrainingArguments, Trainer

# Local imports
from svlearn_lora_ft import config

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

    This function loads three JSON files containing labeled text chunks for each subject,
    merges them into a single dataset, maps string labels to integer IDs, and splits
    the dataset into training and test sets.

    Args:
        history_file (str): Path to History JSON file.
        physics_file (str): Path to Physics JSON file.
        biology_file (str): Path to Biology JSON file.

    Returns:
        tuple:
            - train_dataset (Dataset): Training split as Hugging Face Dataset.
            - test_dataset (Dataset): Test split as Hugging Face Dataset.
            - label2id (dict): Mapping of class labels to integer IDs.
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


def prepare_dataset(tokenizer, train_dataset: Dataset, test_dataset: Dataset):  
    """
    Tokenize and format the subject datasets for model training.

    This function applies the DistilBERT tokenizer to the raw text data, renames the
    label column to match Trainer requirements, and sets the dataset format for use
    with PyTorch.

    Args:
        tokenizer: DistilBERT tokenizer.
        train_dataset (Dataset): Training split of the dataset.
        test_dataset (Dataset): Test split of the dataset.

    Returns:
        tuple:
            - tokenized_train (Dataset): Tokenized training dataset with PyTorch tensors.
            - tokenized_test (Dataset): Tokenized test dataset with PyTorch tensors.
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

# =============================================================================
# MODEL / METRICS
# =============================================================================

def get_model_tokenizer_collator(num_labels, label2id):
    """
    Initialize DistilBERT model, tokenizer, and data collator for classification.

    Loads the pretrained DistilBERT base model with a sequence classification head
    for the given number of labels. Also prepares the tokenizer and data collator
    for padding batches dynamically during training.

    Args:
        num_labels (int): Number of output classes.
        label2id (dict): Mapping of labels to integer IDs.

    Returns:
        tuple:
            - model (DistilBertForSequenceClassification): DistilBERT classification model.
            - tokenizer (DistilBertTokenizer): Tokenizer for text preprocessing.
            - data_collator (DataCollatorWithPadding): Batch collator for dynamic padding.
    """
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {v: k for k, v in label2id.items()}

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    return model, tokenizer, data_collator


def compute_metrics(pred):
    """
    Compute evaluation metrics for classification.

    Uses predictions and true labels to compute standard evaluation metrics:
    accuracy, precision, recall, and F1 score. Metrics are computed in a
    weighted fashion to account for class imbalance.

    Args:
        pred (EvalPrediction): A named tuple containing predictions (logits) and labels.

    Returns:
        dict: A dictionary with keys "accuracy", "precision", "recall", "f1".
    """
    logits, labels = pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
def evaluate(model, tokenizer, data_collator, eval_dataset, eval_output_dir):
    """
    Evaluate a trained model on the evaluation dataset
    
    This function creates a Trainer instance specifically for evaluation
    and computes metrics on the test dataset.
    
    Args:
        model: Trained DistilBERT model to evaluate
        tokenizer: Tokenizer for text processing
        data_collator: Function to create properly padded batches
        eval_dataset: Dataset to evaluate on
        eval_output_dir: Directory to save evaluation results
        
    Returns:
        dict: Evaluation results containing computed metrics
    """
    # Training arguments for evaluation (minimal configuration)
    training_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        remove_unused_columns=False,          # Keep unused columns in the dataset    
        label_names=["labels"], 
    )
    
    # Create Trainer instance for evaluation
    trainer = Trainer(
        model=model,                            # The model to be evaluated
        args=training_args,                     # The evaluation arguments
        eval_dataset=eval_dataset,              # Evaluation dataset
        tokenizer=tokenizer,                    # The tokenizer
        data_collator=data_collator,            # Data collator for dynamic padding
        compute_metrics=compute_metrics,        # Function to compute metrics
    )

    return trainer.evaluate()

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Main training script for DistilBERT full fine-tuning on custom subject dataset.

    Steps performed:
    1. Detects available compute device (GPU, MPS, or CPU).
    2. Loads and prepares the subject dataset from JSON files.
    3. Initializes the DistilBERT model, tokenizer, and data collator.
    4. Tokenizes train/test splits and formats them for PyTorch.
    5. Defines Hugging Face TrainingArguments (hyperparameters, evaluation strategy).
    6. Initializes the Hugging Face Trainer with model, data, and metrics.
    7. Trains the model, saving the best checkpoint by accuracy.
    8. Reports total training time and outputs final summary.
    """

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  
    
    # Dataset paths
    history_file = config["dataset-paths"]["history_file"]
    physics_file = config["dataset-paths"]["physics_file"]
    biology_file = config["dataset-paths"]["biology_file"]
    
    # Output directory
    output_dir = config["final-ft-model-paths"]["subject_model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    train_dataset, test_dataset, label2id = get_subject_dataset(history_file, physics_file, biology_file)

    # Model + tokenizer
    model, tokenizer, data_collator = get_model_tokenizer_collator(num_labels=3, label2id=label2id)

    # Tokenize
    tokenized_train, tokenized_test = prepare_dataset(tokenizer, train_dataset, test_dataset)

    # Training args
    use_cpu, no_cuda = False, False
    if device == "cpu":
        use_cpu, no_cuda = True, True

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
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    start_time = time.time()
    print("Starting FULL fine-tuning of DistilBERT for subject classification...")
    trainer.train()
    end_time = time.time()

    # Save model
    trainer.model.save_pretrained(f"{output_dir}/best_model")
    print(f"Fully fine-tuned subject classification model saved to: {output_dir}/best_model")

    # Training time
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training started at: {time.ctime(start_time)}")
    print(f"Training ended at: {time.ctime(end_time)}")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")

    # Summary
    print("\n" + "="*80)
    print("FULL FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Key characteristics of this approach:")
    print("1. ALL model parameters were updated (~66M parameters)")
    print("2. Higher computational and memory requirements")
    print("3. Longer training time compared to LoRA")
    print("4. Generally better performance potential")
    print("5. Larger final model size")
    print("6. Risk of catastrophic forgetting")
    print("\nCompare this with LoRA approaches:")
    print("- LoRA: ~1-5M trainable parameters, faster training, smaller models")
    print("- Full FT: ~66M trainable parameters, slower training, larger models")
    print("="*80)
