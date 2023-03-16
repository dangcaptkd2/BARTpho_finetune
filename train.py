import nltk
import os
nltk.download('punkt')
os.environ["WANDB_DISABLED"] = "true"

import os
import re

import nltk
# import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset, load_metric, load_from_disk

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import yaml
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(">> config", config)


dataset = load_dataset("csv", data_files="./save/data.csv")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

metric = load_metric("rouge")

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")

def preprocess_function(examples):
    max_input_length = 1024
    max_target_length = 256
    inputs = [doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-word")
model.eval()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

batch_size = config['batch_size']
args = Seq2SeqTrainingArguments(
    "./run",
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=config['epoch'],
    predict_with_generate=True,
    generation_max_length=100,
    generation_num_beams=1,
    )

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("/save/model_3")