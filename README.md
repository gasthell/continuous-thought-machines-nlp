# Continuous-Thought Machines for NLP (CTM-NLP)

**An adaptation of the [Continuous-Thought Machines](https://github.com/SakanaAI/continuous-thought-machines) architecture by Sakana AI for Natural Language Processing (NLP) tasks.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Original Paper](https://img.shields.io/badge/Original_Paper-Sakana_AI-orange)](https://pub.sakana.ai/ctm/)

This repository contains the code and resources for applying the Continuous-Thought Machines (CTM) architecture to NLP tasks. The original CTM was introduced as a versatile architecture capable of solving problems across different domains, from image classification to maze solving [1]. We adapt its core principles for deep text understanding and generation.

## Core Concepts

The key idea behind CTM is that "thought takes time, and reasoning is a process" [1]. Unlike traditional models like Transformers with a fixed number of layers, CTM introduces an internal time axis. This allows the model to dynamically "think" about the input data for as long as needed to solve a given task.

For NLP, this translates to:

*   **Dynamic Processing Depth**: Simple sentences can be processed quickly, while complex syntactic and semantic structures can trigger more "thinking" iterations.
*   **Improved Handling of Long-Range Dependencies**: Thanks to neurons that process the history of incoming signals, the model can potentially capture context better in long documents.
*   **Neural Synchronization as an Attention Mechanism**: Instead of classic attention mechanisms, information is encoded in the firing times of neurons, which could be a more efficient way to highlight important parts of the text.

## Supported Tasks

This architecture has been adapted and tested for the following NLP tasks:

*   **Text Classification** (e.g., sentiment analysis)
*   **(In progress) Semantic Similarity Analysis**
*   **(In progress) Text Generation**

## Installation

It is recommended to use `conda` to set up the environment.

```bash
# Create and activate the environment
conda create --name=ctm_nlp python=3.10
conda activate ctm_nlp

# Install dependencies
pip install -r requirements.txt
```

Ensure you have a compatible version of PyTorch installed.

## Quick Start

Here is an example of how to use a CTM-NLP model for text classification .

```python
# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import time
import os
import urllib.request
import tarfile
import csv
from collections import Counter
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
VOCAB_SIZE_LIMIT = 10000
MAX_SEQ_LEN = 512

EPOCHS = 3
LEARNING_RATE = 0.001

CTM_D_MODEL = 256
CTM_D_INPUT = 128
CTM_ITERATIONS = 10
CTM_HEADS = 4
CTM_SYNCH_OUT = 128
CTM_SYNCH_ACTION = 64
CTM_SYNAPSE_DEPTH = 2
CTM_MEMORY_LENGTH = 10
CTM_MEMORY_HIDDEN = 32
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2

# Example of donwloading AG_NEWS
def download_and_extract_ag_news(root='./data'):
    """Donwnloading AG_NEWS"""
    url = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
    data_path = os.path.join(root, 'ag_news_csv')
    
    if os.path.exists(data_path):
        print("Dataset already downloaded and extracted.")
    else:
        print("Downloading AG_NEWS dataset...")
        os.makedirs(root, exist_ok=True)
        tgz_path = os.path.join(root, 'ag_news_csv.tgz')
        urllib.request.urlretrieve(url, tgz_path)
        print("Extracting...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=root)
        os.remove(tgz_path)
        print("Done.")
        
    train_data, test_data = [], []
    with open(os.path.join(data_path, 'train.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Класс, Заголовок, Описание
            train_data.append((int(row[0]), row[1] + " " + row[2]))
            
    with open(os.path.join(data_path, 'test.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            test_data.append((int(row[0]), row[1] + " " + row[2]))
            
    return train_data, test_data

def simple_tokenizer(text):
    """Simple example of tokenization"""
    return text.lower().strip().split()
    # return word_tokenize(text.lower().strip())

def build_vocab(data, tokenizer, max_size):
    """word -> index"""
    counter = Counter()
    for _, text in data:
        counter.update(tokenizer(text))

    most_common_words = [word for word, _ in counter.most_common(max_size - 2)] # -2 for <pad> и <unk>
    
    word_to_idx = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(most_common_words):
        word_to_idx[word] = i + 2
        
    return word_to_idx

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = NewsDataset(train_data)
test_dataset = NewsDataset(test_data)
    
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

results = {}

# Initialize model
ctm_model = CTM_NLP(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    iterations=CTM_ITERATIONS,
    d_model=CTM_D_MODEL,
    d_input=CTM_D_INPUT,
    # out_dims=NUM_CLASS,
    heads=CTM_HEADS,
    n_synch_out=CTM_SYNCH_OUT,
    n_synch_action=CTM_SYNCH_ACTION,
    synapse_depth=CTM_SYNAPSE_DEPTH,
    memory_length=CTM_MEMORY_LENGTH,
    deep_nlms=True,
    memory_hidden_dims=CTM_MEMORY_HIDDEN,
    do_layernorm_nlm=False,
    dropout=0.2
).to(DEVICE)
    
optimizer_ctm = torch.optim.AdamW(ctm_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Training
for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train_acc, train_loss = train_epoch(ctm_model, train_dataloader, optimizer_ctm, criterion, model_type='ctm')
        test_acc, test_loss = evaluate(ctm_model, test_dataloader, criterion, model_type='ctm')
        
        print(f'CTM Epoch: {epoch}, Time: {time.time() - epoch_start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%')
    
results['CTM_NLP'] = test_acc
```

## Repository Structure

```
├── models/             # Source code for the adapted CTM-NLP architecture
├── ag_news.ipynb       # Starter
└── requirements.txt    # Project dependencies
```

## Acknowledgements

This work is a direct adaptation and extension of the ideas presented by the **Sakana AI** team. Huge thanks to the authors for their groundbreaking work and for open-sourcing their code.

*   **Original Repository**: [SakanaAI/continuous-thought-machines](https://github.com/SakanaAI/continuous-thought-machines)
*   **Technical Report**: [pub.sakana.ai/ctm/](https://pub.sakana.ai/ctm/)

## License

This project is licensed under the Apache 2.0 License, the same as the original CTM repository.
