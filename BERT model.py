# Import the required libraries
import numpy as np
import pandas as pd
import torch

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Retrieve the data
train_df = pd.read_csv('../kaggle/input/train.csv')

# Remove rows where either 'question1' or 'question2' is not a string (e.g., NaN values)
train_df = train_df[train_df['question2'].apply(lambda x: isinstance(x, str))]
train_df = train_df[train_df['question1'].apply(lambda x: isinstance(x, str))]

# Split data into training and validation sets. Only using the first 5,000 rows.
q1_train, q1_val, q2_train, q2_val, train_label, test_label =  train_test_split(
    train_df['question1'].iloc[:5000],
    train_df['question2'].iloc[:5000],
    train_df['is_duplicate'].iloc[:5000],
    test_size=0.2,
    stratify=train_df['is_duplicate'].iloc[:5000])

# Tokenization using BERT's tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encoding = tokenizer(list(q1_train), list(q2_train),
                           truncation=True, padding=True, max_length=100)
test_encoding = tokenizer(list(q1_val), list(q2_val),
                          truncation=True, padding=True, max_length=100)

# Create a dataset for PyTorch
class QuoraDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Retrieve a single sample from the dataset
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

# Instantiate the dataset
train_dataset = QuoraDataset(train_encoding, list(train_label))
test_dataset = QuoraDataset(test_encoding, list(test_label))

# Utility function to compute accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Initialize the BERT model
from transformers import BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Set up the optimizer and learning rate scheduler
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# Training function
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # Forward pass
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # Backward pass to compute gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update model parameters
        optim.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("Epoch: %d, Iteration: %d, Loss: %.4f, Progress: %.2f%%" % (
            epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

# Validation function
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            # Forward pass for validation
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average validation loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")

# Run training and validation for 2 epochs
for epoch in range(2):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
