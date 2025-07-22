import numpy as np
import pandas as pd
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spacy
from sklearn.metrics import f1_score

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Preprocess data
def preprocess_data(df):
    df = df[['message', 'sender_annotation']].copy()
    df['sender_annotation'] = df['sender_annotation'].astype(int)
    return df

test_data = preprocess_data(load_data("data/test_sm.jsonl"))
validation_data = preprocess_data(load_data("data/validation_sm.jsonl"))

def load_glove_embeddings(glove_path, embedding_dim=200):
    word_to_vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            word_to_vec[word] = vector
    return word_to_vec

glove_path = "glove.6B/glove.6B.300d.txt"
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim=300)

nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    return [token.text.lower() for token in nlp(text)]

MAX_SEQ_LEN = 50  # Define max sequence length

def convert_text_to_embedding(text, glove_embeddings, embedding_dim=300, max_seq_len=100):
    tokens = text.split()  # Tokenization
    embeddings = [glove_embeddings[word] if word in glove_embeddings else np.zeros(embedding_dim) for word in tokens]

    # Pad or truncate to `max_seq_len`
    if len(embeddings) > max_seq_len:
        embeddings = embeddings[:max_seq_len]
    else:
        embeddings += [np.zeros(embedding_dim)] * (max_seq_len - len(embeddings))

    return np.array(embeddings, dtype=np.float32)  # Ensure consistent dtype

# Apply function to datasets
test_data['embeddings'] = test_data['message'].apply(lambda x: convert_text_to_embedding(x, glove_embeddings))
validation_data['embeddings'] = validation_data['message'].apply(lambda x: convert_text_to_embedding(x, glove_embeddings))


class MessageDataset(Dataset):
    def __init__(self, df):
        self.embeddings = torch.tensor(np.stack(df['embeddings'].values), dtype=torch.float32)
        self.labels = torch.tensor(df['sender_annotation'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create dataset objects
test_dataset = MessageDataset(test_data)
validation_dataset = MessageDataset(validation_data)

# Create DataLoaders
BATCH_SIZE = 32
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=300, hidden_size=100, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # BiLSTM has 2x hidden_size output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        pooled = torch.max(lstm_out, dim=1)[0]  # Max pooling
        dropped = self.dropout(pooled)
        output = self.fc(dropped)
        return self.sigmoid(output).squeeze(1)

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier().to(device)

model.load_state_dict(torch.load("best_lstm_model.pth"))
model.to(device)
model.eval()

# Initialize lists for predictions and true labels
all_preds = []
all_labels = []

# Evaluate on test data
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Convert probabilities to binary predictions (assuming binary classification)
        predicted = (outputs > 0.5).float()

        # Compute accuracy
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Store predictions and labels for F1-score calculation
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Accuracy
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Compute F1-score (Macro F1)
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Macro F1-score: {f1:.4f}")