# 🕵️ NLP Project — Deception Detection in Diplomacy

## 🧠 Overview

This project aims to detect deception in text-based, multi-party conversations from the strategic board game Diplomacy. Deceptive language is notoriously subtle and context-dependent, especially in games where trust and strategy intertwine. Our approach combines linguistic, stylistic, and contextual features with a BiLSTM-based neural model to classify messages as deceptive or truthful.

The final model integrates message embeddings with numerical cues such as sentiment polarity, stylometric features, and game metadata, outperforming traditional baselines in both accuracy and deception-specific F1 scores.

---

## 👨‍💻 Team

- Akshat Chaw Parmar — 2022050 — akshat22050@iiitd.ac.in  
- Rishi Pendyala — 2022403 — rishi22403@iiitd.ac.in  
- Vimal Jayant Subburaj — 2022571 — vimal22571@iiitd.ac.in  
- Institute: IIIT Delhi  
- Semester: Winter 2025  
- Course: [CSE556 - Natural Language Processing](https://techtree.iiitd.ac.in/viewDescription/filename?=CSE556)  
- Instructor: [Dr. Shad Akhtar](https://scholar.google.co.in/citations?user=KUcO6LAAAAAJ&hl=en)

---

## 📦 Dataset

We use the QANTA Diplomacy dataset comprising over 17,000 annotated player messages from multiple games.

Each message includes:
- 💬 Message Content
- 🎭 Deception Labels: sender (intent), receiver (perceived truth)
- 🎲 Game Metadata: season, year, game ID, player info
- 📈 Game Scores: current score & delta

This rich annotation allows for nuanced deception modeling at the message level within evolving strategic interactions.

---

## 🔍 Exploratory Data Analysis

- Temporal message patterns peak during Spring & Fall turns (key strategic phases).
- Class imbalance: majority of messages are truthful.
- Message lengths typically range from 20–100 characters.
- game_score_delta is highly skewed; large swings are rare but critical.
- Low correlation among numerical features → calls for hybrid feature engineering.

---

## 🛠️ Features

🗣️ Textual:
- Sentiment Polarity ([-1, 1])
- Passive Voice (binary)
- Harbinger Word Presence (discourse markers)
- Formality Score ([0, 1])

📊 Stylometric:
- Avg. Sentence Length
- Avg. Word Length
- Type-Token Ratio
- Function Word Frequency
- Third-person Pronoun Count
- Readability Scores (Flesch Ease, FK Grade Level)
- Punctuation Counts

🧩 Metadata:
- Season, Year
- Player roles, message index
- Game score and delta

All features are normalized or scaled before model input.

---

## 🧪 Model Architecture

### ✅ Final Model: BiLSTM Classifier

- GloVe 300D Embeddings (frozen)
- BiLSTM Layer (bidirectional, hidden_dim=100)
- Max-pooling + Dropout (p=0.5)
- Fully Connected Layer → Sigmoid
- Loss: Binary Cross-Entropy

📈 Evaluation:
- Accuracy
- Macro F1
- Lie F1 (to measure deception-specific performance)


## BiLSTM Model Testing

This document provides a step-by-step explanation of how the BiLSTM model is tested using the provided Python code. The process involves loading pre-trained embeddings, preprocessing data, defining datasets, and evaluating the model.

---

## Prerequisites

Before running the testing script, ensure the following prerequisites are met:

1. **Dependencies**: Install the required Python libraries:
    - `numpy`
    - `pandas`
    - `json`
    - `torch`
    - `spacy`
    - `scikit-learn`
    Run: pip install -r requirements.txt in the terminal
    Run this in terminal as well for spacy(en_core_web_sm): python3 -m spacy download en_core_web_sm

2. **Pre-trained Model**: Ensure the pre-trained BiLSTM model file (`best_lstm_model.pth`) is available in the working directory.
3. **GloVe Embeddings**: Download the GloVe embedding file (`glove.6B.300d.txt`) and provide the correct path in the code.
4. **Data Files**: Ensure the `test_sm.jsonl` and `validation_sm.jsonl` files are available and correctly formatted.

---

## Steps in the Testing Process

### 1. Load and Preprocess Data

The data is loaded from JSONL files (`test_sm.jsonl` and `validation_sm.jsonl`) and preprocessed to retain only the `message` and `sender_annotation` columns.

```python
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
```

### 2. Load GloVe Embeddings

The GloVe embeddings are loaded into a dictionary for converting text into vector representations.

```python
def load_glove_embeddings(glove_path, embedding_dim=300):
    word_to_vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            word_to_vec[word] = vector
    return word_to_vec
```

### 3. Convert Text to Embeddings

The `message` column of the dataset is converted into fixed-size sequences of embeddings using the GloVe vectors.

```python
def convert_text_to_embedding(text, glove_embeddings, embedding_dim=300, max_seq_len=100):
    tokens = text.split()
    embeddings = [glove_embeddings[word] if word in glove_embeddings else np.zeros(embedding_dim) for word in tokens]

    # Pad or truncate
    if len(embeddings) > max_seq_len:
        embeddings = embeddings[:max_seq_len]
    else:
        embeddings += [np.zeros(embedding_dim)] * (max_seq_len - len(embeddings))

    return np.array(embeddings, dtype=np.float32)

# Apply function to datasets
test_data['embeddings'] = test_data['message'].apply(lambda x: convert_text_to_embedding(x, glove_embeddings))
validation_data['embeddings'] = validation_data['message'].apply(lambda x: convert_text_to_embedding(x, glove_embeddings))
```

### 4. Create Custom Dataset Class

The `MessageDataset` class is used to create PyTorch datasets for the test and validation data.

```python
class MessageDataset(Dataset):
    def __init__(self, df):
        self.embeddings = torch.tensor(np.stack(df['embeddings'].values), dtype=torch.float32)
        self.labels = torch.tensor(df['sender_annotation'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
```

### 5. Define and Load the BiLSTM Model

The BiLSTM model is defined with an LSTM layer, dropout, and a fully connected layer. The model is then loaded with the pre-trained weights.

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=300, hidden_size=100, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.max(lstm_out, dim=1)[0]
        dropped = self.dropout(pooled)
        output = self.fc(dropped)
        return self.sigmoid(output).squeeze(1)

# Load the model
model = BiLSTMClassifier().to(device)
model.load_state_dict(torch.load("best_lstm_model.pth"))
model.eval()
```

### 6. Evaluate the Model

The model is evaluated on the test dataset, and performance metrics like accuracy and F1-score are computed.

```python
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()

        # Compute accuracy
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Store predictions and labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Compute Macro F1-score
f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Macro F1-score: {f1:.4f}")
```

---

## Output Metrics

1. **Accuracy**: Displays the proportion of correct predictions.
2. **Macro F1-score**: Measures the balance between precision and recall across all classes.
