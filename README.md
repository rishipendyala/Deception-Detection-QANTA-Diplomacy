# 🕵️ Deception Detection in Strategic Conversations

A deep learning-based project to detect **deceptive messages** in long-term, multi-party textual interactions — specifically modeled for the strategic game **Diplomacy**.

---

## 📌 Overview

Deception in text communication is a subtle and complex problem, especially in strategic environments like **Diplomacy**, where players may use both truth and lies to influence others. This project aims to build a model that classifies each message as **deceptive** or **truthful** by combining:

- Linguistic features (syntax, sentiment, style)
- Game dynamics (player relations, scores, metadata)
- Sequence modeling (using LSTM architecture)

---

## 📂 Dataset

We use the **QANTA Diplomacy Dataset**, which contains:

- **17,000+ messages** from 12 Diplomacy games
- Each message includes:
  - Text content
  - `sender_label`: whether the sender *admits* to lying
  - `receiver_label`: whether the receiver *perceives* it as a lie
  - Temporal metadata (season, year)
  - Player metadata (speaker, receiver)
  - Game scores (score and delta)

### Key Characteristics:
- Longitudinal, message-level deception
- Annotated by both **sender intent** and **receiver perception**
- Strategic, real-time environment

---

## 📊 Exploratory Data Analysis

- **Message length**: Typically short (20–100 chars)
- **Score distribution**: Most changes near zero (few dramatic betrayals)
- **Temporal peaks**: More messages during Spring and Fall (decision phases)
- **Low feature correlation**: Requires composite features for better modeling

---

## 🧠 Features

### Linguistic
- **Sentiment Polarity** (−1 to +1)
- **Stylometry**: avg sentence/word length, TTR, function words, pronoun usage
- **Readability**: Flesch Reading Ease, FK Grade Level
- **Passive Voice**: binary indicator
- **Formality Score**: 0 (casual) to 1 (formal)
- **Harbinger Words**: rhetorical cue words from claim, premise, etc.

### Metadata
- **Game phase**: year, season
- **Game dynamics**: score, delta
- **Player roles**: speaker, receiver

---

## 🧰 Model Architecture

### ✅ Final Model
A **hybrid LSTM + numerical features** architecture:

- Frozen **pre-trained embeddings**
- **LSTM layer** for message sequences
- Concatenation with **engineered features**
- Two **fully connected layers**
- **Output**: binary classification (lie or truth)

### ❌ Tried and Failed
- **Transformer-based model**: underfit due to small dataset and class imbalance
- **Feedforward baseline**: discarded sequence order, poor contextual awareness

---

## 📈 Results

| Model               | Accuracy | Macro F1 | Lie F1 |
|--------------------|----------|----------|--------|
| Baseline (LSTM)    | 90.84    | 49.51    | -      |
| Baseline (LogReg)  | 91       | -        | -      |
| Paper (Peskov et al.) | N/A   | 57       | 27     |
| **Final (ours)**    | 88       | **60**   | **26** |

> **Note**: Lie F1 is critical due to class imbalance — higher Macro F1 does not guarantee good deception detection.

---

## 📚 Literature

- [It Takes Two to Lie: One to Lie and One to Listen (ACL 2020)](http://cs.umd.edu/~jbg/docs/2020_acl_diplomacy.pdf)
- [The Mafiascum Dataset (arXiv)](https://arxiv.org/abs/1811.07851)

---

## 👨‍💻 Authors

- **Akshat Chaw Parmar** – [akshat22050@iiitd.ac.in](mailto:akshat22050@iiitd.ac.in)
- **Rishi Pendyala** – [rishi22403@iiitd.ac.in](mailto:rishi22403@iiitd.ac.in)
- **Vimal Jayant Subburaj** – [vimal22571@iiitd.ac.in](mailto:vimal22571@iiitd.ac.in)
