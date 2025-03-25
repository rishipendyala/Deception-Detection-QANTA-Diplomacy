import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import emoji
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

class DataProcessor:
    def __init__(self, dataframe, glove_vectors=None, fasttext_vectors=None):
        self.df = dataframe
        self.stop_words = set(stopwords.words('english'))
        self.glove_vectors = glove_vectors
        self.fasttext_vectors = fasttext_vectors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_text(self, text):
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stop words and extra spaces
        text = ' '.join([word for word in word_tokenize(text.lower()) if word not in self.stop_words])
        return text

    def preprocess_dataset(self):
        self.df['cleaned_message'] = self.df['message'].apply(self.preprocess_text)

    def vectorize(self, method='tfidf'):
        if method == 'tfidf':
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(self.df['cleaned_message'])
        elif method == 'glove':
            vectors = self._get_embeddings(self.df['cleaned_message'], self.glove_vectors)
        elif method == 'fasttext':
            vectors = self._get_embeddings(self.df['cleaned_message'], self.fasttext_vectors)
        elif method == 'bert':
            vectors = self._get_bert_embeddings(self.df['cleaned_message'])
        else:
            raise ValueError("Unsupported vectorization method")
        return vectors

    def _get_bert_embeddings(self, texts):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        encoded_inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

    def _get_embeddings(self, texts, embedding_model, max_length=50):
        embeddings = []
        for text in texts:
            tokens = word_tokenize(text)
            text_embeddings = []
            for token in tokens[:max_length]:
                try:
                    # Use float32 for embedding vectors
                    embedding = embedding_model[token].astype(np.float32)
                except KeyError:
                    # Create a zero vector with float32 data type
                    embedding = np.zeros(embedding_model.vector_size, dtype=np.float32)
                text_embeddings.append(embedding)
            if len(text_embeddings) < max_length:
                # Create padding with float32 data type
                padding = [np.zeros(embedding_model.vector_size, dtype=np.float32)] * (max_length - len(text_embeddings))
                text_embeddings.extend(padding)
            # Convert text_embeddings to a float32 NumPy array
            embeddings.append(np.array(text_embeddings, dtype=np.float32))
        # Convert the entire list to a NumPy array with float32 data type
        return np.array(embeddings, dtype=np.float32)  # Shape: (num_samples, max_length, embedding_dim)


    def fit_transform(self, vectorization_method='tfidf'):
        self.preprocess_dataset()
        vectors = self.vectorize(method=vectorization_method)
        print("\nCompleted fit_transform with method:", vectorization_method)
        return vectors
