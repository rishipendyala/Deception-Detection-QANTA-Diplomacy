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
        # Extract power imbalance features
        self.extract_power_imbalance_features()

    def extract_power_imbalance_features(self):
        """Extract features that represent power imbalance"""
        # Feature 1: Message length (longer messages may indicate dominance)
        self.df['message_length'] = self.df['message'].apply(len)
        
        # Feature 2: Message complexity (more complex language can indicate power)
        self.df['word_count'] = self.df['message'].apply(lambda x: len(word_tokenize(x)))
        self.df['avg_word_length'] = self.df['message'].apply(
            lambda x: np.mean([len(w) for w in word_tokenize(x)]) if len(word_tokenize(x)) > 0 else 0
        )
        
        # Feature 3: Command words frequency (directives indicate authority)
        command_words = ['must', 'should', 'need', 'have to', 'required', 'necessary', 'immediately']
        self.df['command_count'] = self.df['message'].apply(
            lambda x: sum(1 for word in command_words if word in x.lower())
        )
        
        # Feature 4: Question marks (questions can indicate less power)
        self.df['question_count'] = self.df['message'].apply(lambda x: x.count('?'))
        
        # Feature 5: Politeness markers (more politeness can indicate less power)
        politeness_words = ['please', 'thank', 'thanks', 'appreciate', 'grateful', 'sorry']
        self.df['politeness_count'] = self.df['message'].apply(
            lambda x: sum(1 for word in politeness_words if word in x.lower())
        )
        
        # Normalize all features
        power_features = ['message_length', 'word_count', 'avg_word_length', 
                          'command_count', 'question_count', 'politeness_count']
        
        for feature in power_features:
            max_val = self.df[feature].max() if self.df[feature].max() > 0 else 1
            self.df[f'{feature}_norm'] = self.df[feature] / max_val
        
        # Create a composite power score
        self.df['power_score'] = (
            self.df['message_length_norm'] * 0.1 + 
            self.df['word_count_norm'] * 0.2 + 
            self.df['avg_word_length_norm'] * 0.2 + 
            self.df['command_count_norm'] * 0.3 - 
            self.df['question_count_norm'] * 0.1 - 
            self.df['politeness_count_norm'] * 0.1
        )
        
        # Handle NaN values
        self.df['power_score'] = self.df['power_score'].fillna(0)

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

    def get_power_features(self):
        """Return power imbalance features as a numpy array"""
        power_features = ['power_score', 'message_length_norm', 'word_count_norm', 
                          'avg_word_length_norm', 'command_count_norm', 
                          'question_count_norm', 'politeness_count_norm']
        return self.df[power_features].values

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
