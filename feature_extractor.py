from utils import get_sentiment_polarity, extract_stylometric_features, is_passive_advanced, estimate_formality, extract_harbinger_features

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, text):
        features = {}

        # Sentiment Polarity
        features['sentiment_polarity'] = get_sentiment_polarity(text)

        # Stylometric Features
        stylometric_features = extract_stylometric_features(text)
        features.update(stylometric_features)

        # Passive Voice Detection
        features['is_passive'] = is_passive_advanced(text)

        # Formality Estimation
        features['formality_score'] = estimate_formality(text)

        # Harbinger Features
        harbinger_features = extract_harbinger_features(text)
        features.update(harbinger_features)

        return features
