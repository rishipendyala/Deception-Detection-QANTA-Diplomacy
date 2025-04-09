import spacy
import textstat
import string

nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

def extract_stylometric_features(text):
    # Define third-person pronouns
    THIRD_PERSON_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}

    # Count third-person pronouns
    def count_third_person_pronouns(text):
        tokens = word_tokenize(text.lower())
        return sum(1 for word in tokens if word in THIRD_PERSON_PRONOUNS)

    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)
    num_chars = sum(len(word) for word in words)

    # Type-Token Ratio
    ttr = len(set(words)) / len(words) if words else 0

    # Function words (commonly used non-content words)
    function_words = {'the', 'and', 'a', 'an', 'in', 'on', 'for', 'with', 'to', 'of', 'at', 'by', 'from'}
    function_word_count = sum(1 for token in doc if token.text.lower() in function_words)

    # Pronouns
    pronouns = {"I", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them", "my", "our", "your", "his", "their"}
    pronoun_usage = sum(1 for token in doc if token.text.lower() in pronouns)

    # First-person singular pronouns
    first_person_singular_pronouns = {"i", "me", "my", "mine", "myself"}
    first_person_singular_count = sum(1 for token in doc if token.text.lower() in first_person_singular_pronouns)

    # First-person plural pronouns
    first_person_plural_pronouns = {"we", "us", "our", "ours", "ourselves"}
    first_person_plural_count = sum(1 for token in doc if token.text.lower() in first_person_plural_pronouns)

    # Third-person pronouns
    third_person_pronoun_count = count_third_person_pronouns(text)

    # Punctuation counts
    punctuation_counts = {p: text.count(p) for p in string.punctuation}

    features = {
        "avg_sentence_length": num_words / num_sentences if num_sentences else 0,
        "avg_word_length": num_chars / num_words if num_words else 0,
        "type_token_ratio": ttr,
        "function_word_count": function_word_count,
        "pronoun_usage": pronoun_usage,
        "third_person_pronoun_count": third_person_pronoun_count,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "comma_count": punctuation_counts.get(",", 0),
        "period_count": punctuation_counts.get(".", 0),
        "exclamation_count": punctuation_counts.get("!", 0),
    }

    return features

text = "The report was submitted by the intern. It is believed to be accurate. We think more reviews are needed!"
text = "I was with a friend. They said you need to be careful. I think they are right."
features = extract_stylometric_features(text)
for k, v in features.items():
    print(f"{k}: {v}")
