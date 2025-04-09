from textblob import TextBlob
import spacy
import textstat
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import language_tool_python

# Ensure required NLTK resources are downloaded
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

def get_sentiment_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

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

def is_passive_advanced(sentence, verbose=False):
    doc = nlp(sentence)
    has_passive_aux = False
    has_nsubjpass = False
    has_participle_passive = False
    patterns_matched = []

    for token in doc:
        # Direct passive aux (e.g., was, is being, will be)
        if token.dep_ == "auxpass":
            has_passive_aux = True
            patterns_matched.append("auxpass")

        # Passive nominal subject
        if token.dep_ == "nsubjpass":
            has_nsubjpass = True
            patterns_matched.append("nsubjpass")

        # Verb in past participle form (VBN) with no agent
        if token.tag_ == "VBN" and token.dep_ in {"acl", "advcl", "relcl", "ROOT"}:
            # Check if it's part of a passive clause
            for child in token.children:
                if child.dep_ == "aux" and child.text.lower() in {"is", "was", "were", "been", "being"}:
                    has_participle_passive = True
                    patterns_matched.append("VBN + aux")

    if verbose:
        print(f"[DEBUG] Matched patterns: {patterns_matched}")

    # Final decision: return if any known passive pattern is detected
    return any([has_passive_aux, has_nsubjpass, has_participle_passive])

# Simple set of informal phrases & contractions
INFORMAL_WORDS = {
    "lol", "bro", "nah", "dude", "lmao", "gonna", "wanna", "kinda", "yep", "nope", "ya", "huh",
    "omg", "like", "idk", "btw", "tbh", "brb", "yo", "gimme"
}

CONTRACTIONS = re.compile(r"\b(?:[a-zA-Z]+n't|'ll|'re|'ve|'d|'m|'s)\b")

def estimate_formality(text):
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    words = [word for word in tokens if word.isalpha()]

    # Heuristics
    has_contractions = bool(CONTRACTIONS.search(text))
    informal_word_count = sum(1 for word in words if word in INFORMAL_WORDS)
    total_words = len(words)
    avg_word_len = sum(len(word) for word in words) / total_words if total_words > 0 else 0
    proper_punctuation = text.strip().endswith(('.', '?', '!'))

    # Score calculation (weights are heuristic-based)
    score = 0
    if not has_contractions:
        score += 0.2
    if informal_word_count == 0:
        score += 0.3
    if avg_word_len > 4.5:
        score += 0.2
    if proper_punctuation:
        score += 0.2
    if text[0].isupper():
        score += 0.1

    return min(round(score, 2), 1.0)

# Harbinger Keywords and Feature Extraction
harbinger_keywords = {
    "claim": {"accordingly", "as a result", "consequently", "conclude that", "clearly", "demonstrates that",
              "entails", "follows that", "hence", "however", "implies", "in fact", "in my opinion", "in short",
              "in conclusion", "indicates that", "it follows that", "it is highly probable that", "it is my contention",
              "it should be clear that", "i believe", "i mean", "i think", "must be that", "on the contrary",
              "points to the conclusions", "proves that", "shows that", "so", "suggests that", "therefore", "thus",
              "the truth of the matter", "to sum up", "we may deduce"},
    "subjectivity": {"abandon", "abhor", "abomination", "absurd", "abuse", "acceptable", "acclaim", "accurate",
                     "accusation", "accuse", "acquittal", "admirable", "admire", "admonish", "adorable", "adore", 
                     "afraid", "aggressive", "alarming", "amazing", "angry", "annoying", "anxious"},
    "expansion": {"additionally", "also", "alternatively", "although", "as if", "as though", "as well", "besides",
                  "further", "furthermore", "however", "in addition", "indeed", "in fact", "moreover", "next",
                  "nonetheless", "on the other hand", "otherwise", "rather", "similarly", "specifically", "then", "yet"},
    "contingency": {"accordingly", "as a result", "because", "consequently", "hence", "if", "in turn", "since", "so that", "therefore", "thus", "unless", "until", "when"},
    "premise": {"assuming that", "because", "deduced", "derived from", "due to", "follows from", "for", "given that",
                "in view of", "indicated by", "may be inferred", "researchers found that", "since the evidence is"},
    "temporal_future": {"after", "afterward", "as soon as", "finally", "later", "next", "once", "then", "thereafter", "ultimately", "until"},
    "temporal_other": {"before", "earlier", "in turn", "meanwhile", "now that", "previously", "simultaneously", "since", "still", "when", "while"},
    "comparisons": {"although", "as if", "as though", "by comparison", "by contrast", "conversely", "however",
                    "in contrast", "instead", "nevertheless", "nonetheless", "on the contrary", "rather", "still", "though", "whereas", "while"}
}

def extract_harbinger_features(text):
    tokens = word_tokenize(text.lower())
    text_joined = " ".join(tokens)

    features = {}

    for category, keywords in harbinger_keywords.items():
        count = sum(1 for kw in keywords if kw in text_joined)
        features[category] = count

    return features