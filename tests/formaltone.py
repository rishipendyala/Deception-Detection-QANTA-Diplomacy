import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

# Simple set of informal phrases & contractions
INFORMAL_WORDS = {
    "lol", "bro", "nah", "dude", "lmao", "gonna", "wanna", "kinda", "yep", "nope", "ya", "huh",
    "omg", "like", "idk", "btw", "tbh", "brb", "yo", "gimme"
}

CONTRACTIONS = re.compile(r"\\b(?:[a-zA-Z]+n't|'ll|'re|'ve|'d|'m|'s)\\b")

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


formal_text = "Dear Sir, I would like to request your assistance with the following matter."
informal_text = "yo dude can u help me out with this thing lol"

print("Formal:", estimate_formality(formal_text))     # e.g., 0.9
print("Informal:", estimate_formality(informal_text)) # e.g., 0.2


import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

# Simple set of informal phrases & contractions
INFORMAL_WORDS = {
    "lol", "bro", "nah", "dude", "lmao", "gonna", "wanna", "kinda", "yep", "nope", "ya", "huh",
    "omg", "like", "idk", "btw", "tbh", "brb", "yo", "gimme"
}

CONTRACTIONS = re.compile(r"\\b(?:[a-zA-Z]+n't|'ll|'re|'ve|'d|'m|'s)\\b")

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


formal_text = "Dear Sir, I would like to request your assistance with the following matter."
informal_text = "yo dude can u help me out with this thing lol"

print("Formal:", estimate_formality(formal_text))     # e.g., 0.9
print("Informal:", estimate_formality(informal_text)) # e.g., 0.2

