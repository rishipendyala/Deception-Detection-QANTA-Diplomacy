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

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def extract_harbinger_features(text):
    tokens = word_tokenize(text.lower())
    text_joined = " ".join(tokens)

    features = {}

    for category, keywords in harbinger_keywords.items():
        count = sum(1 for kw in keywords if kw in text_joined)
        features[category] = count

    return features

text = "I believe this is true. In fact, it suggests that we may deduce the outcome."
features = extract_harbinger_features(text)

for k, v in features.items():
    print(f"{k}: {v}")