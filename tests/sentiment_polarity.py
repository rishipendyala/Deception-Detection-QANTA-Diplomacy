from textblob import TextBlob

def get_sentiment_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

text = "This movie was decent."
sentiment = get_sentiment_polarity(text)
print(f"Sentiment polarity: {sentiment}")