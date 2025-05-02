import json
import random
import nltk
import numpy as np
import string
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download("punkt")
nltk.download("wordnet")

# Load intents
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocess
def clean_text(text):
    if not text.strip():
        return ""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return " ".join(tokens)

# Prepare training data
corpus = []
tags = []
tag_classes = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(clean_text(pattern))
        tags.append(intent["tag"])
    tag_classes[intent["tag"]] = intent["responses"]

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.array(tags)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open("chat_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

CONFIDENCE_THRESHOLD = 0.2

# Chat Function
def chat():
    print("🤖 TheISSeer is ready! Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() in ["quit", "exit", "አልነጋገርም"]:
            print("Bot: Logging off... Goodbye!")
            break
        if not inp.strip():
            print("Bot: Please type something!")
            continue

        cleaned = clean_text(inp)
        vect_inp = vectorizer.transform([cleaned])
        pred_prob = model.predict_proba(vect_inp)[0]
        pred_index = np.argmax(pred_prob)
        confidence = pred_prob[pred_index]

        if confidence > CONFIDENCE_THRESHOLD:
            tag = model.classes_[pred_index]
            response = random.choice(tag_classes[tag])
        else:
            response = random.choice([
                "Hmm, I’m not sure I understood that. Try rephrasing?",
                "Sorry, that’s outside my training data for now!",
                "Could you clarify that a bit?"
            ])
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
