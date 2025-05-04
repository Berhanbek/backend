import os
import nltk

nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)
import numpy as np
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

stemmer = PorterStemmer()

# Download resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)

# Add filler words and custom stop words
filler_words = {"um", "uh", "you know", "like", "okay", "so", "well"}
custom_stop_words = {"please", "thank", "thanks"}
stop_words.update(filler_words)
stop_words.update(custom_stop_words)

# Contractions dictionary
contractions = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not",
}

def expand_contractions(sentence):
    words = sentence.split()
    expanded_words = [contractions[word] if word in contractions else word for word in words]
    return TreebankWordDetokenizer().detokenize(expanded_words)

# Clean & tokenize text
def tokenize(sentence):
    sentence = sentence.lower()
    sentence = expand_contractions(sentence)  # Expand contractions
    sentence = re.sub(r"\s+", " ", sentence)  # Normalize spaces
    sentence = re.sub(r"[^\w\s]", "", sentence)  # Remove punctuation
    return [stem(word) for word in nltk.word_tokenize(sentence) if word not in stop_words]

# Stem a word
def stem(word):
    return stemmer.stem(word.lower())

# Bag of words with optional frequency weighting
def bag_of_words(tokenized_sentence, words_vocab, use_frequency=False):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words_vocab), dtype=np.float32)

    word_count = {}
    for word in sentence_words:
        word_count[word] = word_count.get(word, 0) + 1

    for idx, vocab_word in enumerate(words_vocab):
        if vocab_word in sentence_words:
            if use_frequency:
                bag[idx] = word_count[vocab_word]  # frequency weight
            else:
                bag[idx] = 1.0  # binary presence
    return bag
