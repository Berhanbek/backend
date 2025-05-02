import json
import torch
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Load the trained model
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def get_response_from_ai(text):
    """Process the input text and return a response."""
    tokens = tokenize(text)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Get prediction from the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate confidence
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    print(f"Predicted tag: {tag} | Confidence: {confidence.item()}")

    # If confidence is high, return a response
    if confidence.item() >= 0.3:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return {"response": intent["responses"], "tag": tag}

    # If confidence is low, add the new knowledge dynamically
    return {"response": "I don't know the answer to that yet. Can you teach me?", "tag": "unknown"}

def add_to_intents(tag, patterns, responses):
    """Dynamically add new knowledge to the intents."""
    new_intent = {
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    }
    intents["intents"].append(new_intent)
    print(f"✅ Added new intent: {tag}")

def save_intents_to_file():
    with open("intents.json", "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=4)
    print("✅ Intents saved to intents.json")

if __name__ == "__main__":
    print("🗣️ Real-Time AI Initialized")
    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        result = get_response_from_ai(text)
        print(f"Bot: {result['response']}")

        if result["tag"] == "unknown":
            print("🤖 Let's add this to my knowledge base.")
            tag = input("Enter a tag for this knowledge: ")
            patterns = [text]
            responses = [input("Enter a response: ")]


            add_to_intents(tag, patterns, responses)
            save_intents_to_file()