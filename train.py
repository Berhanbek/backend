import json
import random
import numpy as np
from tqdm import tqdm  # For progress bars

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Prepare data from intents.json
def prepare_data(intents_path):
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words, tags, xy = [], [], []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
    tags = sorted(set(tags))

    X_train, y_train = [], []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    return np.array(X_train), np.array(y_train), all_words, tags


# Custom Dataset class
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y
        self.n_samples = len(X)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model


def save_model(model, file_path, input_size, hidden_size, output_size, all_words, tags):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, file_path)
    print(f"Training complete. Model saved to {file_path}")


def main():
    set_seed()

    # Load and preprocess data
    X_train, y_train, all_words, tags = prepare_data("intents.json")
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    # Hyperparameters
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001

    # Setup dataset and dataloader
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on device: {device}")
    print(f"Input Size: {input_size}, Output Size: {output_size}, Tags: {tags}")

    # Train the model
    trained_model = train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Save trained model
    save_model(trained_model, "data.pth", input_size, hidden_size, output_size, all_words, tags)


if __name__ == "__main__":
    main()
