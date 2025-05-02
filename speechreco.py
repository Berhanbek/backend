import speech_recognition as sr
import torch
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
import os
import google.generativeai as genai
import json

# Load intents data
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Load the trained model
FILE = "data.pth"
data = torch.load(FILE)
from dotenv import load_dotenv
import time
load_dotenv()
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")

def voice_to_text(language="en-US", pause_threshold=2, energy_threshold=300):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold
    recognizer.energy_threshold = energy_threshold

    try:
        with sr.Microphone() as source:
            print("🎙️ Calibrating for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("⏳ Listening will start in 1 second...")
            time.sleep(1)
            print("✅ Calibration complete. Speak now!")

            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
            print("🔍 Recognizing...")

            # Save the audio to a file
            with open("captured_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print("✅ Audio saved as 'captured_audio.wav'")

            text = recognizer.recognize_google(audio, language="en-US")
            print("✅ You said:", text)
            return text

    except sr.WaitTimeoutError:
        print("⌛ Timeout: No speech detected. Try speaking sooner.")
    except sr.UnknownValueError:
        print("❌ Speech not understood. Please speak clearly.")
    except sr.RequestError as e:
        print(f"⚠️ Google API Error: {e}")
    except Exception as e:
        print(f"🚨 Unexpected error: {e}")
    
    return None

def get_response_from_ai(text):
    # Preprocess the text
    tokenized_text = tokenize(text)
    bag = bag_of_words(tokenized_text, all_words)
    bag = torch.from_numpy(bag).unsqueeze(0)

    # Predict intent
    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate confidence
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    print(f"Predicted Intent: {tag} | Confidence: {confidence.item()}")

    # If confidence is high, return the response from the trained AI
    if confidence.item() >= 0.2:  # Lowered threshold for testing
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return intent["responses"][0]  # Return the first response
    else:
        print("⚠️ Confidence too low. Falling back to Gemini AI.")
        response = gemini_model.generate_content([text])
        print("Gemini AI Response:", response)
        return response.text if hasattr(response, "text") else str(response)

if __name__ == "__main__":
    print("🗣️ Voice to Text Assistant Initialized")
    text = voice_to_text()
    if text:
        response = get_response_from_ai(text)
        print("🤖 AI Response:", response)
    else:
        print("❌ No valid speech detected.")
