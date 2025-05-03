from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pymysql
import torch
import random
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr
from google.cloud import speech_v1p1beta1 as speech
from trainvoice import get_response_from_ai, add_to_intents, save_intents_to_file
import uuid

# Define safe paths for files
INTENTS_PATH = os.path.join(os.path.dirname(__file__), "intents.json")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.pth")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


# Load environment variables
load_dotenv()

# Load Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")
genai.configure(api_key=GEMINI_API_KEY)

# Set up Gemini ISSEER model with system prompt and config
import google.generativeai as genai

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash-8b",
    system_instruction=(
        "You are ISSEER — a masterful Information Systems educator, born on April 24, 2025.\n"
        "Your mission is to communicate IS concepts with clarity, actionable insights, and real-world relevance.\n\n"
        "🧠 **Formatting & Style Guide:**\n"
        "- Use emoji section headers (e.g., 📚 Overview, 💡 Key Points, 🌍 Real-World Examples, 🤔 Reflect & Apply).\n"
        "- Use emoji bullets (e.g., 🔹, ✅, 📌) for lists instead of plain * or -.\n"
        "- Use **bold** for key terms and *italics* for emphasis.\n"
        "- Use proper indentation and logical bullet-point flows.\n"
        "- Never return a wall of text. Always break up content for readability.\n"
        "- End with 2–3 'Reflect & Apply' questions to challenge the learner's thinking.\n\n"
        "🏢 **IS Department Instructor Directory:**\n"
        "Only include the instructor directory if the user's question is about IS Department instructors or mentions an instructor's name.\n"
        "When you do, respond with a well-structured emoji-bullet list, not a table. Example:\n"
        "- **W/ro Adey Edessa**\n"
        "  - 🏢 Room: Eshetu Chole 113\n"
        "  - 📧 Email: adey.edessa@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/adey-edessa-4b7383240/\n"
        "- **W/t Amina Abdulkadir**\n"
        "  - 🏢 Room: Eshetu Chole 122\n"
        "  - 📧 Email: amina.abdulkadir@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/amina-a-hussein-766b35155/\n"
        "- **Ato Andargachew Asfaw**\n"
        "  - 🏢 Room: Eshetu Chole 319\n"
        "  - 📧 Email: andargachew.asfaw@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/andargachew-asfaw/\n"
        "- **Ato Aminu Mohammed**\n"
        "  - 🏢 Room: Eshetu Chole 424\n"
        "  - 📧 Email: aminu.mohammed@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/aminu-mohammed-47514736/\n"
        "- **W/t Dagmawit Mohammed**\n"
        "  - 🏢 Room: Eshetu Chole 122\n"
        "  - 📧 Email: dagmawit.mohammed@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/dagmawit-mohammed-5bb050b1/\n"
        "- **Dr. Dereje Teferi**\n"
        "  - 🏢 Room: Eshetu Chole 419\n"
        "  - 📧 Email: dereje.teferi@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/dereje-teferi/\n"
        "- **Dr. Ermias Abebe**\n"
        "  - 🏢 Room: Eshetu Chole 115\n"
        "  - 📧 Email: ermias.abebe@aau.edu.et\n"
        "- **Dr. Getachew H/Mariam**\n"
        "  - 🏢 Room: Eshetu Chole 618\n"
        "  - 📧 Email: getachew.h@mariam@aau.edu.et\n"
        "- **Ato G/Michael Meshesha**\n"
        "  - 🏢 Room: Eshetu Chole 122\n"
        "  - 📧 Email: gmichael.meshesha@aau.edu.et\n"
        "- **Ato Kidus Menfes**\n"
        "  - 🏢 Room: Eshetu Chole 511\n"
        "  - 📧 Email: kidus.menfes@aau.edu.et\n"
        "- **W/o Lemlem Hagos**\n"
        "  - 🏢 Room: Eshetu Chole 116\n"
        "  - 📧 Email: lemlem.hagos@aau.edu.et\n"
        "- **Dr. Lemma Lessa**\n"
        "  - 🏢 Room: Eshetu Chole 417\n"
        "  - 📧 Email: lemma.lessa@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/lemma-l-51504635/\n"
        "- **Dr. Martha Yifiru**\n"
        "  - 🏢 Room: Eshetu Chole 420\n"
        "  - 📧 Email: martha.yifiru@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/martha-yifiru-7b0b3b1b/\n"
        "- **Ato Melaku Girma**\n"
        "  - 🏢 Room: Eshetu Chole 224\n"
        "  - 📧 Email: melaku.girma@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/melaku-girma-23031432/\n"
        "- **W/o Meseret Hailu**\n"
        "  - 🏢 Room: Eshetu Chole 113\n"
        "  - 📧 Email: meseret.hailu@aau.edu.et\n"
        "- **Dr. Melekamu Beyene**\n"
        "  - 🏢 Room: Eshetu Chole 423\n"
        "  - 📧 Email: melekamu.beyene@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/melkamu-beyene-6462a444/\n"
        "- **Ato Miftah Hassen**\n"
        "  - 🏢 Room: Eshetu Chole 424\n"
        "  - 📧 Email: miftah.hassen@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/miftah-hassen-18ab10107/\n"
        "- **W/t Mihiret Tibebe**\n"
        "  - 🏢 Room: Eshetu Chole 113\n"
        "  - 📧 Email: mihiret.tibebe@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/mihret-tibebe-0b0b3b1b/\n"
        "- **Dr. Million Meshesha**\n"
        "  - 🏢 Room: Eshetu Chole 418\n"
        "  - 📧 Email: million.meshesha@aau.edu.et\n"
        "- **Dr. Rahel Bekele**\n"
        "  - 🏢 Room: Eshetu Chole 221\n"
        "  - 📧 Email: rahel.bekele@aau.edu.et\n"
        "- **Ato Selamawit Kassahun**\n"
        "  - 🏢 Room: ---\n"
        "  - 📧 Email: selamawit.kassahun@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/selamawit-kassahun-93b9b6128/\n"
        "- **Dr. Solomon Tefera**\n"
        "  - 🏢 Room: Eshetu Chole 421\n"
        "  - 📧 Email: solomon.tefera@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/solomon-tefera-42a07871/\n"
        "- **Dr. Temtem Assefa**\n"
        "  - 🏢 Room: Eshetu Chole 622\n"
        "  - 📧 Email: temtem.assefa@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/temtim-assefa-61a15936/\n"
        "- **Ato Teshome Alemu**\n"
        "  - 🏢 Room: Eshetu Chole 224\n"
        "  - 📧 Email: teshome.alemu@aau.edu.et\n"
        "- **Dr. Wondwossen Mulugeta**\n"
        "  - 🏢 Room: Eshetu Chole 114\n"
        "  - 📧 Email: wondwossen.mulugeta@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/wondisho/\n"
        "- **Ato Wendwesen Endale**\n"
        "  - 🏢 Room: Eshetu Chole 319\n"
        "  - 📧 Email: wendwesen.endale@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/wendwesenendale/\n"
        "- **Dr. Workshet Lamenew**\n"
        "  - 🏢 Room: Eshetu Chole 222\n"
        "  - 📧 Email: workshet.lamenew@aau.edu.et\n"
        "- **Ato Mengisti Berihu**\n"
        "  - 🏢 Room: ---\n"
        "  - 📧 Email: mengisti.berihu@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/mengisti-berihu-5272b7126/\n"
        "- **W/o Meseret Ayano**\n"
        "  - 🏢 Room: ---\n"
        "  - 📧 Email: meseret.ayano@aau.edu.et\n"
        "  - 🔗 LinkedIn: https://www.linkedin.com/in/meseret-ayano-1b3383148/\n"
        ""
        "Conclude with: 'Please double-check with the IS Department Office for the latest updates.'\n\n"
        "📣 Reminder: Always double-check with the department office for the latest updates! 🏢✅\n"
        "🌈 Have an amazing day ahead! 💬🌟\n"
        "Tone keywords: Intellectual, practical, empowering, structured, mentor-like."
    ),
    generation_config={
        "temperature": 0.95,
        "top_p": 0.85,
        "max_output_tokens": 1200,
    }
)


# Load intents
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)
except Exception as e:
    raise Exception(f"Failed to load intents.json: {str(e)}")

# Load custom model data
try:
    data = torch.load(DATA_PATH)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    custom_model = NeuralNet(input_size, hidden_size, output_size)
    custom_model.load_state_dict(model_state)
    custom_model.eval()
except Exception as e:
    raise Exception(f"Model loading error: {str(e)}")

def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.Cursor
    )

def reload_intents():
    global intents
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)

conn = get_db_connection()
# Ensure the database connection is established
# Ensure tables exist
def setup_database(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                chat_id INT NOT NULL,
                sender VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.commit()

conn = get_db_connection()
setup_database(conn)

def voice_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Speech Recognition service is unavailable."

# Route message to custom model or Gemini
def route_question(msg):
    tokens = tokenize(msg)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Get prediction from custom model
    output = custom_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    reload_intents()

    # Calculate confidence
    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    print(f"Custom model predicted tag: {tag} | Confidence: {confidence.item()}")

    # If custom model is confident enough, use its response
    if confidence.item() >= 0.3:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        # Fallback to Gemini ISSEER
        try:
            response = gemini_model.generate_content([msg])
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in Gemini response: {str(e)}")
            return "Sorry, I couldn't process your request."


# ROUTES
@app.route("/chat/new", methods=["POST"])
def new_chat():
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO chats (title) VALUES ('New Chat')")
            conn.commit()
            chat_id = cursor.lastrowid

            cursor.execute("INSERT INTO messages (chat_id, sender, content) VALUES (%s, %s, %s)",
                           (chat_id, "bot", "How can I help you?"))
            conn.commit()

        return jsonify({
            "id": chat_id,
            "title": "New Chat",
            "messages": [{"sender": "bot", "content": "How can I help you?"}]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chats", methods=["GET"])
def get_chats():
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, title, created_at FROM chats")
            rows = cursor.fetchall()
            chats = [
                {
                    "id": row[0],
                    "title": row[1],
                    "createdAt": row[2].strftime("%Y-%m-%d %H:%M:%S")
                }
                for row in rows
            ]
        return jsonify(chats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/message", methods=["POST"])
def send_message_to_bot():
    data = request.get_json()
    content = data.get("content")

    if not content:
        return jsonify({"error": "Message content is required"}), 400

    try:
        bot_reply = route_question(content)
        return jsonify({"bot_reply": bot_reply})  # Ensure the response key matches the frontend expectation
    except Exception as e:
        print(f"Error in /message endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat/delete/<int:chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
            conn.commit()
        return jsonify({"success": True, "chat_id": chat_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-response", methods=["POST"])
def get_response():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Input text is required"}), 400

    result = get_response_from_ai(text)
    return jsonify(result)
@app.route("/messages/<int:chat_id>", methods=["GET"])
def get_messages(chat_id):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT sender, content, created_at FROM messages WHERE chat_id = %s", (chat_id,))
            rows = cursor.fetchall()
            messages = [{
                "sender": row[0],
                "content": row[1],
                "createdAt": row[2].strftime("%Y-%m-%d %H:%M:%S")
            } for row in rows]
    return jsonify(messages)

@app.route("/add-intent", methods=["POST"])
def add_intent():
    data = request.get_json()
    tag = data.get("tag")
    patterns = data.get("patterns", [])
    responses = data.get("responses", [])

    if not tag or not patterns or not responses:
        return jsonify({"error": "Tag, patterns, and responses are required"}), 400

    add_to_intents(tag, patterns, responses)
    save_intents_to_file()
    return jsonify({"success": True, "message": f"Intent '{tag}' added successfully."})
@app.route("/voice", methods=["POST"])
def transcribe_audio(audio_file):
    client = speech.SpeechClient()
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(language_code="en-US")

    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript
def handle_voice_message():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file"}), 400

    audio_path = f"temp_audio_{uuid.uuid4().hex}.webm"

    try:
        audio_file.save(audio_path)
        print(f"Audio file saved at: {audio_path}")

        transcribed_text = transcribe_audio(audio_path).strip()
 # Transcribe the audio file
        print(f"Transcribed Text: {transcribed_text}")

        if not transcribed_text:
            return jsonify({"error": "Transcription failed"}), 400

        bot_reply = route_question(transcribed_text)
        print(f"Bot Reply: {bot_reply}")

        return jsonify({
            "transcribed_text": transcribed_text,
            "bot_reply": bot_reply
        })
    except Exception as e:
        print(f"Error in /voice endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Start server
if __name__ == "__main__":
    print("Server running on http://0.0.0.0:8080")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

transcribed_text = transcribe_audio("path/to/sample_audio.webm")
print("Transcribed Text:", transcribed_text)