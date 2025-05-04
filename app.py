import nltk
import os

NLTK_DATA_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download only necessary NLTK resources
nltk.download("punkt", download_dir=NLTK_DATA_PATH)
nltk.download("stopwords", download_dir=NLTK_DATA_PATH)
nltk.download("wordnet", download_dir=NLTK_DATA_PATH)
nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_PATH)
nltk.download("omw-1.4", download_dir=NLTK_DATA_PATH)  # Manually include missing resource
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import json
from dotenv import load_dotenv
import google.generativeai as genai
from nltk_utils import tokenize


# Define safe paths for files
INTENTS_PATH = os.path.join(os.path.dirname(__file__), "intents.json")

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

def reload_intents():
    global intents
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)

# Simple rule-based intent matching
def get_intent_response(msg):
    tokens = tokenize(msg)
    reload_intents()
    for intent in intents["intents"]:
        for pattern in intent.get("patterns", []):
            pattern_tokens = tokenize(pattern)
            if any(token in tokens for token in pattern_tokens):
                if intent.get("responses"):
                    return random.choice(intent["responses"])
    # Fallback to default intent if exists
    for intent in intents["intents"]:
        if intent.get("tag") == "default" and intent.get("responses"):
            return random.choice(intent["responses"])
    return None

# Route message to intent or Gemini
def route_question(msg):
    # Try to match a real intent (not default)
    tokens = tokenize(msg)
    reload_intents()
    for intent in intents["intents"]:
        if intent.get("tag") == "default":
            continue
        for pattern in intent.get("patterns", []):
            pattern_tokens = tokenize(pattern)
            if any(token in tokens for token in pattern_tokens):
                if intent.get("responses"):
                    return random.choice(intent["responses"])
    # No real intent matched, try Gemini
    try:
        result = gemini_model.generate_content([msg])
        if hasattr(result, "text") and result.text:
            return result.text
        elif hasattr(result, "candidates") and result.candidates:
            return result.candidates[0].text
        else:
            return str(result)
    except Exception as e:
        # If Gemini fails, fallback to default intent
        for intent in intents["intents"]:
            if intent.get("tag") == "default" and intent.get("responses"):
                return random.choice(intent["responses"])
        return "Sorry, I couldn't process your request."
# ROUTES
@app.route("/message", methods=["POST"])
def send_message_to_bot():
    data = request.get_json()
    content = data.get("content")
    if not content:
        return jsonify({"error": "Message content is required"}), 400
    try:
        bot_reply = route_question(content)
        return jsonify({"bot_reply": bot_reply})
    except Exception as e:
        print(f"Error in /message endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/add-intent", methods=["POST"])
def add_intent():
    data = request.get_json()
    tag = data.get("tag")
    patterns = data.get("patterns", [])
    responses = data.get("responses", [])
    if not tag or not patterns or not responses:
        return jsonify({"error": "Tag, patterns, and responses are required"}), 400
    # Add new intent to intents.json
    reload_intents()
    intents["intents"].append({
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    })
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(intents, f, ensure_ascii=False, indent=4)
    return jsonify({"success": True, "message": f"Intent '{tag}' added successfully."})

# Health check route for root
@app.route("/", methods=["GET"])
def health():
    return "Backend is running!", 200

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Server running on http://0.0.0.0:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)