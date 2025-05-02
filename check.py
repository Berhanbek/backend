import google.generativeai as genai
from google.generativeai import types
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")
contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hello Gemini!")])]
response = gemini_model.generate_content(contents)
print(response.text)