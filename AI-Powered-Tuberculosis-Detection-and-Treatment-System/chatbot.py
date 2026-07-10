import logging
import os
import google.generativeai as genai

# Load the Gemini API key from an environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Environment variable 'GEMINI_API_KEY' is missing.")
genai.configure(api_key=API_KEY)
logging.basicConfig(level=logging.INFO)

class TBMedicalAssistant:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Gemini model: {e}")
            raise

    def get_response(self, message: str) -> str:
        prompt = (
            "You are a knowledgeable tuberculosis medical assistant. "
            "Answer questions clearly and concisely.\n\n"
            f"User: {message}\nAssistant:"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"❌ Error generating response: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."

# Simple test run
if __name__ == "__main__":
    assistant = TBMedicalAssistant()
    print(assistant.get_response("What are the symptoms of tuberculosis?"))
