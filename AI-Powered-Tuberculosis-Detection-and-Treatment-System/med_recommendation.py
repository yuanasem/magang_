import logging
import google.generativeai as genai

# Direct API key usage (for testing/dev ONLY)
API_KEY = "AIzaSyDhf6yIIWlL7OR5Ix4H2X1VQeDEP94zuoE"  # 🔐 Replace with your Gemini API key

# Configure Gemini
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing.")
genai.configure(api_key=API_KEY)
logging.basicConfig(level=logging.INFO)

# TB knowledge base - kept for possible future use or extension
MEDICAL_KNOWLEDGE_BASE = [
    {
        "drug": "Isoniazid (INH)",
        "uses": "First-line treatment for all forms of tuberculosis. Prevents TB in people exposed to it.",
        "dosage": "5 mg/kg daily or 15 mg/kg two-three times weekly",
        "side_effects": "Hepatitis, peripheral neuropathy",
        "notes": "Often used in combination with other drugs."
    },
    {
        "drug": "Rifampicin (RIF)",
        "uses": "First-line treatment for all forms of tuberculosis. Effective against both active and latent TB.",
        "dosage": "10 mg/kg daily",
        "side_effects": "Hepatitis, orange discoloration of urine/tears",
        "notes": "Strong inducer of cytochrome P450 enzymes, leading to drug interactions."
    },
    {
        "drug": "Pyrazinamide (PZA)",
        "uses": "First-line treatment for tuberculosis, especially in the initial intensive phase.",
        "dosage": "25 mg/kg daily",
        "side_effects": "Hepatitis, hyperuricemia, arthralgia",
        "notes": "Crucial for shortening treatment duration."
    },
    {
        "drug": "Ethambutol (EMB)",
        "uses": "Prevents drug resistance when used in combination.",
        "dosage": "15–25 mg/kg daily",
        "side_effects": "Optic neuritis (reversible)",
        "notes": "Use with caution in renal impairment."
    }
]

class MedicationRecommender:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Gemini model: {e}")
            raise

    def recommend_medication(self, diagnosis_info: str) -> str:
        # For now, returning fixed plain-text recommendation for "severe" diagnosis.
        # You can extend this to use model.generate_content if needed.
        if "severe" in diagnosis_info.lower():
            return (
                "Based on the diagnosis of severe Tuberculosis, the standard of care involves a multi-drug regimen to effectively treat the infection and prevent the development of drug resistance. "
                "The recommended initial treatment is a combination of the following four first-line drugs:\n\n"
                "Recommended Medication Regimen\n\n"
                "1. Isoniazid (INH)\n"
                "   Uses: First-line treatment for all forms of tuberculosis. Prevents TB in people exposed to it.\n"
                "   Dosage: 5 mg/kg daily or 15 mg/kg two-three times weekly\n"
                "   Side Effects: Hepatitis, peripheral neuropathy\n"
                "   Notes: Often used in combination with other drugs.\n\n"
                "2. Rifampicin (RIF)\n"
                "   Uses: First-line treatment for all forms of tuberculosis. Effective against both active and latent TB.\n"
                "   Dosage: 10 mg/kg daily\n"
                "   Side Effects: Hepatitis, orange discoloration of urine/tears\n"
                "   Notes: Strong inducer of cytochrome P450 enzymes, leading to drug interactions.\n\n"
                "3. Pyrazinamide (PZA)\n"
                "   Uses: First-line treatment for tuberculosis, especially in the initial intensive phase.\n"
                "   Dosage: 25 mg/kg daily\n"
                "   Side Effects: Hepatitis, hyperuricemia, arthralgia\n"
                "   Notes: Crucial for shortening treatment duration.\n\n"
                "4. Ethambutol (EMB)\n"
                "   Uses: Prevents drug resistance when used in combination.\n"
                "   Dosage: 15–25 mg/kg daily\n"
                "   Side Effects: Optic neuritis (reversible)\n"
                "   Notes: Use with caution in renal impairment.\n\n"
                "Summary Note:\n"
                "This four-drug combination (often abbreviated as RIPE) is the standard regimen for the initial intensive phase of treatment for severe tuberculosis. "
                "Treatment must be prescribed and monitored by a qualified healthcare provider to manage side effects and ensure patient-specific dosing."
            )
        else:
            # You can extend this for other severities or use AI to generate
            return "Medication recommendation is currently only available for severe tuberculosis diagnosis."

# For testing
if __name__ == "__main__":
    recommender = MedicationRecommender()
    test_diag = "Severe Tuberculosis diagnosis"
    print(recommender.recommend_medication(test_diag))

