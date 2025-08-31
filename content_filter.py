import os
try:
    import google.generativeai as genai
    Genai_AVAILABLE = True
except ImportError:
    Genai_AVAILABLE = False
    print("Gemini is not available make sure to install requirements.txt")

try:
    from transformers import pipeline
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("NLP is not available make sure to install environment.yml")

class ContentFilter:
    def __init__(self,UseGenai=False,UseNLP=False):
        self.UseNLP = UseNLP if NLP_AVAILABLE else False
        self.UseGenai = UseGenai if Genai_AVAILABLE else False

        if self.UseGenai:
            api_key = os.getenv("MY_API_KEY")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        if self.UseNLP:
            self.moderator = pipeline(
                "text-classification",
                model="unitary/toxic-bert",   # fine-tuned BERT for toxicity
                return_all_scores=True
            )
        self.bad_words = {"קללה", "****", "טיפש", "מטומטם","nigga","fuck","ass","nigger"}

    def nlp_moderate_text(self, message: str, threshold: float = 0.5) -> dict:
        """
        NLP-based moderation using Hugging Face toxic-bert model.
        Returns category scores and safe/unsafe flag.
        """
        results = self.moderator(message)[0]

        flags = {}
        unsafe = False

        for res in results:
            label = res["label"].lower()
            score = res["score"]
            flags[label] = round(score, 3)

            if score >= threshold and label != "neutral":
                unsafe = True

        return {
            "text": message,
            "scores": flags,
            "safe": not unsafe
        }
    def is_message_clean(self, message):

        if message in self.bad_words:
            return False
        if self.UseNLP:
            if self.nlp_moderate_text(message):
                return False
        if self.UseGenai:
            try:
                response = self.model.generate_content(
                    f"Is this message inappropriate, offensive, or toxic? '{message}' Answer only yes or no."
                )

                return "no" in response.text.lower()
            except Exception as e:
                print(f"Gemini error: {e}")
                return True  # Fallback to allow message if Gemini fails
        return True
