from signatures.ecommerce.ecommerce_classification_signature import EcommerceClassificationSignature

import dspy

class GeminiClient(dspy.Module):
    def __init__(self, dspy_signature: dspy.Signature):
        # Gemini secret key: AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8
        # Set up the LM
        super().__init__()
        self.dspy_signature = dspy_signature
        gemini = dspy.Google("models/gemini-1.5-flash",
                         api_key='AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8')
        dspy.settings.configure(lm=gemini, max_tokens=1024)
        self.prog = dspy.ChainOfThought(dspy_signature)

    def forward(self, description):
        return self.prog(description=description)