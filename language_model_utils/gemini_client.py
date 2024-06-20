from signatures.ecommerce.ecommerce_classification_signature import EcommerceClassificationSignature

import dspy

class GeminiClient(dspy.Module):
    def __init__(self, dspy_signature: type[dspy.Signature]) -> None:
        # Gemini secret key: AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8
        # Set up the LM
        super().__init__()
        self.dspy_signature = dspy_signature
        gemini = dspy.Google("models/gemini-1.5-flash",
                         api_key='AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8')
        dspy.settings.configure(lm=gemini, max_tokens=1024)
        self.prog = dspy.ChainOfThought(dspy_signature)

    def forward(self, input_dict: dict) -> dspy.Prediction:
        # ** unpacking dictionary
        for key in input_dict:
            input_dict[key] = str(input_dict[key])
        return self.prog(**input_dict)