import dspy

class GeminiAgent(dspy.Module):
    def __init__(self):
        # Gemini secret key: AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8
        # Set up the LM
        super().__init__()
        gemini = dspy.Google("models/gemini-1.5-flash",
                         api_key='AIzaSyD0uKjqm0mzflAQkMlhnihPymXg9-c2UO8')
        dspy.settings.configure(lm=gemini, max_tokens=1024)
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)