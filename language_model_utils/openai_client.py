from dspy.teleprompt import BootstrapFewShot
import dspy

class OpenAIClient(dspy.Module):
    def __init__(self):
        # OpenAI secret key: sk-proj-NR0mObca3fqtesQKvZtoT3BlbkFJKBBxNqa77UWrQQku7nuV
        # Set up the LM
        super().__init__()
        turbo = dspy.OpenAI(api_key="sk-proj-NR0mObca3fqtesQKvZtoT3BlbkFJKBBxNqa77UWrQQku7nuV", model='gpt-3.5-turbo-instruct', max_tokens=250)
        dspy.settings.configure(lm=turbo)
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)