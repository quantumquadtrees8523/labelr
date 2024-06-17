import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Set up the LM
turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)