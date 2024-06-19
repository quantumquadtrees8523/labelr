import dspy

class EcommerceClassificationSignature(dspy.Signature):
    """Classify product text among 'Household', 'Books', 'Clothing & Accessories', 'Electronics'."""
    text = dspy.InputField()
    category = dspy.OutputField()