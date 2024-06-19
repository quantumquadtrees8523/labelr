import dspy

class EcommerceClassificationSignature(dspy.Signature):
    """Classify product descriptions among 'Household', 'Books', 'Clothing & Accessories', 'Electronics'."""
    description = dspy.InputField()
    category = dspy.OutputField()