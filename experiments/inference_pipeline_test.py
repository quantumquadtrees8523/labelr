# script.py
from data_types.pre_inference_set import PreInferenceSet
from inference_pipeline import InferencePipeline
import dspy

"""
Goal of this test is to get to the point where we can generalize signatures. We want to be able to
convert all prompt outputs to a `llm_inference` and `llm_rationale` fields. We want to make sure
these outputs look clean. 

`llm_inference` must be massagable into a primitive type or string.
`llm_rationale` must be a clean string and limited to a certain size.
"""
# from signatures.ecommerce.ecommerce_classification_signature import EcommerceClassificationSignature

# Current design choice is to use different inference pipelines for each signature. Multi-process it.
# You will still be bottlenecked by model builder api rate limiting. 
class ReviewSentimentSignature(dspy.Signature):
    """Classify review from unhappiest 1 to happiest 5."""
    Review = dspy.InputField()
    llm_inference = dspy.OutputField(desc="Classify Review from unhappiest 1 to happiest 5.")

class EcommerceClassificationSignature(dspy.Signature):
    """Classify the text based on the provided context."""
    text = dspy.InputField(desc="product text.")
    llm_inference = dspy.OutputField(desc="category of product among 'Household', 'Books', 'Clothing & Accessories', 'Electronics' based on text.")

def main():
    # input_filename = '/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'
    InferencePipeline(PreInferenceSet('/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'), ReviewSentimentSignature) #.run()

    # input_filename = '/Users/suryaduggirala/projects/llm-labeling/datasets/ecommerce/ecommerceDataset.csv'
    InferencePipeline(PreInferenceSet('/Users/suryaduggirala/projects/llm-labeling/datasets/ecommerce/ecommerceDataset.csv'), EcommerceClassificationSignature) #.run()

if __name__ == "__main__":
    main()