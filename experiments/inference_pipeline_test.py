# script.py
import time
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
# class ReviewSentimentSignature(dspy.Signature):
#     """Classify review from unhappiest 1 to happiest 5."""
#     Review = dspy.InputField()
#     llm_inference = dspy.OutputField(desc="Classify Review from unhappiest 1 to happiest 5.")

# class EcommerceClassificationSignature(dspy.Signature):
#     """Classify the text based on the provided context."""
#     text = dspy.InputField(desc="product text.")
#     llm_inference = dspy.OutputField(desc="category of product among 'Household', 'Books', 'Clothing & Accessories', 'Electronics' based on text.")

# class ResumeClassificationSignature(dspy.Signature):
#     """Classify the resume based on the provided context."""
#     Resume = dspy.InputField(desc="Resume text.")
#     llm_inference = dspy.OutputField(desc="single specialty of resume among broad skill categories based on text. For example, Data Science or Sales.")    

def main():
    restaurants_dataset = '/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'
    # there should be a way to 
    ip_1 = InferencePipeline(PreInferenceSet(restaurants_dataset), output_filename="cuisine", output_description="the cuisine of the restaurant referenced.").run()
    ip_2 = InferencePipeline(PreInferenceSet(restaurants_dataset), task="extract", output_filename="foods", output_description="the foods that are referenced or an empty string ('')").run()
    pass
    # start_time = time.time()
    # # input_filename = '/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'
    # InferencePipeline(PreInferenceSet('/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'), ReviewSentimentSignature).run()
    # print(time.time() - start_time)

    # start_time = time.time()
    # # input_filename = '/Users/suryaduggirala/projects/llm-labeling/datasets/ecommerce/ecommerceDataset.csv'
    # InferencePipeline(PreInferenceSet('/Users/suryaduggirala/projects/llm-labeling/datasets/ecommerce/ecommerceDataset.csv'), EcommerceClassificationSignature).run()
    # print(time.time() - start_time)

    # start_time = time.time()
    # InferencePipeline(PreInferenceSet('/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/UpdatedResumeDataSet.csv'), ResumeClassificationSignature).run()
    # print(time.time() - start_time)

if __name__ == "__main__":
    main()