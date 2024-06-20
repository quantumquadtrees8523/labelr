# script.py
import time

from elasticsearch import Elasticsearch
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

def main():
    # es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    restaurants_dataset = '/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/Restaurant_Reviews.tsv'
    # there should be a way to 
    restaurants_ip_1 = InferencePipeline(PreInferenceSet(restaurants_dataset), output_filename="cuisine", output_description="single word cuisine or an empty string ('') if you don't know.").run()
    restaurants_ip_2 = InferencePipeline(PreInferenceSet(restaurants_dataset), task="extract", output_filename="foods", output_description="comma separated list of foods that are referenced or an empty string ('') if you don't know.").run()
    restaurants_ip_3 = InferencePipeline(PreInferenceSet(restaurants_dataset), task="extract", output_filename="names", output_description="comma separated list of names that are referenced or an empty string ('') if you don't know.").run()

    resumes_dataset = '/Users/suryaduggirala/projects/llm-labeling/datasets/inference_pipeline_tests/UpdatedResumeDataSet.csv'
    resumes_ip_1 = InferencePipeline(PreInferenceSet(resumes_dataset), task="extract", output_filename="technology", output_description="comma separated list of technologies or an empty string ('') if you don't know.").run()
    resumes_ip_2 = InferencePipeline(PreInferenceSet(resumes_dataset), task="classify", output_filename="business_analytics", output_description="(True/False) whether the resume has the requisite skills for business analytics or an empty string ('') if you don't know.").run()
    resumes_ip_3 = InferencePipeline(PreInferenceSet(resumes_dataset), task="extract", output_filename="names", output_description="comma separated list of names that are referenced or an empty string ('') if you don't know.").run()

if __name__ == "__main__":
    main()