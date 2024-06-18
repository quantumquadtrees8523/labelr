from active_learner import ActiveLearner
from ingestion import IngestionPipeline

pipeline = IngestionPipeline('datasets/data/ecommerceDataset.csv')
# learner = ActiveLearner(1, "what color is the sky?")
# learner.learn()