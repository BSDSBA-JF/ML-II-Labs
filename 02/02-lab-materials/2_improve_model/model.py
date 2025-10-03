from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.pipeline import Pipeline as ImbPipeline

class WhiteBox(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor):
        """
        preprocessor: your ColumnTransformer
        """
        self.preprocessor = preprocessor
        self.pipeline = None

    def make_pipeline(self, model, class_strategy=None):
        """
        Build a pipeline dynamically.

        model: any sklearn classifier
        class_strategy: an imblearn sampler object (e.g., SMOTE(...)) or None
        """
        steps = [('preprocess', self.preprocessor)]
        
        if class_strategy is not None:
            steps.append(('sampler', class_strategy))
        
        steps.append(('model', model))
        
        self.pipeline = ImbPipeline(steps)
        return self.pipeline

    def fit(self, X, y):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call make_pipeline(model, class_strategy) first.")
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call make_pipeline(model, class_strategy) first.")
        return self.pipeline.predict(X)

    def score(self, X, y):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call make_pipeline(model, class_strategy) first.")
        return self.pipeline.score(X, y)
