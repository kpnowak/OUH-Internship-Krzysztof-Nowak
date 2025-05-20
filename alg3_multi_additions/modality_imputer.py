import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ModalityImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for handling missing values in multi-modal data."""
    
    def __init__(self):
        self.means_ = None
        
    def fit(self, X):
        """Compute the mean values for imputation."""
        self.means_ = np.nanmean(X, axis=0)
        return self
        
    def transform(self, X):
        """Impute missing values with mean values."""
        if self.means_ is None:
            raise ValueError("ModalityImputer has not been fitted yet.")
        return np.nan_to_num(X, nan=self.means_)
        
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X) 