from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import ShuffleSplit

class SVM:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]

    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """
        # model = SVC(kernel = 'rbf', C = test_C)
        # sc = StandardScaler()
        # X = sc.fit_transform(self.X)
        # cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        # scores = cross_val_score(model, X, self.y, cv=cv)
        # print(scores)
        # print(test_C, ':', "%0.2f Accuracy %0.2f Standard Deviation" % (scores.mean(), scores.std()))

        preprocessor = ('standard_scaler', StandardScaler())
        model = ('svc_model', SVC(kernel = 'rbf', C = 4, gamma = 'scale'))

        pipeline = Pipeline([preprocessor, model])
        pipeline.fit(self.X, self.y)
        
        return pipeline
        

