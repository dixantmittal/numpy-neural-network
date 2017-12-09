import numpy as np
import sklearn.datasets as dt

Data, Y = dt.make_classification(n_samples=100000, n_features=30, n_redundant=0)

np.save("/Users/dixantmittal/Downloads/X", Data)
np.save("/Users/dixantmittal/Downloads/y", Y)
