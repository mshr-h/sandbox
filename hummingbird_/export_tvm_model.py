import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

import hummingbird.ml

X, y = fetch_california_housing(return_X_y=True)  # input shape: (20640, 8)
X = X.astype(np.float32)  # make sure to use fp32 input

model = RandomForestRegressor(max_depth=8, n_estimators=250)
model.fit(X, y)

sample_input = np.empty([1, 8])
tvm_model = hummingbird.ml.convert(model, "tvm", sample_input)

tvm_model.save("tvm_model")
# -> model will be saved to "tvm_model.zip"

