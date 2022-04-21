import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

import hummingbird.ml
import onnxruntime as ort

import timeit


X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)  # make sure to use fp32 input


def test_logreg():
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    np.testing.assert_equal(model.predict(X), tvm_model.predict(X))


def test_rf():
    model = RandomForestClassifier(max_depth=8)
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    np.testing.assert_equal(model.predict(X), tvm_model.predict(X))


def bench():
    X, y = fetch_california_housing(return_X_y=True)  # input shape: (20640, 8)
    X = X.astype(np.float32)  # make sure to use fp32 input

    model = RandomForestRegressor(max_depth=8, n_estimators=250)
    model.fit(X, y)

    initial_type = [('float_input', FloatTensorType([None, 8]))]
    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type).SerializeToString()

    ort_sess = ort.InferenceSession(onnx_model)
    input_name = ort_sess.get_inputs()[0].name
    label_name = ort_sess.get_outputs()[0].name

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    loop = 20
    res_sk = timeit.timeit(lambda: model.predict(X), number=loop)
    res_ort = timeit.timeit(lambda: ort_sess.run([label_name], {input_name: X}), number=loop)
    res_tvm = timeit.timeit(lambda: tvm_model.predict(X), number=loop)

    print(res_sk, res_ort, res_tvm)


# test_logreg()
# test_rf()
bench()
