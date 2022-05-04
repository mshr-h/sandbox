import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

import hummingbird.ml

from tvm import relay
from tvm.driver import tvmc

# Create trained model
X, y = fetch_california_housing(return_X_y=True)  # input shape: (20640, 8)
X = X.astype(np.float32)  # make sure to use fp32 input

model = RandomForestRegressor(max_depth=8, n_estimators=100)
model.fit(X, y)

# Convert sklearn model to PyTorch model
sample_input = np.empty([1, 8]).astype(np.float32) # required to fix input shape
pt_model = hummingbird.ml.convert(model, "torch.jit", sample_input)

# Load PyTorch model to TVM
shape_dict={"input_0": sample_input.shape}
input_shapes = list(shape_dict.items())
mod, params = relay.frontend.from_pytorch(pt_model.model, input_shapes)

# Convert to TVMCModel
tvmc_model = tvmc.model.TVMCModel(mod, params)

# Compile and export package
tvmc_package = tvmc.compile(tvmc_model, target="llvm", output_format="tar", package_path="california_housing_tvmc.tar", pass_context_configs=["relay.FuseOps.max_depth=50"])

# Run inference
# python -m tvm.driver.tvmc run --fill-mode random --print-time --repeat 200 california_housing_tvmc.tar
