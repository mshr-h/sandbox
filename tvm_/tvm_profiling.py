import numpy as np
import pytest
from io import StringIO
import csv
import os
import json
import sys

import tvm
import tvm.testing
from tvm.runtime import profiler_vm
from tvm import relay, autotvm
from tvm.relay.testing import mlp
from tvm.contrib.debugger import debug_executor
import tvm.contrib.graph_executor as runtime
from tvm import rpc
from tvm.contrib import utils
from tvm.runtime.profiling import Report
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.graph_tuner import DPTuner

@tvm.testing.parametrize_targets
def test_resnet(target, dev, mode, n_layer):
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    dtype = "float32"
    modelname = "resnet-" + str(n_layer)

    log = "results/%s/%s.log" % (modelname, modelname)
    opt = "results/%s/%s_graph_opt.log" % (modelname, modelname)
    mod, params = relay.testing.resnet.get_workload(
        num_layers=n_layer, batch_size=batch_size, dtype=dtype
    )
    data = np.random.uniform(size=input_shape).astype(dtype)

    with autotvm.apply_graph_best(opt):
        with tvm.transform.PassContext(opt_level=3):
            if mode == "vmprofile":
                exe = relay.vm.compile(mod, target, params=params)
                vm = profiler_vm.VirtualMachineProfiler(exe, dev)
                # report = vm.profile([data], func_name="main", number=100, repeat=3, end_to_end=True)
                report = vm.profile(data=data)
                print(report)
            elif mode == "benchmark":
                lib = relay.build_module.build(mod, target=target, params=params)
                module = runtime.GraphModule(lib["default"](dev))
                module.set_input("data", data)
                print("Evaluate inference time cost...")
                print(module.benchmark(dev, number=100, repeat=3)) #, end_to_end=True))
            elif mode == "grprofile":
                exe = relay.build(mod, target, params=params)
                gr = debug_executor.create(exe.get_graph_json(), exe.lib, dev)
                report = gr.profile(data=data)
                print(report)

if __name__ == "__main__":
    layers = 0
    mode = "none"

    if len(sys.argv) != 3:
        print("Usage: python test_runtime_profiling.py [operators] [resnet_layers]")
    else:
        mode = sys.argv[1]
        layers = int(sys.argv[2])

    if (layers == 18 or layers == 50):
        print("Mode: " + mode)
        print("Model: ResNet-" + str(layers))
        test_resnet("llvm", tvm.cpu(), mode, layers)