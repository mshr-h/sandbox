import onnxruntime as ort
import argparse
import timeit
import numpy as np
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to onnx model")

args = parser.parse_args()
model = args.model

session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

def measure_inference_time(session: ort.InferenceSession, repeat=1):
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    # replace string place holder with 1 (usually 'batch_size')
    input_shape = [1 if isinstance(s, str) else s for s in input_shape]

    x = np.random.random(input_shape)
    x = x.astype(np.float32)

    return timeit.repeat(lambda: session.run([output_name], {input_name: x}), number=1, repeat=repeat)

tseries = measure_inference_time(session, 10)

t_ms = [t * 1000 for t in tseries]

t_mean = statistics.mean(t_ms)
t_max = max(t_ms)
t_min = min(t_ms)
t_median = statistics.median(t_ms)
t_std = statistics.stdev(t_ms)

print(" mean (ms)  median (ms)    max (ms)    min (ms)    std (ms)")
print("{:10.4f}   {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}".format(t_mean, t_median, t_max, t_min, t_std))
