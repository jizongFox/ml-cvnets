from cvnets import get_model
from options.opts import get_training_arguments
from utils.benchmark import pytorch_throughput_latency, onnx_throughput_latency

opts = get_training_arguments(parse_args=True)

model = get_model(opts)
# print(model)
pytorch_throughput_latency(model=model, )

onnx_throughput_latency(model, )
