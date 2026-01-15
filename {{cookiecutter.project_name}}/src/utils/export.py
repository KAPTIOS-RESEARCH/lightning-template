import torch
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_onnx(
    model,
    example_input,
    output_path,
    opset=17,
):
    model.eval()
    model.cpu()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    return output_path

def export_quantized_onnx(fp32_onnx_path, output_path):
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )