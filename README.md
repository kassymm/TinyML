# TinyML: Compact CNN for CIFAR-10 with ONNX Quantization

A compact convolutional neural network designed for CIFAR-10 classification under strict size constraints, exported to ONNX and quantized to INT8 for efficient CPU inference.

## Results

| Metric | FP32 (PyTorch) | FP32 (ONNX) | INT8 Static | INT8 Dynamic |
|--------|----------------|-------------|-------------|--------------|
| Size (KB) | 318.3 | 325.2 | 176.0 | 169.9 |
| Accuracy (%) | 91.16 | 91.16 | 90.91 | 90.92 |
| Inference (ms/batch) | 244.46 | 25.73 | 22.81 | 321.20 |

## Architecture

**TinyNet** — a MobileNetV2-inspired network with 78,630 parameters (~318 KB in FP32).

```
Stem:      Conv2d(3→16, 3×3) + BN + ReLU6
Block 1:   InvResSE(16→24,  mid=48,  stride=1)
Block 2:   InvResSE(24→24,  mid=48,  stride=1)  [residual]
Block 3:   InvResSE(24→32,  mid=72,  stride=2)
Block 4:   InvResSE(32→32,  mid=72,  stride=1)  [residual]
Block 5:   InvResSE(32→48,  mid=96,  stride=2)
Block 6:   InvResSE(48→48,  mid=96,  stride=1)  [residual]
Block 7:   InvResSE(48→64,  mid=128, stride=2)
Head:      GlobalAvgPool → Dropout(0.1) → Linear(64→10)
```

Key components:
- **Depthwise separable convolutions** — reduce parameters by ~8–9× vs standard convolutions
- **Squeeze-and-Excitation (SE) blocks** — channel attention with reduction=4
- **Inverted residual structure** — expand → depthwise → project (linear bottleneck)
- **ReLU6 activation** — bounded activations for clean ONNX export and quantization

## Project Structure

```
├── Kassym_Mukhanbetiyar.ipynb              # Main notebook (all code)
├── Kassym_Mukhanbetiyar_requirements.txt   # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r Kassym_Mukhanbetiyar_requirements.txt
```

## Usage

Run all cells in the notebook sequentially:

```bash
jupyter notebook Kassym_Mukhanbetiyar.ipynb
```

The notebook is organized into four sections:

1. **2.1** — Model design and training (150 epochs, SGD + CosineAnnealingLR)
2. **2.2** — ONNX export (opset 13) and ONNXRuntime inference comparison
3. **2.3** — INT8 static quantization (QDQ format, MinMax calibration)
4. **2.4** — INT8 dynamic quantization (bonus)


