# HamQ-ResNet-CIFAR10

Welcome to the HamQ-ResNet-CIFAR10 repository! This project is based on the paper "HamQ: Hamming Weight-based Energy Aware Quantization for Analog Compute-In-Memory Accelerator in Intelligent Sensors" [Link](https://ieeexplore.ieee.org/abstract/document/10489839), which introduces a novel regularizer, HamQ, to enhance the energy efficiency of analog Compute-In-Memory (CIM) accelerators used in machine learning tasks. Our codebase includes a complete setup for training a ResNet model on the CIFAR-10 dataset using the proposed regularizer.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{sharma2024hamq,
  title={HamQ: Hamming Weight-based Energy Aware Quantization for Analog Compute-In-Memory Accelerator in Intelligent Sensors},
  author={Sharma, Sudarshan and Kang, Beomseok and Kidambi, Narasimha Vasishta and Mukhopadhyay, Saibal},
  journal={IEEE Sensors Journal},
  year={2024},
  publisher={IEEE}
}
```

## Project Overview

HamQ (Hamming weight-based Quantization) is a technique developed to reduce the energy consumption of analog CIM accelerators by implementing a regularizer that minimizes the Hamming weight of quantized model weights. This repository contains Python scripts to train a ResNet model on the CIFAR-10 dataset, demonstrating how HamQ can be integrated into a deep learning training pipeline.



## Files in this Repository

- `train.py`: The main script to start the training/evaluation process.
- `net.py`: Defines the ResNet architecture modified to include HamQ.
- `utils.py`: Helper functions for training and data processing.
- `utils_en.py`: Additional utilities for energy calculations and logging.
- `map.py`: Handles the mapping of simulated bitline energy to actual energy.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ and PyTorch installed. You can install all dependencies via:

```bash
pip install -r requirements.txt
```