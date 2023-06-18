#!/bin/bash
pip install wheel
TORCH_CUDA_ARCH_LIST="8.0 9.0" \
pip install --no-build-isolation flash-attn==1.0.7