#!/usr/bin/env bash

if [[ "$*" == *"--sophie"* ]]; then
    echo "Sophie environment detected! Installing custom CUDA/Torch drivers..."
    source ./env/bin/activate
    pip install huggingface_hub
    pip install -r ./build_scripts/build_qwen_2_5_vl_xb_instruct__requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
else
    echo "Standard environment. Installing default drivers..."
    source ./env/bin/activate
    pip install huggingface
    # pip install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils
fi
