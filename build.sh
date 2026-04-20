#!/usr/bin/env bash

python3 -m venv env
source env/bin/activate

MERLIN_USER="xvodra01"
DATASET_FILE_NAME="people_gator__data_export.zip"

# uncomment if dataset download and parse is needed 
#     |
#     |
#     V

# # download dataset
# mkdir -p ./data
# # scp "${MERLIN_USER}@merlin.fit.vutbr.cz:/mnt/matylda1/ivasko/Datasets/${DATASET_FILE_NAME}" ./data/

# # unzip the dataset into a wrapper folder of the same name inside ./data
# unzip "./data/${DATASET_FILE_NAME}" -d "./data/${DATASET_FILE_NAME%.zip}"

# # parse dataset
# python3 ./scripts/parse_dataset.py

MODEL=""
SOPHIE_ARG=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -m)
      MODEL="$2"
      shift 2
      ;;
    --sophie)
      if [[ "$2" == "true" || "$2" == "false" ]]; then
        SOPHIE_ARG="--sophie $2"
        shift 2
      else
        SOPHIE_ARG="--sophie"
        shift 1
      fi
      ;;
    *)
      echo "Unknown option passed: $1" >&2
      exit 1
      ;;
  esac
done

if [ -n "$MODEL" ]; then
  echo "model: $MODEL"
  
  case "$MODEL" in
    "qwen_2_5_vl_3b_instruct")
      echo "running build script for $MODEL"
      bash ./build_scripts/build_qwen_2_5_vl_xb_instruct.sh $SOPHIE_ARG
      hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen_2_5_vl_3b_instruct
      ;;
      
    "qwen_2_5_vl_7b_instruct")
      echo "running build script for $MODEL"
      bash ./build_scripts/build_qwen_2_5_vl_xb_instruct.sh $SOPHIE_ARG
      hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/qwen_2_5_vl_7b_instruct
      ;;

    "qwen_2_5_vl_32b_instruct")
      echo "running build script for $MODEL"
      bash ./build_scripts/build_qwen_2_5_vl_xb_instruct.sh $SOPHIE_ARG
      hf download Qwen/Qwen2.5-VL-32B-Instruct --local-dir ./models/qwen_2_5_vl_32b_instruct
      ;;
    *)
      echo "Warning: No build script mapped for model '$MODEL'"
      ;;
  esac
else
  echo "No -m flag provided. Skipping model-specific build steps."
fi

echo "installing base requirements"
pip install -r requirements.txt
pip install -e .

echo "done"
# OPTIONAL: rm dataset .zip