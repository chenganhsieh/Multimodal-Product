#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
    exit
fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python $(dirname "$0")/main.py  $CONFIG --work-dir=${WORK_DIR} ${@:4}
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/main.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}
fi
