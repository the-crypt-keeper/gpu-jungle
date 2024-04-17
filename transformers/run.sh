#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo config $2
cat $2
sleep 5
