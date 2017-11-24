#!/bin/bash

for OPTIMIZER in Ftrl ProximalGradientDescent Adelta Adagrad Adam RMSProp
do
    for LEARN_RATE in 0.0001 0.001 0.01 0.1
    do
        for BATCH_SIZE in 32 64 256 512
        do

            #!!! Job name must be < 15 characters, first one must be alphabetic
	        echo "Running ith ${OPTIMIZER}, ${LEARN_RATE}, ${BATCH_SIZE}"

            CMD="python3 util.py -o ${OPTIMIZER} -l ${LEARN_RATE} -b ${BATCH_SIZE} -e 20 -f small_model.txt  -m ex04.small_model -t ex04.exercise5"
            ${CMD}
        done
    done
done
