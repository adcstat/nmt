#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 vocab_size param_config model_config checkpoint"
    exit 1
fi

# Assign command line arguments to variables
vocab_size=$1
param_config=$2
model_config=$3
checkpoint=$4

# Arrays of split and beam_width values
splits=("validation" "test")
beam_widths=(3 4 5)

# Iterate over all combinations of split and beam_width
for split in "${splits[@]}"; do
    for beam_width in "${beam_widths[@]}"; do
        echo "Running with split=$split and beam_width=$beam_width"
        python get_bleu_score.py --vocab_size $vocab_size \
                              --param_config $param_config \
                              --model_config $model_config \
                              --checkpoint $checkpoint \
                              --split $split \
                              --beam_width $beam_width
    done
done
