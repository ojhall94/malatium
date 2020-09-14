#!/bin/bash

# Run our model on all stars

for idx in  27 28 81; do
    # echo $idx
    sbatch --export=idx=$idx peakbag.bear
done
    
