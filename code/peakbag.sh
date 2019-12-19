#!/bin/bash

# Run our model on all stars

for idx in  62 63; do
    # echo $idx
    sbatch --export=idx=$idx peakbag.bear
done
    
