#!/bin/bash

# Run our model on all stars

for idx in  1 23; do
    # echo $idx
    sbatch --export=idx=$idx peakbag.bear
done
    
