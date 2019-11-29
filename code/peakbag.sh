#!/bin/bash

# Run our model on all stars

for idx in  1 3 10 21 23 27 28 29 30 32 36 47 55 57 61 62 63 67 74 76 80 81 83 84 88 89 94; do
    # echo $idx
    sbatch --export=idx=$idx peakbag.bear
done
    
