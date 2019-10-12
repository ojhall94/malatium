#!/bin/bash

# Run our model on all stars
#for idx in {0..94}; do
#    sbatch --export=idx=$idx peakbag.bear
#done
for idx in  0, 2, 3, 4, 9, 12, 17, 18, 24, 29, 30, 34, 43, 44, 50, 52, 54, 56, 66, 68, 69, 73, 75, 86, 92; do
    sbatch --export=idx=$idx peakbag.bear
done
    
