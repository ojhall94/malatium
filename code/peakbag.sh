#!/bin/bash

# Run our model on all stars
for idx in {0..5}; do
    sbatch --export=idx=$idx peakbag.bear
done
#for idx in {81..94}; do
#    sbatch --export=idx=$idx peakbag.bear
#done
