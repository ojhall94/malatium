#!/bin/bash

# Run our model on all stars
#for idx in {0..94}; do
#    sbatch --export=idx=$idx peakbag.bear
#done
for idx in  0,  1,  2,  3,  5,  6,  7,  8,  10, 11, 13, 14, 15, 16,
            19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 
            35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 49, 51,
            53, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 70,
            71, 72, 74, 77, 78, 79, 81, 82, 85, 87, 89, 90, 91,
            93; do
    sbatch --export=idx=$idx peakbag.bear
done
    
