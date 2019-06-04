#!/bin/bash

#Loop over all the kics and send sbatch commands
#We have 95 stars
# for idx in {0..95..1}; do
#   sbatch --export=idx=$idx backfit.bear
# done


for idx in {7,8,10,11,12,13,14,15,17,18,71,72,93}; do
  # echo $idx
  sbatch --export=idx=$idx backfit.bear
done
