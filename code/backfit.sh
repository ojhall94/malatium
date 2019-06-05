#!/bin/bash

#Loop over all the kics and send sbatch commands
#We have 95 stars
# for idx in {0..95..1}; do
#   sbatch --export=idx=$idx backfit.bear
# done


for idx in {57,93}; do
  # echo $idx
  sbatch --export=idx=$idx backfit.bear
done
