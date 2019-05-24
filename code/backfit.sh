#!/bin/bash

#Loop over all the kics and send sbatch commands
#We have 95 stars
for idx in {0..95}; do
  sbatch --export=idx=$idx backfit.bear
done
