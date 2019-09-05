#!/bin/bash

#Loop over all the kics and send sbatch commands
#We have 95 stars
for idx in 0, 3, 73, 80, 82, 83, 89, 93, 94; do
  sbatch --export=idx=$idx backfit.bear
done


#sbatch --export=idx=71 backfit.bear
