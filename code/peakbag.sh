#!/bin/bash

# Run our model on all stars

for idx in {0..94}; do
  sbatch --export=idx=$idx peakbag_run.bear
done
