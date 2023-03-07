#!/bin/bash
for alpha in 0.00 0.05 0.10 0.20 0.50 1.00 #10.00 100.00
do
   for type in random average average_output
   do
        python entrypoints/benchmark-server.py --type $type --alpha $alpha --models_dir h5_weak_learners_3 --results_dir results
   done
done
