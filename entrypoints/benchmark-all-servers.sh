#!/bin/bash
for alpha in 0.05 0.10 0.20 0.50 1.00 10.00 100.00
do
   for type in random average gdboost
   do
        python entrypoints/benchmark-server.py --type $type --alpha $alpha --models_dir weak_learners --results_dir results
   done
done
