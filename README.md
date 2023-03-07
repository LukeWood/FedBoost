# FedBoost

# Quickstart

Step 0.) clone

```
git clone https://github.com/LukeWood/FedBoost/
cd FedBoost/
```

Step 1.) install the `fed_boost` package:
```
python setup.py develop
```

Step 2.) train the weak learners:

```
python entrypoints/train-weak-learners.py --model_path='h5_weak_learners_3'
```

Step 3.) test the weak learners:

```
python entrypoints/test-weak-learners.py --model_path='h5_weak_learners_3'
```

Step 4.) benchmark a server:
```
python entrypoints/benchmark-server.py --type random --alpha 0.00 --models_dir h5_weak_learners_3 --results_dir results
```

Step 5.) benchmark all servers:
```
./entrypoints/benchmark-all-servers.sh
```

# References

- https://vision.cornell.edu/se3/wp-content/uploads/2016/08/boosted-convolutional-neural-1.pdf
