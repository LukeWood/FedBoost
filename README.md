# FedBoost

# Quickstart

Step 1.) install the `fed_boost` package:
```
python setup.py develop
```

Step 2.) train the weak learners:

```
python entrypoint/train-weak-learners.py --model_path='weak_learners/'
```

Step 3.) benchmark a server:
```
python entrypoints/benchmark-server.py --type random --alpha 0.00 --models_dir weak_learners
```

Step 4.) benchmark all servers:
```
./entrypoints/benchmark-all-servers.sh
```

# References

- https://vision.cornell.edu/se3/wp-content/uploads/2016/08/boosted-convolutional-neural-1.pdf
