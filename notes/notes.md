# Notes

my notes in updating the code.

## Early Stopping

First I noticed in training that the models converge super fast (like 1-3~ epochs).  
We were training for 100 epochs no matter what.
This is a huge waste of resources, so I added:

```python
callbacks = [
    keras.callbacks.EarlyStopping(patience=3)
]
self.model.fit(
    self.train_images,
    to_categorical(self.train_labels, output_class_size),
    epochs=self.client_epochs,
    validation_data=(
        self.test_images,
        to_categorical(self.test_labels, output_class_size),
    ),
    callbacks=callbacks
)
```

to `Client.train_model()`.  This will stop the training when the validation loss stops going down.
