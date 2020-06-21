# generalization
Back up code for my Medium post discussing how to obtain bounds on test loss using unlabeled data. Running the file dense_NN.py trains a fully connected two-layer neural network on synthetic data and saves its predictions on both the training set and the unlabeled set. Running image_classification.py does the same for a CNN by copying the code from the [tensorflow image classification tutorial](https://www.tensorflow.org/tutorials/images/classification). Assessment of overfitting is then done with the Gen class contained in gen.py. Basic usage is

```
g = Gen('training_predictions.txt', 'testing_predictions.txt')
g = Gen.summary('training_plot_name.png', 'testing_plot_name.png', bins=20)
```

The summary() method creates the training and testing histograms according to the given filenames and reports a variety of bounds and estimates calculated from the input files training_predictions.txt and testing_predictions.txt
