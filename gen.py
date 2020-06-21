import heapq
import pickle
from math import log
import matplotlib.pyplot as plt
import tensorflow as tf


class Gen:
    def __init__(self, training_input, testing_input, task='None'):
        self.training_input = training_input
        self.testing_input = testing_input
        self.task = task
        self.training_predictions, self.testing_predictions = self.load_predictions()

    def load_predictions(self):
        file = open(self.training_input, 'rb')
        training_predictions = pickle.load(file)
        file.close()

        file = open(self.testing_input, 'rb')
        testing_predictions = pickle.load(file)
        file.close()

        if self.task == 'image classification':
            return self.convert_predictions(training_predictions), self.convert_predictions(testing_predictions)
        else:
            return training_predictions, testing_predictions

    @staticmethod
    def convert_predictions(predictions):
        predictions = [tf.math.sigmoid(x[0]) for x in predictions]
        return predictions

    def plot(self, training_output, testing_output, bins=20):
        training_predictions = self.training_predictions
        testing_predictions = self.testing_predictions
        plt.hist(training_predictions, bins)
        plt.ylabel('Counts')
        plt.xlabel('Class predictions')
        plt.title('Labeled training set')
        plt.savefig(training_output, dpi=500)
        plt.show()
        plt.close()

        plt.hist(testing_predictions, bins)
        plt.ylabel('Counts')
        plt.xlabel('Class prediction')
        plt.title('Unlabeled testing set')
        plt.savefig(testing_output, dpi=500)
        plt.show()
        plt.close()

    def get_min_loss(self):
        testing_predictions = self.testing_predictions
        loss = 0
        for pred in testing_predictions:
            if pred < 0.5:
                loss += -log(1 - pred)
            else:
                loss += -log(pred)
        return loss / len(testing_predictions)

    def get_min_mislabeled(self, true_training_balance=0.5):
        testing_predictions = self.testing_predictions
        predicted_test_balance = len([x for x in testing_predictions if x > 0.5]) / len(testing_predictions)
        min_mislabeled = int(len(testing_predictions) * abs(predicted_test_balance - true_training_balance))
        print('minimum mislabeled:', min_mislabeled)
        return min_mislabeled

    def get_refined_min_loss(self):
        m = self.get_min_mislabeled()
        predictions = self.testing_predictions
        if self.task == 'image classification':
            predictions = [x.numpy() for x in predictions]
        else:
            predictions = [x[0] for x in predictions]
        min_loss = 0
        distances = [abs(y_hat - 0.5) for y_hat in predictions]
        book = dict(zip(distances, predictions))
        m_smallest_distances = heapq.nsmallest(m, distances)
        closest_predictions = [book[d] for d in m_smallest_distances]
        farthest_predictions = [y_hat for y_hat in predictions if y_hat not in closest_predictions]

        for y_hat in closest_predictions:
            if y_hat < 0.5:
                min_loss += -log(y_hat)
            else:
                min_loss += -log(1 - y_hat)

        for y_hat in farthest_predictions:
            if y_hat < 0.5:
                min_loss += -log(1 - y_hat)
            else:
                min_loss += -log(y_hat)

        return min_loss / len(predictions)

    def get_estimate(self):
        predictions = self.testing_predictions
        if self.task == 'image classification':
            predictions = [x.numpy() for x in predictions]
        loss = 0
        for pred in predictions:
            if 0 < pred < 1:
                loss += -pred * log(pred) - (1 - pred) * log(1 - pred)
        return loss / len(predictions)

    def summary(self, training_output_plot, testing_output_plot, bins=20):
        self.plot(training_output_plot, testing_output_plot, bins)
        print('minimum loss:', self.get_min_loss())
        print('refined minimum loss:', self.get_refined_min_loss())
        print('estimated loss:', self.get_estimate())
