import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split


class ClaimClassifier():

    def __init__(self, layer_size=20, net_depth=10):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """

        net = [nn.Linear(9, layer_size), nn.ReLU()]
        for i in range(net_depth - 1):
            net.append(nn.Linear(layer_size, layer_size))
            net.append(nn.ReLU())
        net.append(nn.Linear(layer_size, 2))
        net.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*net)

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        X_clean = X_raw / X_raw.max(axis=0)
        return X_clean

    def fit(self, X_raw, y_raw, learning_rate=0.0001, batch_size=100, epochs=10):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of th0.e fitted model
        """

        loss_function = nn.CrossEntropyLoss()
        shuffle_input = True
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        X_clean = self._preprocessor(X_raw)

        trainloader = torch.utils.data.DataLoader(list(zip(X_clean, y_raw)), batch_size=batch_size,
                                                  shuffle=shuffle_input)

        for epoch in range(epochs):
            loss_in_this_epoch = 0
            for images, targets in trainloader:

                optimizer.zero_grad()

                outputs = self.net(images.float())

                loss = loss_function(outputs, targets.long())
                loss.backward()
                loss_in_this_epoch += loss
                optimizer.step()


    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        X_clean = self._preprocessor(X_raw)
        X_clean = torch.as_tensor(np.array(X_clean)).float()
        predictions = self.net(X_clean)
        return predictions.detach().numpy()[:, 1]

    def evaluate_architecture(self, learning_rate, batch_size, epochs, test_x, test_y, train_x, train_y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        self.fit(train_x, train_y, learning_rate=learning_rate, batch_size=batch_size,
                                                     epochs=epochs)
        output = self.predict(test_x)
        output = [1 if r >= 0.5 else 0 for r in output]
        accuracy = accuracy_score(test_y, output)
        roc_score = roc_auc_score(test_y, output)
        return accuracy, roc_score

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


def read_data():
    data = pd.read_csv('part2_training_data.csv')
    with_ones = data.loc[data.made_claim == 1]
    ones = sum(data.made_claim == 1)
    zeroes = sum(data.made_claim == 0)

    for i in range(zeroes // ones - 1):
        data = data.append(with_ones)
    train, test = train_test_split(data, shuffle=True)
    train_x = train.drop(['claim_amount', 'made_claim'], axis='columns')
    train_y = train['made_claim']
    test_x = test.drop(['claim_amount', 'made_claim'], axis='columns')
    test_y = test['made_claim']
    return test_x, test_y, train_x, train_y


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(test_x, test_y, train_x, train_y):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    test_x, test_y, train_x, train_y = test_x.to_numpy(), test_y.to_numpy(), train_x.to_numpy(), train_y.to_numpy()
    batch_size = [30]
    learning_rate = [i for i in np.arange(0.0001, 0.0004, 0.0001)]
    epochs = range(20, 31, 10)
    num_layers = range(6, 14, 2)
    neurons_per_layer = range(20, 51, 10)
    best_batch_size = 0
    best_rate = 0
    best_epoch = 0
    best_combined_score = 0
    best_number_of_layers = 0
    best_number_of_neurons = 0
    for batch in batch_size:
        for rate in learning_rate:
            for epoch in epochs:
                for layers in num_layers:
                    for neurons in neurons_per_layer:
                                claim_classifier = ClaimClassifier(layers, neurons)
                                accuracy, curr_roc_auc_score = claim_classifier.evaluate_architecture(rate, batch, epoch, test_x, test_y, train_x, train_y)
                                if accuracy * curr_roc_auc_score > best_combined_score:
                                    best_combined_score = accuracy * curr_roc_auc_score
                                    best_number_of_layers = layers
                                    best_number_of_neurons = neurons
                                    best_batch_size = batch
                                    best_rate = rate
                                    best_epoch = epoch
                                    claim_classifier.save_model()

    return best_batch_size, best_rate, best_epoch, best_number_of_layers, best_number_of_neurons


if __name__ == '__main__':
    test_x, test_y, train_x, train_y = read_data()
    best_batch_size, best_rate, best_epoch, best_number_of_layers, best_number_of_neurons = ClaimClassifierHyperParameterSearch(
        test_x, test_y, train_x, train_y)
    classifier = load_model()
    output = classifier.predict(test_x.to_numpy())
    output = [1 if r >= 0.5 else 0 for r in output]
    print(accuracy_score(test_y, output))
    print(confusion_matrix(test_y, output))
    print(roc_auc_score(test_y, output))
