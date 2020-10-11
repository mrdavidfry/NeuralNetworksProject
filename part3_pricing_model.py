from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, input_dim=None, layer_size=100, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # ============================================================

        net = [nn.Linear(input_dim, layer_size), nn.ReLU()]
        for i in range(6):
            net.append(nn.Linear(layer_size, layer_size))
            net.append(nn.ReLU())
        net.append(nn.Linear(layer_size, 2))
        net.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*net)


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        X_clean = preprocessing.normalize(X_raw, norm='l2')
        return X_clean


    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        loss_function = nn.CrossEntropyLoss()
        shuffle_input = True
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

        X_clean = self._preprocessor(X_raw)

        trainloader = torch.utils.data.DataLoader(list(zip(X_clean, y_raw)), batch_size=100,
                                                  shuffle=shuffle_input)

        for epoch in range(50):
            loss_in_this_epoch = 0
            for inputs, targets in trainloader:
                optimizer.zero_grad()

                outputs = self.net(inputs.float())

                loss = loss_function(outputs, targets.long())
                loss.backward()
                loss_in_this_epoch += loss
                optimizer.step()

        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        X_clean = self._preprocessor(X_raw)
        # X_clean.astype(float)
        X_clean = torch.as_tensor(np.array(X_clean)).float()
        predictions = self.net(X_clean)
        return predictions.detach().numpy()[:, 1]

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)

    def load_model(self):
        # Please alter this section so that it works in tandem with the save_model method of your class
        with open('part3_pricing_model.pickle', 'rb') as target:
            trained_model = pickle.load(target)
        return trained_model

def read_data():
    data = pd.read_csv('part3_training_data.csv')
    data = pd.DataFrame(data)
    with_ones = data.loc[data.made_claim == 1]
    ones = sum(data.made_claim == 1)
    zeroes = sum(data.made_claim == 0)

    data = data.select_dtypes(exclude=['O'])
    # data.drop(['id_policy', 'pol_coverage', 'pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_drv2', 'drv_sex1', 'drv_sex2', 'vh_fuel', 'vh_make', 'vh_model', 'vh_type'], axis='columns')
    # data.drv_sex1 = data.drv_sex1.map(lambda gender: 1.0 if gender == 'M' else 0.0 if gender == 'F' else 0.5)
    # data.drv_sex2 = data.drv_sex2.map(lambda gender: 1.0 if gender == 'M' else 0.0 if gender == 'F' else 0.5)
    for i in range(zeroes // ones - 1):
        data = data.append(with_ones)

    data = data.fillna(0, axis='rows')
    data = data.loc[:, data.dtypes != 'O']

    correlations = abs(data.corr()['made_claim'])
    cols_to_drop = []
    for col, value in correlations.items():
        if value < 0.009:
            cols_to_drop.append(col)
    data.drop(cols_to_drop, axis='columns')

    train, test = train_test_split(data, shuffle=True)
    claims_raw = train['claim_amount']
    y_raw = train['made_claim']
    train_x = train.drop(['claim_amount', 'made_claim'], axis='columns')
    claims_test = test['made_claim']
    test_x = test.drop(['made_claim', 'claim_amount'], axis='columns')
    return train_x, y_raw, claims_raw, claims_test, test_x


if __name__ == '__main__':
    train_x, y_raw, claims_raw, claims_test, test_x = read_data()
    train_x, y_raw, claims_raw, claims_test, test_x = train_x.to_numpy(), y_raw.to_numpy(), claims_raw.to_numpy(), claims_test.to_numpy(), test_x.to_numpy()
    classifier = PricingModel(train_x.shape[1])
    classifier.fit(train_x, y_raw, claims_raw)
    output = classifier.predict_claim_probability(test_x)
    print(f'output e {output}')
    price = classifier.predict_premium(test_x)
    print("price is ")
    print(price)
    thresh_hold = [i for i in np.arange(0.3, 0.71, 0.3)]
    best_acc = 0
    best_roc = 0
    for thresh in thresh_hold:
        output = [1 if r >= thresh else 0 for r in output]
        acc = accuracy_score(claims_test, output)
        roc_score = roc_auc_score(claims_test, output)
        if roc_score > best_roc:
            best_roc = roc_score
            best_acc = acc
            classifier.save_model()
    print(f'best roc {best_roc}')
    print(f'best acc {best_acc}')


