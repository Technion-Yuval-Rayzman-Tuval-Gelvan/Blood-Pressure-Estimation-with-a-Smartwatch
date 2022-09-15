import os
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import Utils as utils
import Config as cfg
import Plot as plot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Trainer:

    def __init__(self, true_label, false_label, win_dict):
        self.train_full_set = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.svc = None
        self.lda = None
        self.true_label = true_label
        self.false_label = false_label
        self.feature_list = ['s_sqi', 'p_sqi', 'm_sqi', 'e_sqi', 'z_sqi', 'snr_sqi', 'k_sqi', 'corr']

        self.create_dataset(win_dict)

    def extract_x_y(self, dataset):
        x = dataset[self.feature_list].values
        y = (dataset['label'].values == self.true_label.value) * 2 - 1
        return x, y

    def create_dataset(self, full_dataset):
        full_dataset = pd.DataFrame(full_dataset)
        dataset = full_dataset.query(f'label == {self.true_label.value} or label == {self.false_label.value}')

        n_samples = len(dataset)

        ## Generate a random generator with a fixed seed
        rand_gen = np.random.RandomState(1)

        ## Generating a vector of indices
        indices = np.arange(n_samples)

        ## Shuffle the indices
        rand_gen.shuffle(indices)

        ## Split the indices into 60% train / 20% validation / 20% test
        n_samples_train = int(n_samples * 0.6)
        n_samples_val = int(n_samples * 0.2)
        train_indices = indices[:n_samples_train]
        val_indices = indices[n_samples_train:(n_samples_train + n_samples_val)]
        test_indices = indices[(n_samples_train + n_samples_val):]

        train_full_indices = np.concatenate((train_indices, val_indices))

        self.train_full_set = dataset.iloc[train_full_indices]
        self.train_set = dataset.iloc[train_indices]
        self.val_set = dataset.iloc[val_indices]
        self.test_set = dataset.iloc[test_indices]

    def plot_prediction(self, x_train, y_train, mean, std):
        dist = ((x_train - mean) / std) @ self.svc.coef_.T + self.svc.intercept_

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_ylabel(self.true_label.name)
        ax.hist(dist[(y_train == 1)], np.arange(-10, 10, 0.1), alpha=0.5, label=self.true_label.name)
        ax.hist(dist[(y_train == -1)], np.arange(-10, 10, 0.1), alpha=0.5, label=self.false_label.name)

        ax.plot([-1, -1], [0, 25], '--k')
        ax.plot([0, 0], [0, 25], 'k')
        ax.plot([1, 1], [0, 25], '--k')

        ax.set_xlim(-10, 10)
        ax.set_title('$w^Tx+b$')
        ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(f'{cfg.SVM_DIR}/svm_{self.true_label.name}_{self.false_label.name}.png', dpi=240)

    def adjust_c_value(self, x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full, mean, std):
        ## Define the list of C values to test
        c_list = np.logspace(-3, 3, 13)

        risk_array = np.zeros((len(c_list),))

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)

        ## Train and evaluate the algorithm for each C
        for i_c, c in enumerate(c_list):
            svc = SVC(C=c, kernel='linear')
            std[np.where(std == 0)] = 1
            mean[np.where(std == 0)] = 0
            x_norm = (x_train - mean) / std
            svc.fit(x_norm, y_train)

            predictions = svc.predict((x_val - mean) / std)
            risk_array[i_c] = (y_val != predictions).mean()

        ## Extract the optimal C value
        optimal_index = np.argmin(risk_array)
        optimal_c = c_list[optimal_index]

        # print(f'The optimal C is {optimal_c}')

        ## Re-learn and evalute the model with the optimal C
        self.svc = SVC(C=optimal_c, kernel='linear')
        self.svc.fit((x_train_full - mean) / std, y_train_full)
        predictions = svc.predict((x_test - mean) / std)
        test_loss = (y_test != predictions).mean()
        # print(f'The test loss is: {test_loss:.2}')

        ## PLot risk vs. C
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.plot(c_list, risk_array)
        ax.plot(optimal_c, risk_array[optimal_index], '.r')
        ax.set_xlabel('$K$')
        ax.set_ylabel('Risk')
        ax.set_title('Risk vs. $C$');
        fig.savefig(f'{cfg.SVM_DIR}/selecting_c_{self.true_label.name}_{self.false_label.name}.png', dpi=240)

    def run_svm(self):

        print("****** SVM ******")
        print(f"True label: {self.true_label.name}, False label: {self.false_label.name}")

        x_train_full, y_train_full = self.extract_x_y(self.train_full_set)
        x_train, y_train = self.extract_x_y(self.train_set)
        x_val, y_val = self.extract_x_y(self.val_set)
        x_test, y_test = self.extract_x_y(self.test_set)

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)

        ## Create the SVC object
        self.svc = SVC(C=1.0, kernel='linear', class_weight='balanced')

        ## Run the learning algorithm
        std[np.where(std == 0)] = 1
        mean[np.where(std == 0)] = 0
        x_norm = (x_train - mean) / std
        self.svc.fit(x_norm, y_train)

        ## Evaluate in the test set
        predictions = self.svc.predict((x_test - mean) / std)
        test_loss = (y_test != predictions).mean()

        classification_report_actual = classification_report(y_test, predictions)
        print(classification_report_actual)

        self.plot_prediction(x_train, y_train, mean, std)

        self.adjust_c_value(x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full, mean, std)

        print("Weights Coefficients:")
        coef_dict = {}
        for i, coef in enumerate(self.svc.coef_[0]):
            coef_dict[self.feature_list[i]] = coef

        print(coef_dict)

    def run_lda(self):
        print("****** LDA ******")
        print(f"True label: {self.true_label.name}, False label: {self.false_label.name}")

        x_train_full, y_train_full = self.extract_x_y(self.train_full_set)
        x_train, y_train = self.extract_x_y(self.train_set)
        x_val, y_val = self.extract_x_y(self.val_set)
        x_test, y_test = self.extract_x_y(self.test_set)

        ## %%%%%%%%%%%%%%% Your code here - Begin %%%%%%%%%%%%%%%
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(x_train_full, y_train_full)

        # Define method to evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        # evaluate model
        scores = cross_val_score( self.lda, x_train_full, y_train_full, scoring='accuracy', cv=cv, n_jobs=-1)
        print(np.mean(scores))






