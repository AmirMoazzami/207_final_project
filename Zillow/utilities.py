import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew  # for some statistics

import tensorflow as tf
from tensorflow import keras
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""Utility classes for the Zillow Competition"""

np.random.seed(42)


class ZillowData:
    """"""

    def __init__(self, data_folder_path) -> None:
        self.data_folder_path = data_folder_path

    def get_data(self) -> None:
        """
        Load the provided data.
            - properties_2016.csv
            - properties_2017.csv
            - train_2016_v2.csv
            - train_2017.csv
        """
        # load data sets containing information about the properties
        housing_info_2016 = pd.read_csv(
            os.path.join(self.data_folder_path, "properties_2016.csv")
        )
        housing_info_2017 = pd.read_csv(
            os.path.join(self.data_folder_path, "properties_2017.csv")
        )

        # load
        logerr_2016 = pd.read_csv(
            os.path.join(self.data_folder_path, "train_2016_v2.csv")
        )
        logerr_2017 = pd.read_csv(os.path.join(self.data_folder_path, "train_2017.csv"))

        train_2016 = pd.merge(logerr_2016, housing_info_2016, how="left", on="parcelid")
        train_2017 = pd.merge(logerr_2017, housing_info_2017, how="left", on="parcelid")

        self.data = pd.concat([train_2016, train_2017], ignore_index=True)

    def check_for_duplicates(self) -> None:
        """Print the number of duplicate rows."""
        num_unique_rows = len(self.data[["parcelid", "transactiondate"]].value_counts())
        num_rows = self.data.shape[0]
        print(f"Number of duplicates IDs: {num_rows - num_unique_rows} / {num_rows}")

    def train_val_test_split(self, train_prob, printBool=True) -> None:
        """Randomly select rows based on index for train, validation and test sets."""
        n_rows = self.data.shape[0]
        n_train = int(n_rows * train_prob)

        # get train indices
        train_idx = np.random.choice(range(0, n_rows), size=n_train, replace=False)
        val_test_idx = np.array(list(set(range(0, n_rows)) - set(train_idx)))
        # get val and test indices
        val_idx = np.array(val_test_idx[val_test_idx.shape[0] // 2 :])
        test_idx = np.array(val_test_idx[: val_test_idx.shape[0] // 2])

        if printBool:
            print(f"Train set ratio: {train_idx.shape[0] / n_rows:.2f}")
            print(f"Validation set ratio: {val_idx.shape[0] / n_rows:.2f}")
            print(f"Test set ratio: {test_idx.shape[0] / n_rows:.2f}")

        # get data rows
        self.train = self.data.iloc[train_idx, :]
        self.val = self.data.iloc[val_idx, :]
        self.test = self.data.iloc[test_idx, :]

    @staticmethod
    def plot_logerr_hist(df) -> None:
        """"""
        sns.histplot(data=df.logerror, kde=True).set(title="Distribution of logerrors")
        plt.show()

    @staticmethod
    def plot_logerr_QQ(df) -> None:
        """"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = stats.probplot(x=df.logerror, plot=ax)
        ax.set_title("Q-Q Plot")
        plt.show()

    def form_datasets(self):
        """"""
        # get  targets
        self.train_y = self.train[["parcelid", "logerror"]]
        self.val_y = self.val[["parcelid", "logerror"]]
        self.test_y = self.test[["parcelid", "logerror"]]
        # drop targets from data
        self.train_x = self.train.drop(["logerror"], axis=1)
        self.val_x = self.val.drop(["logerror"], axis=1)
        self.test_x = self.test.drop(["logerror"], axis=1)

    @staticmethod
    def get_missing_ratio_df(df):
        """"""
        na_ratio = df.isnull().sum() / len(df)
        na_ratio = na_ratio.drop(na_ratio[na_ratio == 0].index)
        na_ratio = na_ratio.sort_values(ascending=False)
        na_ratio_df = pd.DataFrame({"NAN_ratio": na_ratio})

        return na_ratio_df

    def accepted_features_list(self, lst) -> None:
        """"""
        self.accepted_features_lst = lst

    def drop_unaccepted_features(self) -> None:
        """"""
        self.train_x = self.train_x[self.accepted_features_lst]
        self.val_x = self.val_x[self.accepted_features_lst]
        self.test_x = self.test_x[self.accepted_features_lst]
