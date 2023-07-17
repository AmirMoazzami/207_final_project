import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns  # for nicer plots

sns.set(style="darkgrid")  # default style

import tensorflow as tf
from tensorflow import keras
from keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""Utility classes for the Zillow Competition"""


class ZillowData:
    """"""

    def __init__(self) -> None:
        self.folderPath = r"C:\Users\grays\OneDrive\Berkeley DS\Courses\207 ML\Final Project\Zillow Dataset"

    def get_data(self):
        """
        properties_2016.csv
        properties_2017.csv
        train_2016_v2.csv
        train_2017.csv
        """
        properties_2016 = pd.read_csv(
            os.path.join(self.folderPath, "properties_2016.csv")
        )
        properties_2017 = pd.read_csv(
            os.path.join(self.folderPath, "properties_2017.csv")
        )
        train_2016 = pd.read_csv(os.path.join(self.folderPath, "train_2016_v2.csv"))
        train_2017 = pd.read_csv(os.path.join(self.folderPath, "train_2017.csv"))

        train_2016 = pd.merge(train_2016, properties_2016, how="left", on="parcelid")
        train_2017 = pd.merge(train_2017, properties_2017, how="left", on="parcelid")

        all_properties = pd.concat(
            [properties_2016, properties_2017], ignore_index=True
        )
        all_training = pd.concat([train_2016, train_2017], ignore_index=True)

        return all_properties, all_training

    def get_train_2016(self):
        """"""
        answers_df = pd.read_csv(
            os.path.join(self.folderPath, "train_2016_v2.csv"),
            parse_dates=["transactiondate"],
        )
        features_df = pd.read_csv(os.path.join(self.folderPath, "properties_2016.csv"))
        train_df = pd.merge(answers_df, features_df, on="parcelid", how="left")
        return
