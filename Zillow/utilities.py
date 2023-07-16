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

    def get_train_2016(self):
        """"""
        answers_df = pd.read_csv(
            os.path.join(self.folderPath, "train_2016_v2.csv"),
            parse_dates=["transactiondate"],
        )
        features_df = pd.read_csv(os.path.join(self.folderPath, "properties_2016.csv"))
        train_df = pd.merge(answers_df, features_df, on="parcelid", how="left")
        return
