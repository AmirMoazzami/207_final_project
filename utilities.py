import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Python file to hold utility classes and functions.
"""


class DataHandler:
    """Class to load and handle the data."""

    def __init__(self) -> None:
        self.load_data()

    def load_data(self) -> None:
        """Load train and test data"""
        train_df = pd.read_csv("train.csv", index_col=0)
        self.train_data = train_df.iloc[:, :-1]
        self.train_labels = train_df.iloc[:, -1]

        self.test_data = pd.read_csv("test.csv", index_col=0)
