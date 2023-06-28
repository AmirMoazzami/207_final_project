import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    @staticmethod
    def plot_label_counts(df):
        figure, axes = plt.subplots(figsize=(20, 5), facecolor="white")
        axes.bar(df.unique(), df.value_counts())
        axes.set_title("Distribution of class examples in training set", size=20)
        axes.set_xlabel("Class labels", size=15)
        axes.set_ylabel("Representation counts", size=15)
        plt.grid()
        plt.show()

    @staticmethod
    def plot_feature_counts(df):
        num_cols = 4
        num_rows = (len(df.columns)) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(18, 4 * num_rows),
            facecolor="white",
        )
        for i, col_name in enumerate(df.columns):
            ax = axes[(i - 1) // num_cols, (i - 1) % num_cols]
            sns.countplot(data=df, x=col_name, ax=ax)
            ax.set_title(f"{col_name.title()}", fontsize=18)
            ax.set_xlabel(col_name.title(), fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=12)
            sns.despine()
        plt.tight_layout()
        plt.show()
