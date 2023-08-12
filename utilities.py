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

"""
Utility classes for the Zillow Competition

This script contains the ZillowData class which is used in the Zillow_streamlined_feature_selection.ipynb
notebook.
"""

np.random.seed(42)


class ZillowData:
    """Class to help handle the Zillow data."""

    def __init__(self, data_folder_path) -> None:
        self.data_folder_path = data_folder_path

    def get_data(self) -> None:
        """
        Load the provided data.
            Files containing the features
            - properties_2016.csv
            - properties_2017.csv
            Files containing the target logerror values
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
        """Plot a histogram of the logerror values."""
        sns.histplot(data=df.logerror, kde=True).set(title="Distribution of logerrors")
        plt.show()

    @staticmethod
    def plot_logerr_QQ(df) -> None:
        """Plot QQ of logerror values."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = stats.probplot(x=df.logerror, plot=ax)
        ax.set_title("Q-Q Plot")
        plt.show()

    def form_datasets(self):
        """Split train-validation-test into x and y sets."""
        # get targets
        self.train_y = self.train[["logerror"]]
        self.val_y = self.val[["logerror"]]
        self.test_y = self.test[["logerror"]]
        # drop targets from data
        self.train_x = self.train.drop(["logerror"], axis=1)
        self.val_x = self.val.drop(["logerror"], axis=1)
        self.test_x = self.test.drop(["logerror"], axis=1)

    @staticmethod
    def get_missing_ratio_df(df):
        """Return a df of each column and their ratio of missing rows."""
        na_ratio = df.isnull().sum() / len(df)
        # na_ratio = na_ratio.drop(na_ratio[na_ratio == 0].index)
        na_ratio = na_ratio.sort_values(ascending=False)
        na_ratio_df = pd.DataFrame({"NAN_ratio": na_ratio})

        return na_ratio_df

    @staticmethod
    def investigate_column(column, df, bins=20):
        """"""
        print(f"Column: {column}")
        print(f"Nonzero count: {np.count_nonzero(df[column])}")
        na_ratio = df[column].isnull().sum() / len(df[column])
        print(f"Ratio of missing data: {na_ratio}")
        print(f"Type: {df[column].dtypes}")

        try:
            df.hist(column=column, bins=bins)
            plt.show()
        except:
            print("\nUnable to plot histogram.\n")

        print(f"Number of unique values: {len(df[column].unique())}")
        print(f"Unique values: {sorted(df[column].unique())}")

    # # # Data Cleaning # # #
    def drop_column(self, column, printBool=False):
        self.train_x = self.train_x.drop(labels=column, axis=1)
        self.val_x = self.val_x.drop(labels=column, axis=1)
        self.test_x = self.test_x.drop(labels=column, axis=1)

        if printBool:
            print(f"train_x shape: {self.train_x.shape}")
            print(f"val_x shape: {self.val_x.shape}")
            print(f"test_x shape: {self.test_x.shape}")

    @staticmethod
    def drop_column_static(df, column):
        df = df.drop(labels=column, axis=1)
        return df

    def clean_column_impute_binary(
        self, column, fill_value, replace_value, replace_with, printBool=False
    ):
        self.train_x[column] = self.train_x[column].fillna(fill_value)
        self.train_x[column] = self.train_x[column].replace(replace_value, replace_with)

        self.val_x[column] = self.val_x[column].fillna(fill_value)
        self.val_x[column] = self.val_x[column].replace(replace_value, replace_with)

        self.test_x[column] = self.test_x[column].fillna(fill_value)
        self.test_x[column] = self.test_x[column].replace(replace_value, replace_with)

        if printBool:
            print(
                f"train_x {column} unique values: {sorted(self.train_x[column].unique())}"
            )

    @staticmethod
    def clean_column_impute_binary_static(
        df, column, fill_value, replace_value, replace_with
    ):
        df[column] = df[column].fillna(fill_value)
        df[column] = df[column].replace(replace_value, replace_with)
        return df

    def impute_column(self, column, fill_value, printBool=False):
        self.train_x[column] = self.train_x[column].fillna(fill_value)
        self.val_x[column] = self.val_x[column].fillna(fill_value)
        self.test_x[column] = self.test_x[column].fillna(fill_value)

        if printBool:
            print(
                f"train_x {column} unique values: {sorted(self.train_x[column].unique())}"
            )

    @staticmethod
    def impute_column_static(df, column, fill_value):
        df[column] = df[column].fillna(fill_value)
        return df

    def clean_all_data(self):
        """Drop or impute all columns."""
        self.drop_column(column="buildingclasstypeid")
        self.drop_column(column="finishedsquarefeet13")
        self.drop_column(column="basementsqft")
        self.drop_column(column="storytypeid")
        self.drop_column(column="yardbuildingsqft26")
        self.clean_column_impute_binary(
            column="fireplaceflag", fill_value=0, replace_value=True, replace_with=1
        )
        self.drop_column(column="architecturalstyletypeid")
        self.drop_column(column="typeconstructiontypeid")
        self.drop_column(column="finishedsquarefeet6")
        self.clean_column_impute_binary(
            column="decktypeid", fill_value=0, replace_value=66.0, replace_with=1
        )
        self.drop_column(column="pooltypeid10")
        self.impute_column(column="poolsizesum", fill_value=0)
        self.impute_column(column="pooltypeid2", fill_value=0)
        self.clean_column_impute_binary(
            column="hashottuborspa", fill_value=0, replace_value=True, replace_with=1
        )
        self.clean_column_impute_binary(
            column="taxdelinquencyflag", fill_value=0, replace_value="Y", replace_with=1
        )
        self.drop_column(column="taxdelinquencyyear")
        self.impute_column(column="yardbuildingsqft17", fill_value=0)
        self.drop_column(column="finishedsquarefeet15")
        self.drop_column(column="finishedsquarefeet50")
        self.drop_column(column="finishedfloor1squarefeet")
        self.impute_column(column="fireplacecnt", fill_value=0)
        self.drop_column(column="threequarterbathnbr")
        self.impute_column(column="pooltypeid7", fill_value=0)
        self.impute_column(column="poolcnt", fill_value=0)
        self.drop_column(column="numberofstories")
        self.impute_column(
            column="airconditioningtypeid",
            fill_value=self.train_x["airconditioningtypeid"].mode().item(),
        )
        self.impute_column(column="garagetotalsqft", fill_value=0)
        self.impute_column(column="garagecarcnt", fill_value=0)
        self.drop_column(column="regionidneighborhood")
        self.impute_column(
            column="heatingorsystemtypeid",
            fill_value=self.train_x["heatingorsystemtypeid"].mode().item(),
        )
        self.impute_column(
            column="buildingqualitytypeid",
            fill_value=self.train_x["buildingqualitytypeid"].median(),
        )
        self.drop_column(column="propertyzoningdesc")
        self.impute_column(
            column="unitcnt", fill_value=self.train_x["unitcnt"].mode().item()
        )
        self.impute_column(
            column="lotsizesquarefeet",
            fill_value=self.train_x["lotsizesquarefeet"].median(),
        )
        self.impute_column(
            column="finishedsquarefeet12",
            fill_value=self.train_x["finishedsquarefeet12"].median(),
        )
        self.impute_column(
            column="regionidcity", fill_value=self.train_x["regionidcity"].mode().item()
        )
        self.drop_column(column="calculatedbathnbr")
        self.drop_column(column="fullbathcnt")
        self.impute_column(
            column="yearbuilt", fill_value=self.train_x["yearbuilt"].mode().item()
        )
        self.drop_column(column="censustractandblock")
        self.impute_column(
            column="calculatedfinishedsquarefeet",
            fill_value=self.train_x["calculatedfinishedsquarefeet"].median(),
        )
        self.impute_column(
            column="structuretaxvaluedollarcnt",
            fill_value=self.train_x["structuretaxvaluedollarcnt"].median(),
        )
        self.impute_column(
            column="regionidzip", fill_value=self.train_x["regionidzip"].mode().item()
        )
        self.impute_column(
            column="taxamount", fill_value=self.train_x["taxamount"].median()
        )
        self.impute_column(
            column="taxvaluedollarcnt",
            fill_value=self.train_x["taxvaluedollarcnt"].median(),
        )
        self.impute_column(
            column="landtaxvaluedollarcnt",
            fill_value=self.train_x["landtaxvaluedollarcnt"].median(),
        )
        self.drop_column(column="rawcensustractandblock")
        self.impute_column(
            column="propertylandusetypeid",
            fill_value=self.train_x["propertylandusetypeid"].mode().item(),
        )
        self.impute_column(column="fips", fill_value=self.train_x["fips"].mode().item())
        self.impute_column(
            column="longitude", fill_value=self.train_x["longitude"].median()
        )
        self.drop_column(column="roomcnt")
        self.impute_column(
            column="latitude", fill_value=self.train_x["latitude"].median()
        )
        self.impute_column(
            column="regionidcounty",
            fill_value=self.train_x["regionidcounty"].mode().item(),
        )
        self.impute_column(
            column="assessmentyear",
            fill_value=self.train_x["assessmentyear"].mode().item(),
        )
        self.impute_column(
            column="bedroomcnt", fill_value=self.train_x["bedroomcnt"].median()
        )
        self.impute_column(
            column="bathroomcnt", fill_value=self.train_x["bathroomcnt"].median()
        )
        self.drop_column(column="propertycountylandusecode")

        self.drop_column(column="parcelid")

        # handle dates
        # train x
        datetime_obj = pd.to_datetime(self.train_x["transactiondate"]).dt
        self.train_x["day"] = datetime_obj.day
        self.train_x["year"] = (datetime_obj.year).astype("category")
        self.train_x["month"] = (
            (datetime_obj.year - 2016) * 12 + datetime_obj.month
        ).astype("category")
        self.train_x = self.train_x.drop(labels="transactiondate", axis=1)

        self.train_x["age_from_2016"] = 2016.0 - self.train_x["yearbuilt"]

        # save the train_x yearbuilt as a series to be used for transforming other datasets
        self.train_x_yearbuilt = self.train_x["yearbuilt"]
        self.train_x = self.train_x.drop(labels="yearbuilt", axis=1)

        # validation x
        datetime_obj = pd.to_datetime(self.val_x["transactiondate"]).dt
        self.val_x["day"] = datetime_obj.day
        self.val_x["year"] = (datetime_obj.year).astype("category")
        self.val_x["month"] = (
            (datetime_obj.year - 2016) * 12 + datetime_obj.month
        ).astype("category")
        self.val_x = self.val_x.drop(labels="transactiondate", axis=1)

        self.val_x["age_from_2016"] = 2016.0 - self.val_x["yearbuilt"]
        self.val_x = self.val_x.drop(labels="yearbuilt", axis=1)

        # test x
        datetime_obj = pd.to_datetime(self.test_x["transactiondate"]).dt
        self.test_x["day"] = datetime_obj.day
        self.test_x["year"] = (datetime_obj.year).astype("category")
        self.test_x["month"] = (
            (datetime_obj.year - 2016) * 12 + datetime_obj.month
        ).astype("category")
        self.test_x = self.test_x.drop(labels="transactiondate", axis=1)

        self.test_x["age_from_2016"] = 2016.0 - self.test_x["yearbuilt"]
        self.test_x = self.test_x.drop(labels="yearbuilt", axis=1)

    def clean_df(self, df):
        """
        Recieve a dataframe.
        Drop or impute all columns using the x_train distributions of values.
        """
        df = ZillowData.drop_column_static(df, column="buildingclasstypeid")
        df = ZillowData.drop_column_static(df, column="finishedsquarefeet13")
        df = ZillowData.drop_column_static(df, column="basementsqft")
        df = ZillowData.drop_column_static(df, column="storytypeid")
        df = ZillowData.drop_column_static(df, column="yardbuildingsqft26")
        df = ZillowData.clean_column_impute_binary_static(
            df, column="fireplaceflag", fill_value=0, replace_value=True, replace_with=1
        )
        df = ZillowData.drop_column_static(df, column="architecturalstyletypeid")
        df = ZillowData.drop_column_static(df, column="typeconstructiontypeid")
        df = ZillowData.drop_column_static(df, column="finishedsquarefeet6")
        df = ZillowData.clean_column_impute_binary_static(
            df, column="decktypeid", fill_value=0, replace_value=66.0, replace_with=1
        )
        df = ZillowData.drop_column_static(df, column="pooltypeid10")
        df = ZillowData.impute_column_static(df, column="poolsizesum", fill_value=0)
        df = ZillowData.impute_column_static(df, column="pooltypeid2", fill_value=0)
        df = ZillowData.clean_column_impute_binary_static(
            df,
            column="hashottuborspa",
            fill_value=0,
            replace_value=True,
            replace_with=1,
        )
        df = ZillowData.clean_column_impute_binary_static(
            df,
            column="taxdelinquencyflag",
            fill_value=0,
            replace_value="Y",
            replace_with=1,
        )
        df = ZillowData.drop_column_static(df, column="taxdelinquencyyear")
        df = ZillowData.impute_column_static(
            df, column="yardbuildingsqft17", fill_value=0
        )
        df = ZillowData.drop_column_static(df, column="finishedsquarefeet15")
        df = ZillowData.drop_column_static(df, column="finishedsquarefeet50")
        df = ZillowData.drop_column_static(df, column="finishedfloor1squarefeet")
        df = ZillowData.impute_column_static(df, column="fireplacecnt", fill_value=0)
        df = ZillowData.drop_column_static(df, column="threequarterbathnbr")
        df = ZillowData.impute_column_static(df, column="pooltypeid7", fill_value=0)
        df = ZillowData.impute_column_static(df, column="poolcnt", fill_value=0)
        df = ZillowData.drop_column_static(df, column="numberofstories")
        df = ZillowData.impute_column_static(
            df,
            column="airconditioningtypeid",
            fill_value=self.train_x["airconditioningtypeid"].mode().item(),
        )
        df = ZillowData.impute_column_static(df, column="garagetotalsqft", fill_value=0)
        df = ZillowData.impute_column_static(df, column="garagecarcnt", fill_value=0)
        df = ZillowData.drop_column_static(df, column="regionidneighborhood")
        df = ZillowData.impute_column_static(
            df,
            column="heatingorsystemtypeid",
            fill_value=self.train_x["heatingorsystemtypeid"].mode().item(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="buildingqualitytypeid",
            fill_value=self.train_x["buildingqualitytypeid"].median(),
        )
        df = ZillowData.drop_column_static(df, column="propertyzoningdesc")
        df = ZillowData.impute_column_static(
            df, column="unitcnt", fill_value=self.train_x["unitcnt"].mode().item()
        )
        df = ZillowData.impute_column_static(
            df,
            column="lotsizesquarefeet",
            fill_value=self.train_x["lotsizesquarefeet"].median(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="finishedsquarefeet12",
            fill_value=self.train_x["finishedsquarefeet12"].median(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="regionidcity",
            fill_value=self.train_x["regionidcity"].mode().item(),
        )
        df = ZillowData.drop_column_static(df, column="calculatedbathnbr")
        df = ZillowData.drop_column_static(df, column="fullbathcnt")
        df = ZillowData.impute_column_static(
            df, column="yearbuilt", fill_value=self.train_x_yearbuilt.mode().item()
        )
        df = ZillowData.drop_column_static(df, column="censustractandblock")
        df = ZillowData.impute_column_static(
            df,
            column="calculatedfinishedsquarefeet",
            fill_value=self.train_x["calculatedfinishedsquarefeet"].median(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="structuretaxvaluedollarcnt",
            fill_value=self.train_x["structuretaxvaluedollarcnt"].median(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="regionidzip",
            fill_value=self.train_x["regionidzip"].mode().item(),
        )
        df = ZillowData.impute_column_static(
            df, column="taxamount", fill_value=self.train_x["taxamount"].median()
        )
        df = ZillowData.impute_column_static(
            df,
            column="taxvaluedollarcnt",
            fill_value=self.train_x["taxvaluedollarcnt"].median(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="landtaxvaluedollarcnt",
            fill_value=self.train_x["landtaxvaluedollarcnt"].median(),
        )
        df = ZillowData.drop_column_static(df, column="rawcensustractandblock")
        df = ZillowData.impute_column_static(
            df,
            column="propertylandusetypeid",
            fill_value=self.train_x["propertylandusetypeid"].mode().item(),
        )
        df = ZillowData.impute_column_static(
            df, column="fips", fill_value=self.train_x["fips"].mode().item()
        )
        df = ZillowData.impute_column_static(
            df, column="longitude", fill_value=self.train_x["longitude"].median()
        )
        df = ZillowData.drop_column_static(df, column="roomcnt")
        df = ZillowData.impute_column_static(
            df, column="latitude", fill_value=self.train_x["latitude"].median()
        )
        df = ZillowData.impute_column_static(
            df,
            column="regionidcounty",
            fill_value=self.train_x["regionidcounty"].mode().item(),
        )
        df = ZillowData.impute_column_static(
            df,
            column="assessmentyear",
            fill_value=self.train_x["assessmentyear"].mode().item(),
        )
        df = ZillowData.impute_column_static(
            df, column="bedroomcnt", fill_value=self.train_x["bedroomcnt"].median()
        )
        df = ZillowData.impute_column_static(
            df, column="bathroomcnt", fill_value=self.train_x["bathroomcnt"].median()
        )
        df = ZillowData.drop_column_static(df, column="propertycountylandusecode")

        df = ZillowData.drop_column_static(df, column="parcelid")

        # handle dates
        # train x
        datetime_obj = pd.to_datetime(df["transactiondate"]).dt
        df["day"] = datetime_obj.day
        df["year"] = (datetime_obj.year).astype("category")
        df["month"] = ((datetime_obj.year - 2016) * 12 + datetime_obj.month).astype(
            "category"
        )
        df = df.drop(labels="transactiondate", axis=1)

        df["age_from_2016"] = 2016.0 - df["yearbuilt"]
        df = df.drop(labels="yearbuilt", axis=1)

        return df

    def convert_column_type(self, col_type=np.float32):
        self.train_x = self.train_x.astype(col_type)
        self.val_x = self.val_x.astype(col_type)
        self.test_x = self.test_x.astype(col_type)

    def format_x_y(self):
        """"""
        # format targets
        self.train_y_arr = self.train_y["logerror"].to_numpy()
        self.val_y_arr = self.val_y["logerror"].to_numpy()
        self.test_y_arr = self.test_y["logerror"].to_numpy()

        # format the x values
        self.train_x_arr = self.train_x.to_numpy()
        self.val_x_arr = self.val.to_numpy()
        self.test_x_arr = self.test.to_numpy()
