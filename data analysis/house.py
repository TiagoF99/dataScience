import urllib
from bs4 import BeautifulSoup
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
import csv
import numpy as np
import pandas as pd
import flask
import PIL
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as Lr, Ridge


def q1():
    """

    :return:
    """
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("house.csv", delimiter=",")

    # data type of each column
    print(df.dtypes)

    print("\n" + " ===============================================================================" + "\n")
    # top 5 rows
    print(df.head())

    print("\n" + " ===============================================================================" + "\n")

    # drops Unnamed:0 and id columns from data frame
    df = df.drop(axis=1, columns=["Unnamed: 0", "id"])
    print(df.head())

    print("\n" + " ===============================================================================" + "\n")

    # gets count of unique values of the floor column
    floor_count = df['floors'].value_counts().to_frame()
    print(floor_count)

    print("\n" + " ===============================================================================" + "\n")

    # plot that can be used to determine whether houses with a waterfront view or without a waterfront view have more
    # price outliers.
    df1 = df[['waterfront', 'price']]
    sns.boxplot(x=df['waterfront'], y=df['price'], data=df1)
    # plt.show()

    print("\n" + " ===============================================================================" + "\n")

    # scatter plot with sqft_above on x and price on y axis
    # plotted a line of best fit
    # sqft_above is positively correlated to price
    sns.regplot(x=df['sqft_above'], y=df['price'], data=df)
    plt.ylim(0,)
    # plt.show()

    print("\n" + " ===============================================================================" + "\n")

    # predicts the price using the feature 'sqft_living' then calculated the R^2
    # model sort of explains variation around prices mean (approx. 50%)
    lm = Lr()
    x = df[['sqft_living']]
    y = df['price']
    lm.fit(x, y)
    r_squared = lm.score(x,y)
    print(r_squared)

    print("\n" + " ===============================================================================" + "\n")

    # linear model to predict price using those 4 variables
    lm = Lr()
    x = df[['floors', 'waterfront', 'lat', 'sqft_living']]
    y = df['price']
    lm.fit(x, y)
    r_squared = lm.score(x, y)
    print(r_squared)


if __name__ == "__main__":
    q1()