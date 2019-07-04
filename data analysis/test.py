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


def process_csv():
    """
    takes a csv file and processes it and returns it in a table format
    :return:
    """
    pd.set_option('display.max_columns', None)
    # of format ("_____.csv", delimiter = "," , (["","",""] (Optional) -> names of columns ), )
    article_read = pd.read_csv("insurance.csv", delimiter=",")
    #article_read[["", ""]] #indicate the name of the columns you want
    #article_read.(column name) == "" #will return True or False or some shit
    #article_read.head() or .tail() or .sample(int) to get a number of rows
    print(article_read.head())

    #Select the user_id, the country and the topic columns for the users who are from country_2! Print the first five rows only!
    #   CODE: article_read[article_read.country == 'country_2'][['user_id', 'topic', 'country']].head()

    # #to also read csv
    # import csv
    # with open('inequality-red.csv', 'r') as f:
    #     wines = list(csv.reader(f, delimiter=';'))
    # print(wines[:3])
    # np.array(wines)


def func() -> str:
    """

    :param link:
    :return:
    """

    raw_html = simple_get('http://www.fabpedigree.com/james/mathmen.htm')
    html = BeautifulSoup(raw_html, 'html.parser')
    for i, li in enumerate(html.select('a')):
        print(i, li.text)

def other():
    """

    :return:
    """

    import numpy as np
    a = np.array([1,2,5])
    for item in a:
        print(item)

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)