import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from scripts.imputation import impute_mean, impute_min, impute_max
    
# EXERCISE 1

# add a happy test 
# how could your function be misused, edge cases

# tests for imputing with mean value
def test_impute_mean_one_value():
    data = pd.Series([1.0, np.nan, 3.0])    # 1. Define some input data
    expected = pd.Series([1.0, 2.0, 3.0])   # 2. Define what is expected to happen
    actual = impute_mean(data)  				  	       # 3. Run function and record what happens 
    assert_series_equal(expected, actual)   # 4. Make sure expected and actual are equal


def test_impute_mean_two_values():
    #exercise idea: use different arrays
    input_series = pd.Series([1, 2, 3, np.nan, 4, np.nan])
    expected_series = pd.Series([1.0, 2.0, 3.0, 2.5, 4.0, 2.5])
    output_series = impute_mean(series=input_series)
    assert_series_equal(output_series, expected_series)


# tests for imputing with min value
def test_impute_min_one_value():
    data = pd.Series([1.0, np.nan, 3.0])
    expected = pd.Series([1.0, 1.0, 3.0])
    actual = impute_min(data)
    assert_series_equal(expected, actual)

def test_impute_min_two_values():
    #exercise idea: use different arrays
    # note: the data type of the input series is float64, therefore
    # the output series needs to be defined using floating point numbers
    input_series = pd.Series([1, 2, 3, np.nan, 4, np.nan])
    expected_series = pd.Series([1.0, 2.0, 3.0, 1.0, 4.0, 1.0])
    output_series = impute_min(series=input_series)
    assert_series_equal(output_series, expected_series)
    

# tests for imputing with max value
def test_impute_max_one_value():
    data = pd.Series([1.0, np.nan, 3.0])
    expected = pd.Series([1.0, 3.0, 3.0])
    actual = impute_max(data)
    assert_series_equal(expected, actual)

def test_impute_max_two_values():
    #exercise idea: use different arrays
    # note: the data type of the input series is float64, therefore
    # the output series needs to be defined using floating point numbers
    input_series = pd.Series([1, 2, 3, np.nan, 4, np.nan])
    expected_series = pd.Series([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
    output_series = impute_max(series=input_series)
    assert_series_equal(output_series, expected_series)
    

# edge cases

# NaN values only
def test_impute_all_nan():
    data = pd.Series([np.nan, np.nan, np.nan])
    expected = pd.Series([np.nan, np.nan, np.nan])
    actual = impute_mean(data)
    assert_series_equal(expected, actual)
    
# NaN values and None values
def test_impute_all_nan():
    data = pd.Series([np.nan, None, np.nan, None])
    expected = pd.Series([np.nan, np.nan, np.nan, np.nan])
    actual = impute_mean(data)
    assert_series_equal(expected, actual)
