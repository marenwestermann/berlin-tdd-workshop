
import pandas as pd 
import numpy as np 



def impute_mean(series):
    mean_val = series.mean()
    return series.fillna(mean_val)


# EXERCISE 1 - solutions

def impute_min(series: pd.Series) -> pd.Series:
    min_val = series.min()
    return series.fillna(min_val)

def impute_max(series: pd.Series) -> pd.Series:
    max_val = series.max()
    return series.fillna(max_val)