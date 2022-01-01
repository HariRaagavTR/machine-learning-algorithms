import math
import numpy as np
import pandas as pd
import random

def get_entropy_of_dataset(df):
    entropy = 0.0
    n_rows = len(df.index)
    for i in df.iloc[:, -1].value_counts():
        p_i = i / n_rows
        if p_i != 0:
            entropy += -1 * (p_i * math.log(p_i, 2))
    return entropy

def get_avg_info_of_attribute(df, attribute):
    n_rows = len(df.index)
    avg_info = 0.0
    for name, group in df.groupby(attribute):
        avg_info += (len(group.index) / n_rows) * get_entropy_of_dataset(group)
    return avg_info

def get_information_gain(df, attribute):
    return get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)

def get_selected_attribute(df):
    output_dict = {}
    for col in df.columns[:-1]:
        output_dict[col] = get_information_gain(df, col)
    selected_col = max(output_dict, key=output_dict.get)
    return (output_dict, selected_col)
