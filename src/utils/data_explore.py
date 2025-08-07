import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the style for seaborn plots
plt.style.use('seaborn-v0_8-colorblind')

def get_dtypes(data: pd.DataFrame, drop_col: list = []) -> tuple:
    name_of_col = list(data.columns)
    num_var_list = []
    str_var_list = []

    str_var_list = name_of_col.copy()
    for var in name_of_col:
        if np.issubdtype(data[var].dtype, np.number):
            str_var_list.remove(var)
            num_var_list.append(var)

    for var in drop_col:
        if var in str_var_list:
            str_var_list.remove(var)
        if var in num_var_list:
            num_var_list.remove(var)

    all_var_list = str_var_list + num_var_list
    return str_var_list, num_var_list, all_var_list

def describe(data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    result = data.describe(include='all')
    if output_path is not None:
        output = os.path.join(output_path,'describe.csv')
        result.to_csv(output)
        print('Result saved at:', str(output))
    return result

def discrete_var_barplot(x: str, y: str, data: pd.DataFrame, output_path: str = None) -> None:
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Barplot_' + str(x) + '_' + str(y) + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def discrete_var_countplot(x: str, data: pd.DataFrame, output_path: str = None) -> None:
    plt.figure(figsize=(15, 10))
    sns.countplot(x=x, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Countplot_' + str(x) + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def discrete_var_boxplot(x: str, y: str, data: pd.DataFrame, output_path: str = None) -> None:
    plt.figure(figsize=(15, 10))
    sns.boxplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Boxplot_' + str(x) + '_' + str(y) + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def continuous_var_distplot(x: pd.Series, output_path: str = None, bins: int = None) -> None:
    plt.figure(figsize=(15, 10))
    sns.histplot(x=x, kde=True, bins=bins)
    if output_path is not None:
        output = os.path.join(output_path, 'Distplot_' + str(x.name) + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))

def scatter_plot(x: pd.Series, y: pd.Series, data: pd.DataFrame, output_path: str = None) -> None:
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, 'Scatter_plot_' + str(x.name) + '_' + str(y.name) + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def correlation_plot(data: pd.DataFrame, output_path: str = None) -> None:
    corrmat = data.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 11)
    sns.heatmap(corrmat, cmap="YlGnBu", linewidths=.5, annot=True)
    if output_path is not None:
        output = os.path.join(output_path, 'Corr_plot' + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def heatmap(data: pd.DataFrame, output_path: str = None, fmt: str = 'd') -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 11)
    sns.heatmap(data, cmap="YlGnBu", linewidths=.5, annot=True, fmt=fmt)
    if output_path is not None:
        output = os.path.join(output_path, 'Heatmap' + '.png')
        plt.savefig(output)
        print('Image saved at', str(output))