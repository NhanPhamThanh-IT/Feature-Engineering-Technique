import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-colorblind')

def get_dtypes(data: pd.DataFrame, drop_col: list = []) -> tuple:
    """
    Classify columns of a DataFrame into string (categorical) and numeric variables.

    This function scans the provided DataFrame and separates the columns into:
    - `str_var_list`: columns that are non-numeric (object or categorical types).
    - `num_var_list`: columns that contain numeric data.
    - `all_var_list`: combined list of string and numeric variables after dropping specified columns.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame whose column types are to be determined.
    drop_col : list, optional
        List of column names to exclude from the classification (default is an empty list).

    Returns
    -------
    tuple
        A tuple containing:
        - str_var_list (list): List of non-numeric column names.
        - num_var_list (list): List of numeric column names.
        - all_var_list (list): Combined list of both non-numeric and numeric columns after dropping specified ones.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    >>> get_dtypes(df)
    (['B'], ['A'], ['B', 'A'])
    """
    name_of_col = list(data.columns)
    num_var_list = []
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
    """
    Generate descriptive statistics for all columns in the DataFrame.

    Computes statistics such as count, mean, std, min, max, and quartiles for numeric
    columns, and count, unique values, top value, and frequency for categorical columns.

    Optionally saves the result to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    output_path : str, optional
        Directory path to save the output CSV. If None, the result is not saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics.

    Notes
    -----
    - If `output_path` is provided, the output file is named `describe.csv`.
    - Works for both numeric and categorical columns.

    Examples
    --------
    >>> describe(df, output_path='reports/')
    """
    result = data.describe(include='all')
    if output_path is not None:
        output = os.path.join(output_path, 'describe.csv')
        result.to_csv(output)
        print('Result saved at:', str(output))
    return result


def discrete_var_barplot(x: str, y: str, data: pd.DataFrame, output_path: str = None) -> None:
    """
    Create a bar plot for a discrete (categorical) variable against a numeric target.

    Parameters
    ----------
    x : str
        The name of the categorical variable.
    y : str
        The name of the numeric variable to plot on the Y-axis.
    data : pd.DataFrame
        The input DataFrame.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Barplot_<x>_<y>.png` if saved.
    - Uses `seaborn.barplot` under the hood.
    """
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, f'Barplot_{x}_{y}.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def discrete_var_countplot(x: str, data: pd.DataFrame, output_path: str = None) -> None:
    """
    Create a count plot for a discrete (categorical) variable.

    Parameters
    ----------
    x : str
        The name of the categorical variable.
    data : pd.DataFrame
        The input DataFrame.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Countplot_<x>.png` if saved.
    """
    plt.figure(figsize=(15, 10))
    sns.countplot(x=x, data=data)
    if output_path is not None:
        output = os.path.join(output_path, f'Countplot_{x}.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def discrete_var_boxplot(x: str, y: str, data: pd.DataFrame, output_path: str = None) -> None:
    """
    Create a box plot for a categorical variable against a numeric target.

    Parameters
    ----------
    x : str
        The name of the categorical variable.
    y : str
        The name of the numeric variable to plot on the Y-axis.
    data : pd.DataFrame
        The input DataFrame.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Boxplot_<x>_<y>.png` if saved.
    """
    plt.figure(figsize=(15, 10))
    sns.boxplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, f'Boxplot_{x}_{y}.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def continuous_var_distplot(x: pd.Series, output_path: str = None, bins: int = None) -> None:
    """
    Create a distribution plot (histogram + KDE) for a continuous variable.

    Parameters
    ----------
    x : pd.Series
        The continuous variable to plot.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.
    bins : int, optional
        Number of histogram bins. If None, seaborn chooses automatically.

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Distplot_<x.name>.png` if saved.
    """
    plt.figure(figsize=(15, 10))
    sns.histplot(x=x, kde=True, bins=bins)
    if output_path is not None:
        output = os.path.join(output_path, f'Distplot_{x.name}.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def scatter_plot(x: pd.Series, y: pd.Series, data: pd.DataFrame, output_path: str = None) -> None:
    """
    Create a scatter plot for two continuous variables.

    Parameters
    ----------
    x : pd.Series
        The variable to plot on the X-axis.
    y : pd.Series
        The variable to plot on the Y-axis.
    data : pd.DataFrame
        The DataFrame containing the variables.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Scatter_plot_<x.name>_<y.name>.png` if saved.
    """
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=x, y=y, data=data)
    if output_path is not None:
        output = os.path.join(output_path, f'Scatter_plot_{x.name}_{y.name}.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def correlation_plot(data: pd.DataFrame, output_path: str = None) -> None:
    """
    Create a correlation heatmap for all numeric variables in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.

    Returns
    -------
    None

    Notes
    -----
    - Uses Pearson correlation by default.
    - The output file name will be `Corr_plot.png` if saved.
    """
    corrmat = data.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 11)
    sns.heatmap(corrmat, cmap="YlGnBu", linewidths=.5, annot=True)
    if output_path is not None:
        output = os.path.join(output_path, 'Corr_plot.png')
        plt.savefig(output)
        print('Image saved at', str(output))


def heatmap(data: pd.DataFrame, output_path: str = None, fmt: str = 'd') -> None:
    """
    Create a heatmap for a given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The data to visualize in the heatmap.
    output_path : str, optional
        Directory path to save the generated image. If None, the image is not saved.
    fmt : str, optional
        String formatting code to control annotation format (default is 'd' for integers).

    Returns
    -------
    None

    Notes
    -----
    - The output file name will be `Heatmap.png` if saved.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 11)
    sns.heatmap(data, cmap="YlGnBu", linewidths=.5, annot=True, fmt=fmt)
    if output_path is not None:
        output = os.path.join(output_path, 'Heatmap.png')
        plt.savefig(output)
        print('Image saved at', str(output))
