import numpy as np

def outlier_detect_arbitrary(data, col, upper, lower):
    """
    Detects outliers based on user-defined upper and lower bounds.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name to check for outliers.
        upper (float): The upper bound threshold.
        lower (float): The lower bound threshold.

    Returns:
        tuple:
            mask (pd.Series): A boolean Series indicating outlier positions.
            bounds (tuple): The (upper, lower) bounds used for detection.
    """
    mask = (data[col] > upper) | (data[col] < lower)
    print('Num of outlier detected:', mask.sum())
    print('Proportion of outlier detected', mask.mean())
    return mask, (upper, lower)


def outlier_detect_IQR(data, col, threshold=3):
    """
    Detects outliers using the Interquartile Range (IQR) method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name to check for outliers.
        threshold (float, optional): Multiplier of IQR to define bounds. Default is 3.

    Returns:
        tuple:
            mask (pd.Series): A boolean Series indicating outlier positions.
            bounds (tuple): The (upper, lower) bounds calculated from IQR.
    """
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = (data[col] > upper) | (data[col] < lower)
    print('Num of outlier detected:', mask.sum())
    print('Proportion of outlier detected', mask.mean())
    return mask, (upper, lower)


def outlier_detect_mean_std(data, col, threshold=3):
    """
    Detects outliers using the mean and standard deviation method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name to check for outliers.
        threshold (float, optional): Number of standard deviations from the mean. Default is 3.

    Returns:
        tuple:
            mask (pd.Series): A boolean Series indicating outlier positions.
            bounds (tuple): The (upper, lower) bounds calculated from mean ± threshold * std.
    """
    mean = data[col].mean()
    std = data[col].std()
    lower = mean - threshold * std
    upper = mean + threshold * std
    mask = (data[col] > upper) | (data[col] < lower)
    print('Num of outlier detected:', mask.sum())
    print('Proportion of outlier detected', mask.mean())
    return mask, (upper, lower)


def outlier_detect_MAD(data, col, threshold=3.5):
    """
    Detects outliers using the Median Absolute Deviation (MAD) method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name to check for outliers.
        threshold (float, optional): Z-score threshold based on MAD. Default is 3.5.

    Returns:
        pd.Series: A boolean Series indicating outlier positions.
    """
    median = data[col].median()
    mad = np.median(np.abs(data[col] - median))
    if mad == 0:
        mask = data[col] != median
    else:
        z_scores = 0.6745 * (data[col] - median) / mad
        mask = np.abs(z_scores) > threshold
    print('Num of outlier detected:', mask.sum())
    print('Proportion of outlier detected', mask.mean())
    return mask


def impute_outlier_with_arbitrary(data, mask, value, cols):
    """
    Replaces outlier values with a specified arbitrary value.

    Args:
        data (pd.DataFrame): The original DataFrame.
        mask (pd.Series): A boolean Series indicating outlier positions.
        value (Any): The value to impute in place of outliers.
        cols (list): List of column names where imputation should occur.

    Returns:
        pd.DataFrame: A copy of the DataFrame with imputed values.
    """
    data_copy = data.copy()
    for col in cols:
        data_copy.loc[mask, col] = value
    return data_copy


def windsorization(data, col, bounds, strategy='both'):
    """
    Applies Windsorization to limit extreme values to specified bounds.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column to apply Windsorization on.
        bounds (tuple): A tuple of (upper, lower) bounds.
        strategy (str, optional): One of 'both', 'top', or 'bottom' to define which side(s) to clip. Default is 'both'.

    Returns:
        pd.DataFrame: A copy of the DataFrame with Windsorized values.
    """
    data_copy = data.copy()
    upper, lower = bounds
    if strategy in ('both', 'top'):
        data_copy.loc[data_copy[col] > upper, col] = upper
    if strategy in ('both', 'bottom'):
        data_copy.loc[data_copy[col] < lower, col] = lower
    return data_copy


def drop_outlier(data, mask):
    """
    Drops rows identified as outliers.

    Args:
        data (pd.DataFrame): The input DataFrame.
        mask (pd.Series): A boolean Series indicating outlier rows.

    Returns:
        pd.DataFrame: A new DataFrame with outlier rows removed.
    """
    return data[~mask]


def impute_outlier_with_avg(data, col, mask, strategy='mean'):
    """
    Replaces outlier values with a statistical average: mean, median, or mode.

    Args:
        data (pd.DataFrame): The input DataFrame.
        col (str): The column name containing outliers.
        mask (pd.Series): A boolean Series indicating outlier positions.
        strategy (str, optional): One of 'mean', 'median', or 'mode'. Default is 'mean'.

    Returns:
        pd.DataFrame: A copy of the DataFrame with imputed values.
    """
    data_copy = data.copy()
    if strategy == 'mean':
        value = data_copy[col].mean()
    elif strategy == 'median':
        value = data_copy[col].median()
    elif strategy == 'mode':
        value = data_copy[col].mode()[0]
    else:
        return data_copy  # No change if strategy is unknown
    data_copy.loc[mask, col] = value
    return data_copy
