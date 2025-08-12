import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def rf_importance(X_train, y_train, max_depth=10, class_weight=None, top_n=15, n_estimators=50, random_state=0):
    """
    Train a Random Forest classifier and plot the top N most important features.

    This function fits a RandomForestClassifier on the given training data,
    computes feature importances, prints them in descending order, and plots
    a bar chart of the top N features by importance.

    Args:
        X_train (pd.DataFrame): Training feature data. Each column is a feature.
        y_train (array-like): Training labels.
        max_depth (int, optional): Maximum depth of the individual trees. Defaults to 10.
        class_weight (dict, list, str, or None, optional): Weights associated with classes. 
            Defaults to None.
        top_n (int, optional): Number of top features to display in the plot. Defaults to 15.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 50.
        random_state (int, optional): Seed for reproducibility. Defaults to 0.

    Returns:
        RandomForestClassifier: The trained RandomForestClassifier model.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> import pandas as pd
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> model = rf_importance(X, y, top_n=3)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature no:%d feature name:%s (%f)" %
              (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))

    indices = indices[:top_n]
    plt.figure()
    plt.title(f"Feature importances top {top_n}")
    plt.bar(range(top_n), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show()

    return model


def gbt_importance(X_train, y_train, max_depth=10, top_n=15, n_estimators=50, random_state=0):
    """
    Train a Gradient Boosting classifier and plot the top N most important features.

    This function fits a GradientBoostingClassifier on the given training data,
    computes feature importances, prints them in descending order, and plots
    a bar chart of the top N features by importance.

    Args:
        X_train (pd.DataFrame): Training feature data. Each column is a feature.
        y_train (array-like): Training labels.
        max_depth (int, optional): Maximum depth of the individual trees. Defaults to 10.
        top_n (int, optional): Number of top features to display in the plot. Defaults to 15.
        n_estimators (int, optional): Number of boosting stages. Defaults to 50.
        random_state (int, optional): Seed for reproducibility. Defaults to 0.

    Returns:
        GradientBoostingClassifier: The trained GradientBoostingClassifier model.

    Example:
        >>> from sklearn.datasets import load_wine
        >>> import pandas as pd
        >>> data = load_wine()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> model = gbt_importance(X, y, top_n=5)
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns
    std = np.std([tree[0].feature_importances_ for tree in model.estimators_], axis=0)

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature no:%d feature name:%s (%f)" %
              (f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))

    indices = indices[:top_n]
    plt.figure()
    plt.title(f"Feature importances top {top_n}")
    plt.bar(range(top_n), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(top_n), indices)
    plt.xlim([-1, top_n])
    plt.show()

    return model
