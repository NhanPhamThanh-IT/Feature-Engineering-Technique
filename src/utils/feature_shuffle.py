import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def feature_shuffle_rf(X_train, y_train, max_depth=None, class_weight=None, top_n=15, n_estimators=50, random_state=0):
    """
    Evaluate feature importance by measuring the drop in ROC-AUC after shuffling each feature.

    This function trains a RandomForestClassifier on the given training data,
    then iteratively shuffles each feature, measures the ROC-AUC score drop,
    and returns a ranked list of features by importance.

    Args:
        X_train (pd.DataFrame): Training features, each column representing a feature.
        y_train (array-like): Target labels for training.
        max_depth (int, optional): Maximum depth of the individual trees. Defaults to None.
        class_weight (dict, list, str, or None, optional): Weights associated with classes. Defaults to None.
        top_n (int, optional): Number of top features to consider for selection (used only for display). Defaults to 15.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 50.
        random_state (int, optional): Seed for reproducibility. Defaults to 0.

    Returns:
        tuple:
            - auc_drop (pd.DataFrame): DataFrame containing features and their AUC drop values.
            - selected_features (pd.Series): Features with positive AUC drop, sorted by importance.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_breast_cancer
        >>> data = load_breast_cancer()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> auc_drop_df, selected_feats = feature_shuffle_rf(X, y, n_estimators=100, top_n=10)
        >>> print(auc_drop_df.head())
        >>> print("Selected features:", selected_feats.tolist())
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, (model.predict_proba(X_train))[:, 1])
    feature_dict = {}

    for feature in X_train.columns:
        X_train_c = X_train.copy().reset_index(drop=True)
        y_train_c = y_train.copy().reset_index(drop=True)
        X_train_c[feature] = X_train_c[feature].sample(frac=1, random_state=random_state).reset_index(drop=True)
        shuff_auc = roc_auc_score(y_train_c, (model.predict_proba(X_train_c))[:, 1])
        feature_dict[feature] = (train_auc - shuff_auc)
    
    auc_drop = pd.Series(feature_dict).reset_index()
    auc_drop.columns = ['feature', 'auc_drop']
    auc_drop.sort_values(by=['auc_drop'], ascending=False, inplace=True)
    selected_features = auc_drop[auc_drop.auc_drop > 0]['feature']

    return auc_drop, selected_features
