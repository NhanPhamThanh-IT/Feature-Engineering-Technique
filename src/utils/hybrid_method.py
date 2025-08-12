from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def recursive_feature_elimination_rf(X_train, y_train, X_test, y_test,
                                     tol=0.001, max_depth=None,
                                     class_weight=None,
                                     top_n=15, n_estimators=50, random_state=0):
    """
    Perform Recursive Feature Elimination (RFE) using a RandomForestClassifier.

    This function iteratively tests the removal of each feature to evaluate its 
    impact on the model's ROC AUC score. Features that do not significantly reduce 
    the AUC score when removed are discarded.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series or np.ndarray): Training target labels.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series or np.ndarray): Test target labels.
        tol (float, optional): Minimum drop in ROC AUC required to keep a feature. Default is 0.001.
        max_depth (int, optional): Maximum depth of the trees. Default is None.
        class_weight (dict, list, str, or None, optional): Class weights. Default is None.
        top_n (int, optional): Number of top features to consider. Default is 15.
        n_estimators (int, optional): Number of trees in the forest. Default is 50.
        random_state (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        list: List of features to keep after elimination.
    """
    features_to_remove = []
    count = 1
    model_all_features = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    model_all_features.fit(X_train, y_train)
    y_pred_test = model_all_features.predict_proba(X_test)[:, 1]
    auc_score_all = roc_auc_score(y_test, y_pred_test)
    
    for feature in X_train.columns:
        print()
        print('Testing feature: ', feature, ' which is feature ', count,
              ' out of ', len(X_train.columns))
        count += 1
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )
        model.fit(X_train.drop(features_to_remove + [feature], axis=1), y_train)
        y_pred_test = model.predict_proba(
            X_test.drop(features_to_remove + [feature], axis=1)
        )[:, 1]
        auc_score_int = roc_auc_score(y_test, y_pred_test)
        print('New test ROC AUC={}'.format((auc_score_int)))
        print('All features test ROC AUC={}'.format((auc_score_all)))
        diff_auc = auc_score_all - auc_score_int
        if diff_auc >= tol:
            print('Drop in ROC AUC={}'.format(diff_auc))
            print('Keep: ', feature)
        else:
            print('Drop in ROC AUC={}'.format(diff_auc))
            print('Remove: ', feature)
            auc_score_all = auc_score_int
            features_to_remove.append(feature)
    print('DONE!!')
    print('Total features to remove: ', len(features_to_remove))
    features_to_keep = [x for x in X_train.columns if x not in features_to_remove]
    print('Total features to keep: ', len(features_to_keep))
    
    return features_to_keep


def recursive_feature_addition_rf(X_train, y_train, X_test, y_test,
                                  tol=0.001, max_depth=None,
                                  class_weight=None,
                                  top_n=15, n_estimators=50, random_state=0):
    """
    Perform Recursive Feature Addition (RFA) using a RandomForestClassifier.

    This function starts with a single feature and iteratively tests adding 
    each remaining feature to evaluate its impact on the ROC AUC score. 
    Features that significantly increase the AUC score are kept.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series or np.ndarray): Training target labels.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series or np.ndarray): Test target labels.
        tol (float, optional): Minimum increase in ROC AUC required to keep a feature. Default is 0.001.
        max_depth (int, optional): Maximum depth of the trees. Default is None.
        class_weight (dict, list, str, or None, optional): Class weights. Default is None.
        top_n (int, optional): Number of top features to consider. Default is 15.
        n_estimators (int, optional): Number of trees in the forest. Default is 50.
        random_state (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        list: List of features to keep after addition.
    """
    features_to_keep = [X_train.columns[0]]
    count = 1
    model_one_feature = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    model_one_feature.fit(X_train[[X_train.columns[0]]], y_train)
    y_pred_test = model_one_feature.predict_proba(
        X_test[[X_train.columns[0]]]
    )[:, 1]
    auc_score_all = roc_auc_score(y_test, y_pred_test)
    
    for feature in X_train.columns[1:]:
        print()
        print('Testing feature: ', feature, ' which is feature ', count,
              ' out of ', len(X_train.columns))
        count += 1
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )
        model.fit(X_train[features_to_keep + [feature]], y_train)
        y_pred_test = model.predict_proba(
            X_test[features_to_keep + [feature]]
        )[:, 1]
        auc_score_int = roc_auc_score(y_test, y_pred_test)
        print('New test ROC AUC={}'.format((auc_score_int)))
        print('All features test ROC AUC={}'.format((auc_score_all)))
        diff_auc = auc_score_int - auc_score_all
        if diff_auc >= tol:
            print('Increase in ROC AUC={}'.format(diff_auc))
            print('Keep: ', feature)
            auc_score_all = auc_score_int
            features_to_keep.append(feature)
        else:
            print('Increase in ROC AUC={}'.format(diff_auc))
            print('Remove: ', feature)
    print('DONE!!')
    print('Total features to keep: ', len(features_to_keep))
    
    return features_to_keep
