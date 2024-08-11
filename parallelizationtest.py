import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import time

seed = 42
def rfc_best_features(df):
    X = df.iloc[:, 1:]  # features (all columns except the 0th column)
    y = df.iloc[:, 0].astype(float) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    k_values = np.arange(1, min(11, len(X.columns)+1))  # Range of k values to evaluate
    best_k_rfa = 0
    best_accuracy_rfa = 0
    best_features_rfa = []
    best_weights_rfa = []
    clf_rfa = None

    def evaluate_feature(feature):
        features_subset = selected_features_rfa + [feature]
        X_train_selected = X_train[features_subset]
        X_test_selected = X_test[features_subset]

        clf_rfa = RandomForestClassifier(random_state=seed)
        clf_rfa.fit(X_train_selected, y_train)
        y_pred_rfa = clf_rfa.predict(X_test_selected)
        accuracy_rfa = accuracy_score(y_test, y_pred_rfa)

        return accuracy_rfa, feature, clf_rfa.feature_importances_

    start_time = time.time()

    for k in k_values:
        remaining_features = set(X.columns)
        selected_features_rfa = []
        best_accuracy_iter = 0

        while len(selected_features_rfa) < k:
            results = Parallel(n_jobs=-1)(delayed(evaluate_feature)(feature) for feature in remaining_features)

            best_result = max(results, key=lambda x: x[0])
            best_accuracy_feature, best_feature, best_weights = best_result

            selected_features_rfa.append(best_feature)
            remaining_features.remove(best_feature)

            if best_accuracy_feature > best_accuracy_iter:
                best_accuracy_iter = best_accuracy_feature
                best_weights_iter = best_weights

        if best_accuracy_iter > best_accuracy_rfa:
            best_accuracy_rfa = best_accuracy_iter
            best_k_rfa = k
            best_features_rfa = selected_features_rfa
            best_weights_rfa = best_weights_iter

    end_time = time.time()

    print("Best k - Recursive Feature Addition:", best_k_rfa)
    print("Best Accuracy - Recursive Feature Addition:", best_accuracy_rfa)
    print("Selected Features - Recursive Feature Addition:", best_features_rfa)
    print("Feature Weights - Recursive Feature Addition:", best_weights_rfa)
    print("Execution Time:", end_time - start_time, "seconds")
