from itertools import combinations
import random
from data_processing import *

########
# HELPER FUNCTIONS
########

def check_variables_in_tuple(variables, my_tuple):
    return all(var in my_tuple for var in variables)

# accept a list of elements (column names)
# return all possible combinations
# nested list of combinations
# index + 1 indicates number of feature combinations
# from 1 combination to all n combinations
# def get_combos(featurecount):
#     combinations_list = [] 
#     for r in range(1, len(featurecount) + 1):
#         combos = []  # Initialize a temporary list for combinations of current size r
#         for combination in combinations(featurecount, r):
#             combos.append(combination)
#         combinations_list.append(combos)  # Add the temporary list to the nested list
#     return combinations_list

# 2^n possible combinations of up to n features...
# https://math.stackexchange.com/questions/1117236/calculate-the-total-number-of-combinations-over-n-elements-where-the-number-of
def get_combos(featurecount):
    for r in range(1, len(featurecount) + 1):
        combos = []
        for combo in combinations(featurecount, r):
            combos.append(combo)
        yield combos

########
# END HELPER FUNCTIONS
########

def forward_search(df, knn=1):
    best_accuracy = 0.0
    best_features = []
    selected_features = []

    feature_count = len(df.columns)

    for i in range(feature_count):
        local_best_accuracy = 0.0
        local_best_feature = None

        remaining_features = list(set(range(1, feature_count)) - set(selected_features))        
        for feature in remaining_features:
            tmp_features = selected_features + [feature]
            accuracy = evaluate_accuracy(tmp_features, df, knn=knn)
            print("Features:", tmp_features)
            print("Accuracy:", accuracy)
            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                local_best_feature = feature

        selected_features.append(local_best_feature)
        if best_accuracy > local_best_accuracy:
            print("\tWarning, accuracy has decreased!")

        if local_best_accuracy > best_accuracy:
            best_accuracy = local_best_accuracy
            best_features = selected_features.copy()
        print("At level {}, feature set {} was best, accuracy is {}\n".format(i, local_best_feature, local_best_accuracy))
    print("Finished search!! The best feature subset is", best_features, "which has an accuracy of ", f"{best_accuracy:.1%}")
    # return best_features, best_accuracy

    # please fix the menu
    exit()
    pass

def backward_search(df, knn=1):
    best_accuracy = 0.0
    best_features = list(df.columns[1:])
    selected_features = list(df.columns[1:])

    feature_count = len(df.columns) - 1

    for i in range(feature_count):
        local_best_accuracy = 0.0
        # local_worst_accuracy = 999
        local_worst_feature = None
        local_best_feature = None

        for feature in selected_features:
            remaining_features = list(set(selected_features) - set([feature]))
            accuracy = evaluate_accuracy(remaining_features, df, knn=knn)

            print("Features:", remaining_features)
            print("Accuracy:", accuracy)

            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                local_best_feature = remaining_features

        selected_features = local_best_feature

        if local_best_accuracy > best_accuracy:
            print("\tWarning, accuracy has decreased!")

        if local_best_accuracy > best_accuracy:
            best_accuracy = local_best_accuracy
            best_features = selected_features

        print("At level {}, feature {} was best, accuracy is {}\n".format(i, local_best_feature, local_best_accuracy))

    print("Finished search!! The best feature subset is", best_features, "which has an accuracy of", f"{best_accuracy:.1%}")

    # return best_features, best_accuracy

    # please fix the menu
    exit()
    pass

# NOT tuple of features
# accuracy_calc expects list
# just pass in list
def evaluate_accuracy(features, df, knn=1):
    # accuracy = np.random.uniform(0, 1)
    # return random.random()
    return accuracy_calc(df, features, knn=knn)


def brute_force(feature_combinations, df, knn=1):
    best_accuracy = 0.0
    best_features = None

    nesting_level = 1
    selected_features = []
    for features_at_level in feature_combinations:
        local_best_feature = None
        local_best_accuracy = 0.0
        for features in features_at_level:
            print("Feature(s):", features)
            accuracy = evaluate_accuracy(list(features), df, knn=knn)
            print("Accuracy:", accuracy)
            if accuracy > local_best_accuracy:
                local_best_accuracy = accuracy
                local_best_feature = features
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = features
        if best_accuracy > local_best_accuracy:
            print("\tWarning, accuracy has decreased!")

        for element in local_best_feature:
            if element not in selected_features:
                selected_features.append(element)
        print("At level {}, feature set {} was best, accuracy is {}\n".format(nesting_level, local_best_feature, local_best_accuracy))
        nesting_level += 1

    # print highest accuracy combination of features
    print("Finished search!! The best feature subset is", best_features, "which has an accuracy of ", f"{best_accuracy:.1%}")
    # please fix the menu
    exit()
    pass
