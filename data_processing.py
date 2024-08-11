import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier # for finding best k
from sklearn.model_selection import cross_val_score # for custom algo
import matplotlib.pyplot as plt # for plotting increasing k
import flags
normalized = flags.normalized
execute_faster = flags.execute_faster


def import_data(filename):
    df = pd.read_csv(filename, sep='\s+', header=None, engine='python', converters={0: lambda x: 'NEGATIVE' if x == '-' else x})

    # Replace 'NEGATIVE' with '-' in the DataFrame
    df.replace('NEGATIVE', '-', inplace=True)  
    return df

# https://danielcaraway.github.io/html/sklearn_cosine_similarity.html
# cosine similarity not the same as l2 or euclidean distance
# ranking them should result in same order but numbers are different
# see here: https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance

# take example X
# calculate similarity to training data
# use only specified features
# https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/#
# np.linalg.norm, np.dot, or np.sum

def predict_class(X, dataset, features_to_use, knn=1, normalized=0):
    if normalized:
        dataset = dataset.iloc[:, features_to_use + [0]]
        X_numeric = X.astype(float)

        # Normalize the dataset and X
        dataset_normalized = (dataset.iloc[:, :-1] - dataset.iloc[:, :-1].mean()) / dataset.iloc[:, :-1].std()
        X_normalized = (X_numeric - dataset.iloc[:, :-1].mean()) / dataset.iloc[:, :-1].std()

        # for sake of performance, this flag sets usage of numpy functions
        # disable to test manual implementation as requested
        # see Piazza post: https://piazza.com/class/lg1h86lj1w921f/post/68
        nearest_indices = 0

        # for each testing point, calculate L2 distance
        if execute_faster:
            distances = np.linalg.norm(dataset_normalized - X_normalized, axis=1)
            # find closest k points to X
            nearest_indices = np.argsort(distances)[:knn]
        else:
            distances = []
            for i in range(len(dataset)):
                dist = 0
                for j in range(len(features_to_use)):
                    diff = dataset.iloc[i, j] - X_numeric[j]
                    dist += diff * diff
                distances.append(np.sqrt(dist))
            nearest_indices = sorted(range(len(distances)), key=lambda k: distances[k])[:knn]
        # get the class of the k-nearest neighbors
        nearest_labels = dataset.iloc[nearest_indices][0]
        # return the majority class labe
        majority_class = nearest_labels.mode().values[0]
        return majority_class
    else:
        # select relevant features
        # include class column
        dataset = dataset.iloc[:, features_to_use + [0]]
        
        # force input to be as type float
        X_numeric = X.astype(float)

        # for sake of performance, this flag sets usage of numpy functions
        # disable to test manual implementation as requested
        # see Piazza post: https://piazza.com/class/lg1h86lj1w921f/post/68
        # execute_faster = True
        # rewritten later
        nearest_indices = 0

        # for each testing point, calculate L2 distance
        if execute_faster:
            distances = np.linalg.norm(dataset.iloc[:, :-1] - X_numeric, axis=1)
            # find closest k points to X
            nearest_indices = np.argsort(distances)[:knn]
        else:
            distances = []
            for i in range(len(dataset)):
                dist = 0
                for j in range(len(features_to_use)):
                    diff = dataset.iloc[i, j] - X_numeric[j]
                    dist += diff * diff
                distances.append(math.sqrt(dist))
            nearest_indices = sorted(range(len(distances)), key=lambda k: distances[k])[:knn]
    
        # get the class of the k-nearest neighbors
        nearest_labels = dataset.iloc[nearest_indices][0]
        
        # return the majority class label
        majority_class = nearest_labels.mode().values[0]
        return majority_class

# given dataset, feature selection
# calculate accuracy
def accuracy_calc(df, columns_to_use, knn=1):
    predicted_correct = 0

    # iterate through entire dataframe for k fold validation
    for row in df.index:
        # drop current row from dataframe
        # create copy of data w/o current row
        df_copy = df.drop(index=row).reset_index(drop=True)

        # extract values from row
        features_in_column = df.iloc[row][columns_to_use].values

        # make a prediction
        prediction = predict_class(features_in_column, df_copy, columns_to_use, knn=knn)
        # print("row")
        # print(df.iloc[row][0][0])
        # print("prediction")
        # print(prediction)
        # print("-----")

        # if prediction correct increment count
        if prediction == df.iloc[row][0]:
            # print(prediction)
            predicted_correct += 1

    # return accuracy 
    return (predicted_correct/len(df))

def best_k(df):
    if execute_faster == 0:
        print("not supported")
        exit()
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    max_k = int(np.ceil(np.sqrt(df.shape[0]))+2)
    # print(df.shape[0])
    # max_k = int(df.shape[0]/2)
    k_values = list(range(1, max_k, 2))  #try k values from 1 to max_k, skipping evens

    mean_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, features, labels, cv=5)
        mean_scores.append(scores.mean())

    # Find the best k value
    best_k_val = k_values[np.argmax(mean_scores)]
    print("Best k value:", best_k_val)
    plt.plot(k_values, mean_scores)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k')
    plt.show()
    return best_k_val

def plot_accuracy_vs_k(df, columns_to_use, k_values):
    accuracies = []
    for k in k_values:
        accuracy = accuracy_calc(df, columns_to_use, knn=k)
        accuracies.append(accuracy)

    plt.plot(k_values, accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k')
    plt.show()


