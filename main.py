from search import *
from data_processing import *
import itertools
import numpy as np
import random
import time
from flags import *
from parallelizationtest import *

df = import_data('data/small-test-dataset.txt')
df2 = import_data('data/large-test-dataset.txt')
df3 = import_data('data/CS170_Spring_2023_Small_data__28.txt')
df4 = import_data('data/CS170_Spring_2023_Large_data__28.txt')

def feature_selection():
    k = 1

    print("Welcome to Group 28's Feature Selection Algorithm.\n")
    print(f"Flags enabled: normalized -> {int(normalized)}, execute_fast -> {int(execute_faster)}")

    while True:
        # ask for the algorithm option
        print("\nType the number of the option to run.\n")
        print("1. Forward Selection")
        print("2. Backward Elimination")
        print("3. Brute Force (larger dataset will crash)")
        print("4. Test 1-NN Classifier Accuracy with Leave One Out Validation")
        print("5. Custom Algorithm - k search -- Forward") # choose the best one between this and option 6
        print("6. Custom Algorithm - k search-- Backward") #
        print("7. Plot Accuracy vs Increasing K")
        print("8. Custom Algorithm -  Random Forest to find best features ")
        print("---\nOptions:")
        print("10: Set k for k-NN")
        print("11: Test features over k-NN")
        algorithm = int(input("\nSelect option: "))

        if algorithm in [1, 2, 3, 5, 6, 8]:
            # ask for dataset selection
            selected_db = int(input("Select dataset:\n\t1: small-test-dataset.txt\n\t2: large-test-dataset.txt\n\t3: CS170_Spring_2023_Small_data__28.txt\n\t4: CS170_Spring_2023_Large_data__28.txt\nChoice: "))

            if selected_db == 1:
                df_selected = df
            elif selected_db == 2:
                df_selected = df2
            elif selected_db == 3:
                df_selected = df3
            else:
                df_selected = df4

            df_keys = df_selected.columns.tolist()
            df_keys.remove(0)

            print(df_keys)
            
            no_features = accuracy_calc(df_selected, [], knn=k)
            print("\nUsing no features and 'random' evaluation, I get an accuracy of {}%".format(no_features))
            print("\nBeginning search.\n")

            start_time = time.time()

        if algorithm == 1:
            forward_search(df_selected, knn=k)
        elif algorithm == 2:
            backward_search(df_selected, knn=k)
        elif algorithm == 3:
            brute_force(get_combos(df_keys), df_selected, knn=k)
            pass
            # elif algorithm == 5:
            #     print("brute force...")
            #     brute_force(get_combos(df_keys), df_selected, knn=k)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
        
        elif algorithm == 4:
            # leave one out validation and testing
            columns_to_use_small = [3, 5, 7]
            columns_to_use_large = [1, 15, 27]
            
            counts_small = accuracy_calc(df, columns_to_use_small, knn=k)
            counts_large = accuracy_calc(df2, columns_to_use_large, knn=k)
            
            print("\nTesting on small dataset using features {}, result should be around 0.89".format(columns_to_use_small))
            print("Accuracy using features {} is: {}".format(columns_to_use_small, counts_small))

            print("\nTesting on large dataset using features {}, result should be around 0.949".format(columns_to_use_large))
            print("Accuracy using features {} is: {}".format(columns_to_use_large, counts_large))
        
        elif algorithm == 5:
            print("find best k...")
            best_k_val = best_k(df_selected)
            forward_search(df_selected, knn = best_k_val)
            # this shit is way more complicated than it has to be...
            # feature_combinations = get_combos(df_keys)
            # if selected_db == 1:
            #     brute_force(get_combos(df_keys), df, knn=k)
            # else:
            #     # crashes lol
            #     brute_force(get_combos(df_keys), df2, knn=k)
        elif algorithm == 6:
            print("finding best k...")
            best_k_val = best_k(df_selected)
            backward_search(df_selected, knn = best_k_val)
        elif algorithm == 7:
            # columns_to_use = list(range(1, 11))
            columns_to_use = [9, 3, 35, 4, 24, 20]
            k_values = list(range(1, 50,2))
            plot_accuracy_vs_k(df4, columns_to_use, k_values)
        elif algorithm == 8:
            rfc_best_features(df_selected)
        elif algorithm == 10:
            # set k for k-NN
            print("Set custom k for k-NN")
            number = int(input("Enter a number: "))

            if number % 2 == 0:
                print("Even number entered!")
                exit()
            k = number
            # forward_search(df, knn=k)

        elif algorithm == 11:
            # test features over k-NN
            print("Test features over k-NN")
            max_k = int(input("Maximum k: "))
            columns_to_use = [1, 15, 27]

            for i in range(1, max_k+1, 2):
                counts = accuracy_calc(df2, columns_to_use, knn=i)
                print("{}-NN Accuracy using features {} is: {}".format(i, columns_to_use, counts))
            exit()
        
        else:
            print("Invalid option. Please try again.")

feature_selection()