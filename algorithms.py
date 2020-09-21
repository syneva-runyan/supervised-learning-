from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import time
import numpy as np

WORLD_CUP_DATA_SET_NAME = "World cup prediction data"
df_worldcup = pd.read_csv ('./assignment_data.csv')
data_worldcup = df_worldcup[['spi', 'opposing_spi', 'spi_offense', 'opposing_spi_offense', 'spi_defense', 'opposing_spi_defense', 'sixteen']]

HEART_FAILURE_DATA_SET_NAME = 'heart_failure_prediction_data'
df_heartfailure = pd.read_csv ('./heart_failure_clinical_records_dataset.csv')
data_heartfailure = df_heartfailure[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

# helper functions
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

# end helper functions

# create test and train data
def create_train_and_test(test_size, df, data):
    n = len(df.columns)
    labels = df[df.columns[-1]]
    train, test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    return [train, test, labels_train, labels_test, test_size]

def create_all_train_and_test(df, data):
    train_and_test = []
    for test_size in my_range(0.1, 0.9, 0.1):
        n = len(df.columns)
        labels = df[df.columns[-1]]
        train, test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=1, stratify=labels)
        train_and_test.append([train, test, labels_train, labels_test, test_size])
    return train_and_test


# execute training algorithm
def run_algorithm(algorithm, algorithm_name, data_name):
    test_size = []
    time_to_train = []
    time_to_predict = []
    scores = []

    if data_name == HEART_FAILURE_DATA_SET_NAME:
        df = df_heartfailure
        independent_data = data_heartfailure
    else:
        df = df_worldcup
        independent_data = data_worldcup

    # for data in sets:
    for size in my_range(0.1, 0.9, 0.1):
        print("SIZE " + str(size))
        train_time_set = []
        predict_time_set = []
        score_set = []
        test_size.append(size)
        for i in range(0,1000):
            # print("I " + str(i))
            data = create_train_and_test(size, df, independent_data)
            train_time, predict_time, score = algorithm(data[0], data[1], data[2], data[3], data_name)
            train_time_set.append(train_time)
            predict_time_set.append(predict_time)
            score_set.append(score)
        
        # average of runs
        np_train = np.array(train_time_set)
        np_train = np.mean(np_train)
        time_to_train.append(np_train)
        np_predict = np.array(predict_time_set)
        np_predict = np.mean(np_predict)
        time_to_predict.append(np_predict)
        np_score = np.array(score_set)
        np_score = np.mean(np_score)
        scores.append(np_score)
    plot_results(test_size, time_to_train, "Time to Train", algorithm_name, data_name)
    plot_results(test_size, time_to_predict, "Time to Predict", algorithm_name, data_name)
    plot_results(test_size, scores, "Accuracy of Predictions", algorithm_name, data_name)


# plot results
def plot_results(test_size, y, yLabel, algorithm, data_name):
    fig, ax = plt.subplots()
    ax.plot(test_size, y, marker='o', drawstyle="steps-post")
    ax.set_xlabel("Test Set Size")
    ax.set_ylabel(yLabel)
    chart_title = data_name + ": Test Set Size vs " + yLabel + " for " + algorithm 
    ax.set_title(chart_title)
    plt.show()

def getScore(classifier, train, test, labels_train, labels_test):
    start = time.perf_counter()
    classifier.fit(train, labels_train)
    end = time.perf_counter()
    time_to_train = end - start
    start = time.perf_counter()
    classifier.predict(test)
    end = time.perf_counter()
    time_to_predict = end - start
    score = classifier.score(test, labels_test)
    return time_to_train, time_to_predict, score


# Algorithms
def decisionTree(train, test, labels_train, labels_test, data_name):
    cpp__alpha = 0.02142782
    if data_name == HEART_FAILURE_DATA_SET_NAME:
        cpp__alpha = 0.00610128
    else:
        cpp__alpha = 0.02142782

    classifier = DecisionTreeClassifier(ccp_alpha=cpp__alpha)
    return getScore(classifier, train, test, labels_train, labels_test)

def neuralNet(train, test, labels_train, labels_test, data_name):
    classifier = MLPClassifier(solver='adam', activation='logistic')
    return getScore(classifier, train, test, labels_train, labels_test)

def boosting(train, test, labels_train, labels_test, data_name):
    if data_name == HEART_FAILURE_DATA_SET_NAME:
        # cpp__alpha = 0.00610128
        cpp__alpha = 0.009
        # cpp__alpha = 0.012
    else:
        cpp__alpha = 0.04
    classifier = GradientBoostingClassifier(ccp_alpha=0.04)
    return getScore(classifier, train, test, labels_train, labels_test)

def support_vector_machine(train, test, labels_train, labels_test, data_name):
    classifier = SVC(kernel='rbf')
    return getScore(classifier, train, test, labels_train, labels_test)

def support_vector_machine_linear(train, test, labels_train, labels_test, data_name):
    classifier = SVC(kernel='linear', max_iter=1000)
    return getScore(classifier, train, test, labels_train, labels_test)

def k_nearest_neighbors_3(train, test, labels_train, labels_test, data_name):
    classifier = KNeighborsClassifier(n_neighbors=3)
    return getScore(classifier, train, test, labels_train, labels_test)

def k_nearest_neighbors_5(train, test, labels_train, labels_test, data_name):
    classifier = KNeighborsClassifier(n_neighbors=5)
    return getScore(classifier, train, test, labels_train, labels_test)

def k_nearest_neighbors_7(train, test, labels_train, labels_test, data_name):
    classifier = KNeighborsClassifier(n_neighbors=7)
    return getScore(classifier, train, test, labels_train, labels_test)

# Help determine an effective alpha for my training set.
# code credit scikit-learn.org
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
def decide_ccp_alpha( classifier, train, test, labels_train, labels_test):
    path = classifier.cost_complexity_pruning_path(train, labels_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()
    print(ccp_alphas)


def run_algorithms(data_name):
    run_algorithm(decisionTree, "Decision Tree", data_name)
    run_algorithm(neuralNet, "Neural Net", data_name)
    run_algorithm(boosting, "Boosting", data_name)
    run_algorithm(support_vector_machine, "Support Vector Machine: rbf kernel", data_name)
    run_algorithm(support_vector_machine_linear, "Support Vector Machine: linear kernel", data_name)
    run_algorithm(k_nearest_neighbors_3, "K-Nearest Neighbor - k=3", data_name)
    run_algorithm(k_nearest_neighbors_5, "K-Nearest Neighbor - k=5", data_name)
    run_algorithm(k_nearest_neighbors_7, "K-Nearest Neighbor - k=7", data_name)

run_algorithms(WORLD_CUP_DATA_SET_NAME)
run_algorithms(HEART_FAILURE_DATA_SET_NAME)