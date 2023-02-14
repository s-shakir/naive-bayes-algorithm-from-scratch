import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from math import e, pi

def loaddata():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header = None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Species']

    #dividing data
    #every 15th row will be selected for test_data except for the last column
    test_data = df.iloc[::15, :-1]
    #every row apart from every 15th row will be selected for train_data
    train_data = df[df.index % 15 != 0]
    #used later for checking accuracy
    check = df.iloc[::15]
    actual = list(check.Species)

    print("\n-----------------Test Data------------------\n")
    print(test_data, "\n")
    print("\n-----------------Train Data------------------\n")
    print(train_data, "\n")
    print("\n-----------------Actual Class Labels for checking------------------\n")
    print(actual, "\n")
    return test_data, train_data, check, actual

def class_prob(train_data):
    #calculating frequency for each class
    iris_set_cnt = train_data['Species'][train_data['Species']=='Iris-setosa'].count()
    iris_ver_cnt = train_data['Species'][train_data['Species']=='Iris-versicolor'].count()
    iris_vir_cnt = train_data['Species'][train_data['Species']=='Iris-virginica'].count()

    total = train_data['Species'].count()

    #calculating class probability
    iris_set_prob = iris_set_cnt/total
    iris_ver_prob = iris_ver_cnt/total
    iris_vir_prob = iris_vir_cnt/total

    print("\n-----------------Class Probabilities------------------\n")
    print("\n-----------------Iris-setosa Class Probability------------------\n")
    print(iris_set_prob, "\n")
    print("\n-----------------Iris-versicolor Class Probability------------------\n")
    print(iris_ver_prob, "\n")
    print("\n-----------------Iris-virginica Class Probability------------------\n")
    print(iris_vir_prob, "\n")

    return total, iris_set_prob, iris_ver_prob, iris_vir_prob

def normal_dist(train_data):
    # group data by class and calculate means & variance of each feature
    tr_m = train_data.groupby('Species').mean()
    tr_var = train_data.groupby('Species').var()

    print("\n-----------------Calculating Mean------------------\n")
    print(tr_m, "\n")
    print("\n-----------------Calculating Variance------------------\n")
    print(tr_var, "\n")

    return tr_m, tr_var

def prob(f, m, v):
    #applying normal dist formula from wikipedia page
    #f is feature, m is mean, v is variance
    prob = np.exp((-1/2)*((f-m)**2) / (2 * v))
    return prob

def biggest(iris_set, iris_ver, iris_vir):
    #checking which label has the highest prob and assign that as a final label/class
    Max = iris_set
    label = "Iris-setosa"
    if iris_ver > Max:
        Max = iris_ver
        label = "Iris-versicolor"
    if iris_vir > Max:
        Max = iris_vir
        label = "Iris-virginica"
        if iris_ver > iris_vir:
            Max = iris_ver
            label = "Iris-versicolor"
    return label

def predict(test_data, tr_m, tr_var, iris_set_prob, iris_ver_prob, iris_vir_prob):
    #applying  the bayes classifier
    #numerator = prob(class)*prob(sepal_length∣class)*prob(sepal_width∣class)*prob(petal_length∣class)*prob(petal_width∣class)
    predicted = []
    for i in range(10): #loop for iterating through test_data rows
            for k in range(3): #loop for iterating through mean and variance data

                #calling probability function for each row element in test_data, mean and variance
                #calculating probability for each feature
                sep_len_prob = prob(test_data.iloc[i]['sepal_length'], tr_m.iloc[k]['sepal_length'], tr_var.iloc[k]['sepal_length'])
                sep_wid_prob = prob(test_data.iloc[i]['sepal_width'], tr_m.iloc[k]['sepal_width'], tr_var.iloc[k]['sepal_width'])
                pet_len_prob = prob(test_data.iloc[i]['petal_length'], tr_m.iloc[k]['petal_length'], tr_var.iloc[k]['petal_length'])
                pet_wid_prob = prob(test_data.iloc[i]['petal_width'], tr_m.iloc[k]['petal_width'], tr_var.iloc[k]['petal_width'])

                #calculating numerator value from above bayes classifier
                if(k==0): #condition for calculating probability of class Iris-setosa since k in looping through mean and variance data so it will run only when row representing the class is found
                    iris_set = iris_set_prob*sep_len_prob*sep_wid_prob*pet_len_prob*pet_wid_prob

                if(k==1):#condition for calculating probability of class Iris-versicolor since k in looping through mean and variance data so it will run only when row representing the class is found
                    iris_ver = iris_ver_prob*sep_len_prob*sep_wid_prob*pet_len_prob*pet_wid_prob

                if(k==2): #condition for calculating probability of class Iris-virginica since k in looping through mean and variance data so it will run only when row representing the class is found
                    iris_vir = iris_vir_prob*sep_len_prob*sep_wid_prob*pet_len_prob*pet_wid_prob

            lb = biggest(iris_set, iris_ver, iris_vir)#calling biggest function to check which class probability is higher and assign that class label as predicted value
            predicted.append(lb)

    print("\n-----------------Predicted Class Labels------------------\n")
    print(predicted, "\n")

    return predicted

def convert(x):
    for index, item in enumerate(x):

        #converting string values of labels to numeric represent so predicted and actual values could be compare and accuracy can be calculated
        if(x[index] == 'Iris-setosa'):
            x[index] = 0
        elif(x[index] == 'Iris-versicolor'):
            x[index] = 1
        elif(x[index] == 'Iris-virginica'):
            x[index] = 2

def accuracy(actual, predicted, test_data):

    #first convert labels to int for comparison
    actual = convert(actual)
    predicted = convert(predicted)
    correct = 0
    for i in range(10):
        if(predicted == actual):
            correct += 1
    res = correct / float(len(test_data))
    result = res*100
    print("\n-----------------Model Accuracy------------------\n")
    print("the accuracy of the model is: ", result)

def main():
    test_data, train_data, check, actual = loaddata()
    #model training
    total, iris_set_prob, iris_ver_prob, iris_vir_prob = class_prob(train_data)
    tr_m, tr_var = normal_dist(train_data)
    #model prediction
    predicted = predict(test_data, tr_m, tr_var, iris_set_prob, iris_ver_prob, iris_vir_prob)
    #checking accuracy
    accuracy(actual, predicted, test_data)

if __name__ == '__main__':
    main()
