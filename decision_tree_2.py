#-------------------------------------------------------------------------
# AUTHOR: Elena Hernandez
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program aims to train, test, and output the performance of 3 models.
#                These models were created with each training set on the provided test set.
#                This process was repeated ten times, and the average accuracy showed the final classification performance of each model.
# FOR: CS 4210-Assignment #2
# TIME SPENT: The entire homework assignment took me about seven hours.
#             This program took about an hour.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining:
        age = 1 if row[0] == "Young" else 2 if row[0] == "Prepresbyopic" else 3
        prescription = 1 if row[1] == "Myope" else 2
        astigmatism = 1 if row[2] == "Yes" else 2
        tear = 1 if row[3] == "Reduced" else 2
        # adding new row into feature matrix (4D array X)
        X.append([age, prescription, astigmatism, tear])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for row in dbTraining:
        lenses = 1 if row[4] == "Yes" else 2
        Y.append(lenses) # adding new element to vector Y

    #Loop your training and test tasks 10 times here
    total_accuracy = 0
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        correct = 0
        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            age = 1 if data[0] == "Young" else 2 if data[0] == "Prepresbyopic" else 3
            prescription = 1 if data[1] == "Myope" else 2
            astigmatism = 1 if data[2] == "Yes" else 2
            tear = 1 if data[3] == "Reduced" else 2
            class_predicted = clf.predict([[age, prescription, astigmatism, tear]])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            lenses = 1 if data[4] == "Yes" else 2
            if class_predicted == lenses:
                correct += 1
        
        accuracy = correct / len(dbTest)
        total_accuracy += accuracy

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = total_accuracy / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Accuracy when training on {ds}: {avg_accuracy}")
