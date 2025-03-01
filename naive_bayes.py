#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for row in db:
    outlook = 1 if row[1] == "Sunny" else 2 if row[1] == "Overcast" else 3
    temp = 1 if row[2] == "Hot" else 2 if row[2] == "Mild" else 3
    humidity = 1 if row[3] == "High" else 2
    wind = 1 if row[4] == "Strong" else 2
    X.append([outlook, temp, humidity, wind])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in db:
    training_class = 1 if row[5] == "Yes" else 2
    Y.append(training_class)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open("weather_test.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append(row)

#Printing the header os the solution
#--> add your Python code here
        else:
            print(f"{row[0]:<7} {row[1]:<10} {row[1]:<10} {row[3]:<10} {row[4]:<8} {row[5]:<10}  Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for data in dbTest:
    outlook = 1 if data[1] == "Sunny" else 2 if data[1] == "Overcast" else 3
    temp = 1 if data[2] == "Hot" else 2 if data[2] == "Mild" else 3
    humidity = 1 if data[3] == "High" else 2
    wind = 1 if data[4] == "Strong" else 2
    prediction = clf.predict_proba([[outlook, temp, humidity, wind]])[0]
    #print(prediction)
    data[5] = "Yes" if prediction[0] > prediction[1] else "No"
    if max(prediction) >= 0.75:
        print(f"{data[0]:<7} {data[1]:<10} {data[1]:<10} {data[3]:<10} {data[4]:<8} {data[5]:<10}  {max(prediction):.3f}")