# Implement the functions using Math library

import math

while True:
    print("------------------------------------------------")
    print("---------- Math module operation ----------")
    print("------------------------------------------------")
    print("1.CopySign\n2.Factorial\n3.Fmod\n4.Fsum\n5.Isinf\n6.dist\n7.nextafter\n8.GCD\n9.Power")
    print("------------------------------------------------")
    op=int(input("Enter the option : "))
    if op==1:
        a=int(input("Enter the first value : "))
        b=int(input("Enter the second value : "))
        print(math.copysign(a,b)) #Returns the first parameter with second parameter sign
    elif op==2:
        a=int(input("Enter the value : "))
        print(math.factorial(a)) # Returns the factorial value
    elif op==3:
        a=int(input("Enter the first value : "))
        b=int(input("Enter the second value : "))
        print(math.fmod(5,3)) #Returns the modulo division value
    elif op==4:
        lst=[10,20,30,40,50]
        print("Sum of {0} is : {1}".format(lst,math.fsum(lst))) #Returns the sum of the sequence
    elif op==5:
        a=math.inf
        print("The result of {0} is : {1}".format(a,math.isinf(a))) #Return true if the given value is infinate otherwise false
    elif op==6:
        x1=int(input("Enter the x value of point 1 : "))
        y1=int(input("Enter the y value of point 1 : "))
        x2=int(input("Enter the x value of point 2 : "))
        y2=int(input("Enter the y value of point 2 : "))
        print("The distance between ({0},{1}) and ({2},{3}) is : {4}".format(x1,y1,x2,y2,math.dist([x1,y1],[x2,y2]))) # Returns the distance between two points sqrt((x2-x1)^2+(y2-y1)^2)
    elif op==7:
        a=int(input("Enter the first value : "))
        b=int(input("Enter the second value : "))
        print("The next float value of {0} between {1} and {2} is : {3}".format(a,a,b,math.nextafter(a,b))) #Returns the next float value of x between x to y
    elif op==8:
        a=int(input("Enter the first value : "))
        b=int(input("Enter the second value : "))
        print("The gcd of {0} and {1} is : {2}".format(a,b,math.gcd(a,b))) #Returns the GCD value
    elif op==9:
        a=int(input("Enter the base value : "))
        b=int(input("Enter the power value : "))
        print("The powered value of {0} and {1} is : {2}".format(a,b,math.pow(a,b))) #Returns the Power value
    else:
        break

# Create and manipulate multi dimensional array using Numpy

import numpy as np
def matrix_addition(matrix1, matrix2):
return matrix1 + matrix2
def matrix_subtraction(matrix1, matrix2):
return matrix1 - matrix2
def matrix_transpose(matrix):
return np.transpose(matrix)
def matrix_inverse(matrix):
return np.linalg.inv(matrix)
def matrix_determinant(matrix):
return np.linalg.det(matrix)
def matrix_access_element(matrix, row, col):
return matrix[row, col]
def get_matrix_from_input():
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
matrix = []
for i in range(rows):
row = []
for j in range(cols):
element = float(input(f"Enter element at position ({i}, {j}): "))
row.append(element)
matrix.append(row)
return np.array(matrix)
def numpy_switch(choice):
switcher = {
1: "Matrix Addition",
2: "Matrix Subtraction",
3: "Matrix Transpose",
4: "Matrix Inverse",
5: "Matrix Determinant",
6: "Matrix Access Element",
}
print("\nYou selected:", switcher.get(choice, "Invalid Choice"))
if choice == 1:
print("Enter Matrix 1:")
matrix1 = get_matrix_from_input()
print("Enter Matrix 2:")
matrix2 = get_matrix_from_input()
result = matrix_addition(matrix1, matrix2)
print("Result of Matrix Addition:")
print(result)
elif choice == 2:
print("Enter Matrix 1:")
matrix1 = get_matrix_from_input()
print("Enter Matrix 2:")
matrix2 = get_matrix_from_input()
result = matrix_subtraction(matrix1, matrix2)
print("Result of Matrix Subtraction:")
print(result)
elif choice == 3:
print("Enter Matrix:")
matrix = get_matrix_from_input()
result = matrix_transpose(matrix)
print("Result of Matrix Transpose:")
print(result)
elif choice == 4:
print("Enter Matrix:")
matrix = get_matrix_from_input()
result = matrix_inverse(matrix)
print("Result of Matrix Inverse:")
print(result)
elif choice == 5:
print("Enter Matrix:")
matrix = get_matrix_from_input()
result = matrix_determinant(matrix)
print("Determinant of Matrix:")
print(result)
elif choice == 6:
print("Enter Matrix:")
matrix = get_matrix_from_input()
row = int(input("Enter row index: "))
col = int(input("Enter column index: "))
result = matrix_access_element(matrix, row, col)
print(f"Element at ({row}, {col}):", result)
else:
print("Invalid choice. Please try again.")
def main():
print("\nChoose a function:")
print("1: Matrix Addition")
print("2: Matrix Subtraction")
print("3: Matrix Transpose")
print("4: Matrix Inverse")
print("5: Matrix Determinant")
print("6: Matrix Access Element")
print("0: Exit")
while True:
choice = int(input("\nEnter the function you want to use: "))
if choice == 0:
print("Exiting the program.")
break
else:
numpy_switch(choice)
if name == " main ":
main()

# Perform statistical functions using Statistics library
import math
def main():
while True:
num_elements = int(input("Enter the number of elements: "))
numbers = []
for i in range(num_elements):
num = float(input(f"Enter number {i+1}: "))
numbers.append(num)
print("\nSelect an option:")
print("1. Mean")
print("2. Median")
print("3. Mode")
print("4. Standard Deviation")
print("5. Variance")
print("6. Exit")
option = input("Enter your choice (1-6): ")
if option == '1':
print(f"Mean: {calculate_mean(numbers)}")
elif option == '2':
print(f"Median: {calculate_median(numbers)}")
elif option == '3':
print(f"Mode: {calculate_mode(numbers)}")
elif option == '4':
print(f"Standard Deviation: {calculate_standard_deviation(numbers)}")
elif option == '5':
print(f"Variance: {calculate_variance(numbers)}")
elif option == '6':
print("Exiting the program.")
break
else:
print("Invalid option!")
def calculate_mean(numbers):
return sum(numbers) / len(numbers) if len(numbers) > 0 else 0
def calculate_median(numbers):
sorted_numbers = sorted(numbers)
n = len(sorted_numbers)
if n % 2 == 0:
return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
else:
return sorted_numbers[n//2]
def calculate_mode(numbers):
frequency = {}
for num in numbers:
if num in frequency:
frequency[num] += 1
else:
frequency[num] = 1
max_freq = max(frequency.values())
mode = [num for num, freq in frequency.items() if freq == max_freq]
return mode
def calculate_variance(numbers):
mean = calculate_mean(numbers)
squared_diff = sum((x - mean) ** 2 for x in numbers)
return squared_diff / len(numbers)
def calculate_standard_deviation(numbers):
variance = calculate_variance(numbers)
return math.sqrt(variance)
if name == " main ":
main()



# Dataframe implementation using Pandas

import pandas as pd

print("---------------------------------------------")
print("---------- Data Frame using Pandas ----------")
print("---------------------------------------------")
rows=int(input("Enter the row size of the dataframe : "))
cols=int(input("Enter the column size of the dataframe : "))
column=[]
for x in range(cols):
    column.append(input("Enter the column {0} : ".format(x+1)))
data=[]
for x in range(rows):
    col=[]
    print("\nRecord {0}".format(x+1))
    for y in range(cols):
        v=input("Enter the value for {0} : ".format(column[y]))
        #col.append(int(v) if v.isdigit() else "NaN")
        col.append(v)
    data.append(col)

df=pd.DataFrame(data,columns=column)
print("Dataframe : \n")
print(df)

while True:
    print("\n\n1.Information\n2.Description\n3.Index\n4.Columns\n5.Axes\n6.Size\n7.Shape\n8.Ndim\n9.Transpose\n10.Exit")
    op=int(input("\nEnter your choice : "))
    if op==1:
        print("\n\nInformation\n")
        print(df.info())
    elif op==2:
        print("\n\nDescribtion \n",df.describe())
    elif op==3:
        print("\nIndexes are : ",df.index)
    elif op==4:
        print("\nColumns are : ",df.columns)
    elif op==5:
        print("\nAxes are : ",df.axes)
    elif op==6:
        print("\nThe size is : ",df.size)
    elif op==7:
        print("\nThe shape is : ",df.shape)
    elif op==8:
        print("\nThe Ndim is : ",df.ndim)
    elif op==9:
        print(df.T)
    elif op==10:
        break
        exit(0)

# Program to find and handle the missing values in a dataset

import pandas as pd
import numpy as np

print("-----------------------------------------------------")
print("---------- Working with missing values ----------")
print("-----------------------------------------------------")
print("1.Create dataset by manually\n2.Use external dataset\n")
ch=int(input("Enter your choice : "))
if(ch==1):
    dataset = pd.DataFrame(np.random.randn(5, 3),
    index=['a', 'c', 'e', 'f', 'h'],columns=['stock1','stock2', 'stock3'])
    dataset.rename(columns={"one":'stock1',"two":'stock2',"three":'stock3'}, inplace=True)
    dataset = dataset.reindex(['a', 'b', 'c', 'd', 'e','f', 'g', 'h'])
elif ch==2:
    dataset=pd.read_csv("missing_value.csv")
pd.set_option('display.max_columns', None)  
while True:
    print("\n1.IsNull\n.2.NotNull\n3.FillNa\n4.DropNa\n5.Replace")
    op=int(input("\nEnter your option : "))
    print("\nThe original Dataset : ")
    print(dataset.head(10))
    if(op==1):
        print("\nThe result for null is : ")
        print(pd.isnull(dataset.head(10)))
    elif(op==2):
        print("\nThe result for notnull is : ")
        print(pd.notnull(dataset.head(10)))
    elif op==3:
        print("\nThe result for fillna is : ")
        print(dataset.fillna(dataset.mean() if ch==1 else 1))
    elif op==4:
        print("\nThe result for dropna is : ")
        print(dataset.dropna())
    elif op==5:
        value=int(input("Enter the value for replace : "))
        print("\nThe result for replace is : ")
        print(dataset.replace(np.nan,value))
    else:
        break

# Load a dataset and display its structure with statistical summaries using Pandas

import pandas as pd

print("--------------------------------------------------------------------------------------------------------------")
print("Load a Dataset and Understand its Structure with Statistical Summaries using pandas library")
print("--------------------------------------------------------------------------------------------------------------")

data = pd.read_csv('marksheet.csv')
pd.set_option('display.max_columns', None)
print("The given dataset is : \n",data.head(30))
while True:
    print("1. HEAD\n2. TAIL\n3.DESCRIBE\n4. MEAN\n5. MEDAIN\n6. GROUPBY\n7. SHAPE\n8. COLUMNS\n9. INFO\n10. EXIT")
    print("--------------------------------------------------------------------")
    choice = int(input("Enter your choice : "))
    print("--------------------------------------------------------------------")
    if choice == 1:
        print("The first default selected rows in the dataset :\n",data.head())
        print("--------------------------------------------------------------------")
    elif choice == 2:
        print("The last default selected rows in the dataset :\n",data.tail())
        print("--------------------------------------------------------------------")
    elif choice == 3:
        print("The describe of the dataset is :\n",data.describe())
        print("--------------------------------------------------------------------")
    elif choice == 4:
        column = input("Enter the column name to make a group from the following ('Age', 'Science', 'Maths','History','English') : ")
        print("--------------------------------------------------------------------")
        try:
            print("The mean of the dataset is :\n",data[column.capitalize()].mean())
        except:
            print("Only numerical value columns are allowed")
        print("--------------------------------------------------------------------")
    elif choice == 5:
        column = input("Enter the column name to make a group from the following ('Age', 'Science', 'Maths','History','English') : ")
        print("--------------------------------------------------------------------")
        try:
            print("The median of the dataset is :\n",data[column.capitalize()].median())
        except:
            print("Only numerical value columns are allowed")
        print("--------------------------------------------------------------------")
    elif choice == 6:
        columns = ['Age', 'Gender', 'Section']
        column = input("Enter the column name to make a group from the following ('Age', 'Gender', 'Section') : ")
        print("--------------------------------------------------------------------")
        if column.capitalize() in columns:
            print("The groupby of ",column.capitalize()," and Count is  \n",data.groupby(column.capitalize()).count())
        else:
            print("Invalid Column name.....")
        print("--------------------------------------------------------------------")
    elif choice == 7:
        print("The shape of the dataset is : ",data.shape)
        print("--------------------------------------------------------------------")
    elif choice == 8:
        print("The columns of the dataset is : ",data.columns)
        print("--------------------------------------------------------------------")
    elif choice == 9:
        print("The info of the dataset is : ",data.info())
        print("--------------------------------------------------------------------")            
    elif choice == 10:
        print("System is exiting........")
        print("--------------------------------------------------------------------")
        exit()
    else:
        print("__WARNING__ : INVALID CHOICE........!")

# Load a dataset and display its structure with graphs using Pandas and Matplotlib

import pandas  as pd
import matplotlib.pyplot as plt

print("-------------------------------------------------------------------------------------------------------------------------")
print("DATA VISUALIZATION FOR STUDENT EXAM RESULTS")
print("-------------------------------------------------------------------------------------------------------------------------")

data = pd.read_csv('marksheet.csv')

df = pd.DataFrame(data)
while True:
    print("1. LINEAR GRAPH\n2. BAR GRAPH \n3. PIE CHART\n4. SCATTER PLOT\n5. EXIT")
    print("---------------------------------------------------")
    choice = int(input("Enter your choice : "))
    print("---------------------------------------------------")
    xdata=list(df['Section'].unique())
    xdata.sort()
    ydata=list(df.groupby("Section").count()["Age"])
    if choice == 1:
        plt.title("Number of students by section wise",fontsize=14, color="green")
        plt.plot(xdata,ydata)
        plt.xlabel('Section')
        plt.ylabel('Students count')
        plt.show()
    elif choice == 2:
        plt.title("Number of students by section wise",fontsize=14, color="blue")
        plt.bar(xdata, ydata, color ='blue', width = 0.4)
        plt.xlabel('Section')
        plt.ylabel('Student\'s Count')
        plt.show()
    elif choice == 3:
        plt.title("Number of students by section wise",fontsize=14, color="red")
        plt.pie(ydata, labels=xdata)
        plt.show()
    elif choice == 4:
        plt.title("Number of students by section wise",fontsize=14, color="navy")
        plt.scatter(xdata,ydata, c='navy')
        plt.xlabel('Section')
        plt.ylabel('Student\'s count')
        plt.show()
    elif choice == 5:
        print("System is exiting..........")
        print("---------------------------------------------------")
        exit()
    else:
        print("_WARNING_MSG : || INVALID CHOICE ||")
        print("---------------------------------------------------")

# Implementation of Covariance and Correlation
import pandas as pd
import math

data=pd.read_csv('flight_dataset.csv')

def covariance():
    global data
    x_series=data["Duration_hours"]
    y_series=data["Price"]

    x_mean=x_series.mean()
    y_mean=y_series.mean()

    cov=0
    for i in range(0,len(x_series)):
        cov+=(x_series[i]-x_mean)*(y_series[i]-y_mean)
    return cov/(len(x_series)-1)

def correlations():
    global data
    x_series=data["Duration_hours"]
    y_series=data["Price"]

    x_mean=x_series.mean()
    y_mean=y_series.mean()

    x_std=(math.fsum([(xi-x_mean)**2 for xi in x_series])/(len(x_series)-1))**0.5
    y_std=(math.fsum([(yi-y_mean)**2 for yi in y_series])/(len(y_series)-1))**0.5
    #std_y = (math.fsum([(yi - mean_x) ** 2 for yi in y]) / (len(y) - 1)) ** 0.5

    cov=covariance()

    return cov/(x_std*y_std)

while True:
    print("\n\n--------------------------------------")
    print("1.Covariance\n2.Correlations\n3.Exit")
    op=int(input("\nEnter your choice : "))
    if op==1:
        print("The covariance between Durations and Prices is ",covariance())
    elif op==2:
        print("The correlations between Durations and Prices is ",correlations())
    elif op==3:
        exit(0)

# Perform Normalization in a given dataset
import pandas as pd
import math

def mean(x):
       return (math.fsum(x) / len(x))

def standard_dev(x):
       mean_x = mean(x)
       std_x = (math.fsum([(xi - mean_x) ** 2 for xi in x]) / (len(x) - 1)) ** 0.5
       return std_x

def min_max_normalization(x):
       min_x = min(x)
       max_x = max(x)
       new_min = 0.0
       new_max = 1.0
       for i in range(len(x)):
              norm_x = ((x[i] - min_x) / (max_x - min_x))*(new_max - new_min)+new_max
              print("Normalization form of ",x[i]," is %.2f" %norm_x)

def z_score_normalization(x):
       mean_x = mean(x)
       std_x = standard_dev(x)
       for i in range(len(x)):
              print("Normalization form of ",x[i]," is %.2f" %math.fabs((x[i] - mean_x) / std_x))

def decimal_scaling_normalization(x):
       power = int((math.log10(max(x))+1)-1)
       print(power)
       denominator = 10 ** power
       print(denominator)
       for i in range(len(x)):
              print("Normalization form of ",x[i]," is %.2f" %(x[i] / denominator))

def getColumns(df):
       ch = input("Are you want to see the columns in the Dataset (y / n): ")
       if (ch.lower() == "yes" or ch.lower() == "y"):
              print("The columns in the Dataset :\n-----------------------------------------------------\n",df.columns)
              print("-----------------------------------------------------")
       else:
              print("_MSG_ : Further operations will be proceed.....")
       print("-----------------------------------------------------")
       

print("-----------------------------------------------------")
print("-------- NORMALIZATION IN DATASET -----------")
print("-----------------------------------------------------")

df = pd.read_csv("flight_dataset.csv")
df.dropna(inplace=True)
df=df[0:10]
print("The Taken Dataset Values are \n-----------------------------------------------------\n",df.head(8))

while True:
       print("---------------------------------------------")
       print("\n1. MIN-MAX NORMALIZATION\n2. DECMAL SCALING NORMALIZATION\n3. Z-SCORE NORMALIZATION\n4. EXIT")
       print("---------------------------------------------")

       choice = int(input("Enter your choice : "))

       try:
              if choice == 1:
                     print("<- - - - Please choose the one numerical value columns - - - ->")
                     getColumns(df)
                     column = input("Enter the  Column name : ")
                     print("-----------------------------------------------------")
                     x = list(df[column])
                     min_max_normalization(x)

              elif choice == 2:
                     print("<- - - - Please choose the one numerical value columns - - - ->")
                     getColumns(df)
                     column = input("Enter the  Column name : ")
                     print("-----------------------------------------------------")
                     x = list(df[column])
                     decimal_scaling_normalization(x)

              elif choice == 3:
                     print("<- - - - Please choose the one numerical value columns - - - ->")
                     getColumns(df)
                     column = input("Enter the  Column name : ")
                     print("-----------------------------------------------------")
                     x = list(df[column])
                     z_score_normalization(x)

              elif choice == 4:
                     print("-----------------------------------------------")
                     print("------------ System is Exiting ----------------")
                     print("-----------------------------------------------")
                     exit(0)

              else:
                     print("WARN : You Choose Invalid Choice ......!!!")

       except:
              print("ERROR : You Choose Invalid Column ......!!!")

# Home Price Prediction using Linear Regression for error based learning

import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

df=pd.read_csv("home_price.csv")

plt.scatter(df.Area,df.Amount)
plt.title("House price based on area")
plt.xlabel("Area (in sqft)")
plt.ylabel("Amount")
plt.show()

new_df=df.drop("Amount",axis="columns")
new_df

price=df.Amount
price

#Create linear regression object
reg=linear_model.LinearRegression()
reg.fit(new_df,price)

area=4500
# area=int(input("Enter the area of house: "))
# if(area>0):
print("The price of ",area,"/sqft house is ",reg.predict([[area]])[0])
# else:
#   print("Invalid input")

# Perform Naïve Bayes Classification for loan dataset using scikit-learn

import pandas as pd

df=pd.read_csv('loan_data.csv')
df.head()

df.info()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.xticks (rotation= 45, ha='right');

pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)
pre_df.head()
from sklearn.model_selection import train_test_split

X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train);
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred = model.predict(X_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

# Perform SVM Classification using iris dataset
from sklearn.datasets import load_iris
import pandas as pd

iris_sklearn = load_iris()
iris = pd.DataFrame(data=iris_sklearn['data'], columns=iris_sklearn['feature_names'])
iris['species'] = pd.Categorical.from_codes(iris_sklearn.target, iris_sklearn.target_names)
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(iris, hue='species', palette='Set1')
plt.show()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)
irisdata.head(10)
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']
print(X,y)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True)
X_train['sepal-length'][1]
print(X_train['sepal-length'][0], y_train[0])
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Implement Decision tree algorithm for iris dataset

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris)

x=iris.data[:,2:]
y=iris.target
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x_train,y_train)


y_pred=tree_clf.predict(x_test)
print(y_pred)


accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

plt.figure(figsize=(20,15))
plot_tree(tree_clf,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()


# Create own dataset and find Gini index for overall dataset

from sklearn.datasets import load_iris
import numpy as np
iris=load_iris()
y=iris.target

def gini_index(data):
    classes=np.unique(data)
    gini=0
    for cls in classes:
        p=np.sum(data == cls)/len(data)
        gini+=p**2
    gini=1-gini
    return gini
gini_dataset=gini_index(y)
print("Gini Index for the whole dataset:",gini_dataset)

X=iris.data
attributes=iris.feature_names
for i,attribute in enumerate(attributes):
    values=X[:,i]
    unique_values=np.unique(values)
    gini_attribute=0
    for val in unique_values:
        idx=np.where(values == val)[0]
        gini_val=gini_index(y[idx])
        weight=len(idx)/len(y)
        gini_attribute+=weight*gini_val
    print("Gini Index for attribute ",attribute,":",gini_attribute)

# Implement KNN classification algorithm for a given dataset using scikit-learn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris

iris=load_iris()
print(iris)

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=KNeighborsClassifier(n_neighbors=7)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Predicted values:")
print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
confusion_mat=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(confusion_mat)


# Implement Random forest algorithm with iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

headers=['sepal-length','sepal-width','pedal-length','pedal-width','class']
ds=pd.read_csv(path,names=headers)
print(ds.head())

x=ds.iloc[:,:-1].values
y=ds.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,random_state=16)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

result=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(result)
result1=classification_report(y_test,y_pred)
print("Classification Report:",)
print(result1)
result2=accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

plt.figure(figsize=(8,6))
sns.heatmap(result,annot=True,fmt='d',cmap="Reds",xticklabels=ds['class'].unique(),yticklabels=ds['class'].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Implement K-means clustering for a given dataset using scikit-learn

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

x=np.array([[1.713,1.518],[0.180,1.786],[0.353,1.240],[0.940,1.560],[1.486,0.759],[1.260,1.106],[1.540,0.419]])

y=np.array([0,1,1,0,0,1,1])

kmeans=KMeans(n_clusters=3,random_state=7).fit(x,y)
print("Centroide : ")
print(kmeans.cluster_centers_)

print("Predicted value ")

class1=kmeans.predict([[1.713,1.586]])
class2=kmeans.predict([[0.180,1.786]])
print(class1)
print(class2)

# Create a CNN model for image classification

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(8,8))
for i in range(25):
plt.subplot(5,5,i+1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(train_images[i])
plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy
(from_logits=True),metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images,
test_labels))
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print("Test Accuracy:",test_acc)

# Demonstrate data visualization Technique using Seaborn

import seaborn as sns
import matplotlib.pyplot as plt
data = sns.load_dataset("tips")
while True:
    print("\nEnter the option:")
    print("1. Line Chart")
    print("2. Histogram")
    print("3. Box Plot")
    print("4. Exit")
    option = input("Choose an option: ")
    if option == '4':
        print("Exiting the program.")
        break
    print("\nAvailable columns:")
    print(data.columns.tolist())
    column = input("Enter the column name you want to visualize: ")
    if column not in data.columns:
        print(f"Column '{column}' does not exist in the dataset.")
        continue
    if option == '1':
        sns.lineplot(data=data, x=data.index, y=column)
        plt.title(f"Line Chart of {column}")
        plt.grid()
        plt.show()
    elif option == '2':
        sns.histplot(data=data, x=column, bins=20, kde=True)
        plt.title(f"Histogram of {column}")
        plt.grid()
        plt.show()
    elif option == '3':
        sns.boxplot(x=data[column])
        plt.title(f"Box Plot of {column}")
        plt.grid()

        plt.show()
    else:
        print("Invalid option chosen. Please try again.")
