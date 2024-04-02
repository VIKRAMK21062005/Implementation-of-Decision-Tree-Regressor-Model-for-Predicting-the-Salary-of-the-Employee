# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean squared error, and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIKRAM K
RegisterNumber:  212222040180
*/

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)

r2=metrics.r2_score(y_test,y_pred)
print(r2)

dt.predict([[5,6]])

```

## Output:

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/bc9d2710-a7b0-4541-9765-4c897b694f26)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/c76d63e7-34d1-49e6-b4b9-65c0ff42b8b0)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/935ac006-b750-4901-805f-e705f914eb16)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/c7d85fe4-cac4-4e7f-9ad6-b2ce1c4fbd87)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/7caf5f5b-70e8-4b68-9a36-57b892de5386)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/f9975fc6-46d5-434c-a316-92818cdafc85)

![image](https://github.com/VIKRAMK21062005/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120624033/490435a5-9817-4b02-91c9-ba020f576769)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
