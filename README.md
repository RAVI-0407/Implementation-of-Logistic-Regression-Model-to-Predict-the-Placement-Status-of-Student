# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAVIPRASATH K
RegisterNumber:  212224230225
*/
```
```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
data=pd.read_csv("drive/MyDrive/ML/Placement_Data.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/4558da90-fbd1-417b-8f79-9a782c1d9dfa)
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/user-attachments/assets/c4f2c7e9-bc05-4700-aebf-e0b602dc2abd)
```
data1.isnull()
```
![image](https://github.com/user-attachments/assets/15e8f86d-d546-498a-94e6-5520cd48cece)
```
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![image](https://github.com/user-attachments/assets/3ef10981-7699-4f30-8436-0f9e5fec37e8)
```
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/65112a53-8b8b-4240-86b4-4be5f918dab8)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/36da5b32-52fa-4b0e-a80d-23288efcc44b)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/f24f0e64-f994-469c-9895-d4a724b8aab2)
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
![image](https://github.com/user-attachments/assets/f1bbfe8c-7f53-4d08-b4dd-a1e61b661313)
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/4d542a7d-873b-49e3-80b0-0f1944ebe922)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
