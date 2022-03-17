import pandas as pd
import numpy as np
df=pd.read_csv("SalaryGender.csv")
print(df.info())
print(df.shape)
print(df.describe)
print(df.columns)
x=df.drop(["Salary"],axis=1)
y=df.iloc[:,df.columns=="Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100,test_size=0.1)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
a=accuracy_score(y_train,y_pred)
print(a)

