import numpy as np
import pandas as pd
import sklearn
a=pd.read_csv("titanticsurival.csv")
a.shape
a.head()
income_set=set(a["sex"])
a["sex"]=a["sex"].map({"female":0 ,:"male":1}).astype(int)
x=a.drop("survived")
y=a["survived"]
print(x.columns[x.isna().any()])
x.age=x.age.fillna(age.mean())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
print(acc)

