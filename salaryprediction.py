
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
a=pd.read_csv("Salary_Data.csv")
print(a.shape)
print(a.head)
income=set(a["Salary"])
a["Salary"]=a["Salary"].map({'<=50000':0,'>50000':1}).astype(float)
x=a.iloc[:,:-1]
y=a.iloc[:-1]
z=y.columns[x.isna().any()]
print(z)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
res=StandardScaler()
x_train=res.fit_transform(x_train)
x_test=res.transform(x_test)
for i in range(1,40):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(y_pred)
    acc=accuracy_score(y_test,y_pred)
    print(acc)
