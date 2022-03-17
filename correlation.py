import pandas as pd
d=pd.read_csv("middle_tn_school.csv")
print(d.describe())
d[['reduced_lunch','school_rating']].groupby(['school_rating']).describe().unstack()
print(d[['reduced_lunch','school_rating']].corr())