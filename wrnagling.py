import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
dataset=load_diabetes()
print(dataset['feature_names'])
print(pd.DataFrame(data=np.c_[dataset['data'],dataset['target']],columns=dataset["feature_names"]+["target"]))
import matplotlib.pyplot as plt
for column in dataset:
    plt.boxplot(column)

