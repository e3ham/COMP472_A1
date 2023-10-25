import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("penguins.csv")
categories = ['island', 'sex']

ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(data[categories])
df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categories))

data = data.drop(columns=categories)

data = pd.concat([data, df], axis=1)

print(data)


# without using SKLearn:
# data = pd.read_csv("penguins.csv")
# data = pd.get_dummies(data, columns=['island', 'sex'])
# print(data)
