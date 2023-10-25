import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reading the CSV files
abalone = pd.read_csv("abalone.csv")
penguins = pd.read_csv("penguins.csv")
penguinCategories = ['island', 'sex']

# Initializing the OHE for the penguins dataset
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(penguins[penguinCategories])

# Creating a dataframe for the data numeration
df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(penguinCategories))

# Replacing the old column and replacing it with our new dataframe
penguins = penguins.drop(columns=penguinCategories)
penguins = pd.concat([penguins, df], axis=1)
# print(penguins)

# without using SKLearn (only pandas):
# penguins = pd.read_csv("penguins.csv")
# penguins = pd.get_dummies(penguins, columns=['island', 'sex'])
# print(penguins)

# Getting the percentages needed for plotting graphs
percentagesPenguins = penguins["species"].value_counts(normalize=True) * 100
percentagesAbalone = abalone["Type"].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 6))
percentagesPenguins.plot(kind="bar", color='blue')
plt.title("Percentages of instances for each species")
plt.xlabel("Species")
plt.ylabel("Percentage")
plt.savefig("penguin-classes.png")

plt.figure(figsize=(8, 6))
percentagesPenguins.plot(kind="bar", color='green')
plt.title("Percentages of instances for each type of abalone")
plt.xlabel("Type")
plt.ylabel("Percentage")
plt.savefig("abalone-classes.png")

# Splitting the dataset in order to test them on our models
X = penguins.drop(columns=["species"])
Y = penguins["species"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("Y_train shape:", Y_train.shape)
# print("Y_test shape:", Y_test.shape)

W = abalone.drop(columns=["Type"])
Z = abalone["Type"]
W_train, W_test, Z_train, Z_test = train_test_split(W, Z, random_state=42)
# print("W_train shape:", W_train.shape)
# print("W_test shape:", W_test.shape)
# print("Z_train shape:", Z_train.shape)
# print("X_test shape:", Z_test.shape)

# Testing on the base decision tree and plotting it graphically (max depth for abalone)
baseDT_penguins = DecisionTreeClassifier()
baseDT_penguins.fit(X_train, Y_train)
Y_pred = baseDT_penguins.predict(X_test)
plt.figure(figsize=(12, 8))
plot_tree(baseDT_penguins, filled=True, feature_names=X.columns, class_names=baseDT_penguins.classes_, rounded=True)
plt.title("Penguins Decision Tree Visualization")
plt.show()
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

baseDT_abalone = DecisionTreeClassifier(max_depth=5)
baseDT_abalone.fit(W_train, Z_train)
Z_pred = baseDT_abalone.predict(W_test)
plt.figure(figsize=(12, 8))
plot_tree(baseDT_abalone, filled=True, feature_names=X.columns, class_names=baseDT_abalone.classes_, rounded=True)
plt.title("Abalone Decision Tree Visualization")
plt.show()
print("Accuracy:", metrics.accuracy_score(Z_test, Z_pred))
