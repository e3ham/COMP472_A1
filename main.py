import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Function to write performance metrics to a file
def write_performance_metrics(classifier, X_test, y_true, y_pred, file_name):
    report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(file_name, 'a') as f:
        if hasattr(classifier, 'best_params_'):
            f.write(f"Best Parameters: {classifier.best_params_}\n")
        f.write("(B) Confusion Matrix:\n")
        f.write(f"{metrics.confusion_matrix(y_true, y_pred)}\n")
        f.write("(C) Classification Report:\n")
        f.write(f"{metrics.classification_report(y_true, y_pred, zero_division=0)}\n")
        f.write(f"(D) Accuracy: {metrics.accuracy_score(y_true, y_pred)}\n")
        f.write(f"Macro Average F1 Score: {report['macro avg']['f1-score']}\n")
        f.write(f"Weighted Average F1 Score: {report['weighted avg']['f1-score']}\n\n")


# Function for repeated evaluation
def repeated_evaluation(classifier, X, y, file_name, n_runs=5):
    accuracy_scores = []
    f1_macro_scores = []
    f1_weighted_scores = []

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        f1_macro_scores.append(metrics.f1_score(y_test, y_pred, average='macro'))
        f1_weighted_scores.append(metrics.f1_score(y_test, y_pred, average='weighted'))

    # Write the average and variance of the scores to the file
    with open(file_name, 'a') as f:
        f.write(f"------------------------------------------\n")
        f.write(f"Repeated Evaluation for {type(classifier).__name__}\n")
        f.write(f"Average Accuracy: {np.mean(accuracy_scores)}\n")
        f.write(f"Variance in Accuracy: {np.var(accuracy_scores)}\n")
        f.write(f"Average Macro F1 Score: {np.mean(f1_macro_scores)}\n")
        f.write(f"Variance in Macro F1 Score: {np.var(f1_macro_scores)}\n")
        f.write(f"Average Weighted F1 Score: {np.mean(f1_weighted_scores)}\n")
        f.write(f"Variance in Weighted F1 Score: {np.var(f1_weighted_scores)}\n\n")


# Reading the CSV files
abalone = pd.read_csv("abalone.csv")
penguins = pd.read_csv("penguins.csv")
penguinCategories = ['island', 'sex']

# Initializing the OHE for the penguins dataset
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(penguins[penguinCategories])

# Creating a dataframe for the data numeration
df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(penguinCategories))

# manual conversion of categories assuming the dataframe is created beforehand
##df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})
##df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})

# Replacing the old column and replacing it with our new dataframe
penguins = penguins.drop(columns=penguinCategories)
penguins = pd.concat([penguins, df], axis=1)
# print(penguins)

# without using OHE:
# penguins['island'] = penguins['island'].astype('category')
# penguins['sex'] = penguins['sex'].astype('category')
# penguins['island'] = penguins['island'].cat.codes
# penguins['sex'] = penguins['sex'].cat.codes

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
percentagesAbalone.plot(kind="bar", color='green')
plt.title("Percentages of instances for each sex of abalone")
plt.xlabel("Type")
plt.ylabel("Percentage")
plt.savefig("abalone-classes.png")

# Data splitting for both datasets
X_penguins = penguins.drop(columns=["species"])
Y_penguins = penguins["species"]
X_train_penguins, X_test_penguins, Y_train_penguins, Y_test_penguins = train_test_split(X_penguins, Y_penguins)
W_abalone = abalone.drop(columns=["Type"])
Z_abalone = abalone["Type"]
W_train_abalone, W_test_abalone, Z_train_abalone, Z_test_abalone = train_test_split(W_abalone, Z_abalone)

# Base-DT for Penguins
baseDT_penguins = DecisionTreeClassifier()
baseDT_penguins.fit(X_train_penguins, Y_train_penguins)
Y_pred_penguins = baseDT_penguins.predict(X_test_penguins)
p_dt_report = metrics.classification_report(Y_test_penguins, Y_pred_penguins, output_dict=True, zero_division=0)

# write to file
with open("penguin-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Base-DT with default parameters\n")
    f.write("(B) Confusion Matrix:\n")
    f.write(f"{metrics.confusion_matrix(Y_test_penguins, Y_pred_penguins)}\n")
    f.write("(C) Classification Report:\n")
    f.write(f"{metrics.classification_report(Y_test_penguins, Y_pred_penguins, zero_division=0)}\n")
    f.write(f"(D) Accuracy: {metrics.accuracy_score(Y_test_penguins, Y_pred_penguins)}\n")
    f.write(f"Macro Average F1 Score: {p_dt_report['macro avg']['f1-score']}\n")
    f.write(f"Weighted Average F1 Score: {p_dt_report['weighted avg']['f1-score']}\n\n")

# Base-DT for Abalone
baseDT_abalone = DecisionTreeClassifier()
baseDT_abalone.fit(W_train_abalone, Z_train_abalone)
Z_pred_abalone = baseDT_abalone.predict(W_test_abalone)
a_dt_report = metrics.classification_report(Z_test_abalone, Z_pred_abalone, output_dict=True, zero_division=0)

# write to file
with open("abalone-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Base-DT with default parameters\n")
    f.write("(B) Confusion Matrix:\n")
    f.write(f"{metrics.confusion_matrix(Z_test_abalone, Z_pred_abalone)}\n")
    f.write("(C) Classification Report:\n")
    f.write(f"{metrics.classification_report(Z_test_abalone, Z_pred_abalone, zero_division=0)}\n")
    f.write(f"(D) Accuracy: {metrics.accuracy_score(Z_test_abalone, Z_pred_abalone)}\n")
    f.write(f"Macro Average F1 Score: {a_dt_report['macro avg']['f1-score']}\n")
    f.write(f"Weighted Average F1 Score: {a_dt_report['weighted avg']['f1-score']}\n\n")

# Evaluate with write_performance_metrics and repeated_evaluation
print(f"Base-DT Parameters for Penguins: {baseDT_penguins.get_params()}")
print(f"Base-DT Accuracy for Penguins: {metrics.accuracy_score(Y_test_penguins, Y_pred_penguins)}")

# After training and evaluation for Base-DT (Abalone)
print(f"Base-DT Parameters for Abalone: {baseDT_abalone.get_params()}")
print(f"Base-DT Accuracy for Abalone: {metrics.accuracy_score(Z_test_abalone, Z_pred_abalone)}")

# Decision tree visualization for Penguins
plt.figure(figsize=(20, 10))
plot_tree(baseDT_penguins, filled=True, feature_names=X_train_penguins.columns,
          class_names=['Adelie', 'Chinstrap', 'Gentoo'])
plt.savefig('penguins_BaseDT.png')

# Decision tree visualization for Abalone (with limited depth for visualization purposes)
plt.figure(figsize=(20, 10))
plot_tree(baseDT_abalone, filled=True, feature_names=W_train_abalone.columns, class_names=['Male', 'Female', 'Infant'],
          max_depth=3)
plt.savefig('abalone_BaseDT.png')

# Define the hyperparameters grid for Top-DT
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],  # None means unlimited
    'min_samples_split': [2, 4, 6]
}

# Perform grid search for Top-DT (Penguins)
grid_search_penguins_dt = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5)
grid_search_penguins_dt.fit(X_train_penguins, Y_train_penguins)

# Perform grid search for Top-DT (Abalone)
grid_search_abalone_dt = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5)
grid_search_abalone_dt.fit(W_train_abalone, Z_train_abalone)

# Train and evaluate Top-DT with best parameters found (Penguins)
topDT_penguins = grid_search_penguins_dt.best_estimator_
Y_pred_topDT_penguins = topDT_penguins.predict(X_test_penguins)

# write to file
with open("penguin-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Top-DT with gridsearch\n")
    f.write("Parameters: criterion: [gini, entropy], max depth: [none, 10, 20], minimum sample split: [2, 4, 6]\n")
    f.write(f"Best parameters: {grid_search_penguins_dt.best_params_}")
write_performance_metrics(topDT_penguins, X_test_penguins, Y_test_penguins, Y_pred_topDT_penguins,
                          'penguin-performance.txt')
repeated_evaluation(topDT_penguins, X_penguins, Y_penguins, 'penguin-performance.txt')

# Train and evaluate Top-DT with best parameters found (Abalone)
topDT_abalone = grid_search_abalone_dt.best_estimator_
Y_pred_topDT_abalone = topDT_abalone.predict(W_test_abalone)

# write to file
with open("abalone-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Top-DT with gridsearch\n")
    f.write("Parameters: criterion: [gini, entropy], max depth: [none, 10, 20], minimum sample split: [2, 4, 6]\n")
    f.write(f"Best parameters: {grid_search_abalone_dt.best_params_}")
write_performance_metrics(topDT_abalone, W_test_abalone, Z_test_abalone, Y_pred_topDT_abalone,
                          'abalone-performance.txt')
repeated_evaluation(topDT_abalone, W_abalone, Z_abalone, 'abalone-performance.txt')

# Decision tree visualization for Penguins
plt.figure(figsize=(20, 10))
plot_tree(topDT_penguins, filled=True, feature_names=X_train_penguins.columns,
          class_names=['Adelie', 'Chinstrap', 'Gentoo'])
plt.savefig('penguins_TopDT.png')

# Decision tree visualization for Abalone (with limited depth for visualization purposes)
plt.figure(figsize=(20, 10))
plot_tree(topDT_abalone, filled=True, feature_names=W_train_abalone.columns, class_names=['Male', 'Female', 'Infant'],
          max_depth=3)
plt.savefig('abalone_TopDT.png')

print(f"Top-DT Best Parameters for Penguins: {grid_search_penguins_dt.best_params_}")
print(f"Top-DT Accuracy for Penguins: {metrics.accuracy_score(Y_test_penguins, Y_pred_topDT_penguins)}")

print(f"Top-DT Best Parameters for Abalone: {grid_search_abalone_dt.best_params_}")
print(f"Top-DT Accuracy for Abalone: {metrics.accuracy_score(Z_test_abalone, Y_pred_topDT_abalone)}")

# Base-MLP for Penguins
base_mlp_penguins = MLPClassifier(hidden_layer_sizes=(100, 100),
                                  activation='logistic',
                                  solver='sgd')
base_mlp_penguins.fit(X_train_penguins, Y_train_penguins)
Y_pred_mlp_penguins = base_mlp_penguins.predict(X_test_penguins)
print(f"Base-MLP Parameters for Penguins: {base_mlp_penguins.get_params()}")
print(f"Base-MLP Accuracy for Penguins: {metrics.accuracy_score(Y_test_penguins, Y_pred_mlp_penguins)}")
p_mlp_report = metrics.classification_report(Y_test_penguins, Y_pred_mlp_penguins, output_dict=True, zero_division=0)

# write to file
with open("penguin-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Base-MLP with default parameters\n")
    f.write("(B) Confusion Matrix:\n")
    f.write(f"{metrics.confusion_matrix(Y_test_penguins, Y_pred_mlp_penguins)}\n")
    f.write("(C) Classification Report:\n")
    f.write(f"{metrics.classification_report(Y_test_penguins, Y_pred_mlp_penguins, zero_division=0)}\n")
    f.write(f"(D) Accuracy: {metrics.accuracy_score(Y_test_penguins, Y_pred_mlp_penguins)}\n")
    f.write(f"Macro Average F1 Score: {p_mlp_report['macro avg']['f1-score']}\n")
    f.write(f"Weighted Average F1 Score: {p_mlp_report['weighted avg']['f1-score']}\n\n")

# Base-MLP for Abalone
base_mlp_abalone = MLPClassifier(hidden_layer_sizes=(100, 100),
                                 activation='logistic',
                                 solver='sgd')
base_mlp_abalone.fit(W_train_abalone, Z_train_abalone)
Z_pred_mlp_abalone = base_mlp_abalone.predict(W_test_abalone)
a_mlp_report = metrics.classification_report(Z_test_abalone, Z_pred_mlp_abalone, output_dict=True, zero_division=0)

# write to file
with open("abalone-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Base-MLP with default parameters\n")
    f.write("(B) Confusion Matrix:\n")
    f.write(f"{metrics.confusion_matrix(Z_test_abalone, Z_pred_mlp_abalone)}\n")
    f.write("(C) Classification Report:\n")
    f.write(f"{metrics.classification_report(Z_test_abalone, Z_pred_mlp_abalone, zero_division=0)}\n")
    f.write(f"(D) Accuracy: {metrics.accuracy_score(Z_test_abalone, Z_pred_mlp_abalone)}\n")
    f.write(f"Macro Average F1 Score: {a_mlp_report['macro avg']['f1-score']}\n")
    f.write(f"Weighted Average F1 Score: {a_mlp_report['weighted avg']['f1-score']}\n\n")
print(f"Base-MLP Parameters for Abalone: {base_mlp_abalone.get_params()}")
print(f"Base-MLP Accuracy for Abalone: {metrics.accuracy_score(Z_test_abalone, Z_pred_mlp_abalone)}")

# Define the hyperparameters grid for Top-MLP
mlp_param_grid = {
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'activation': ['logistic', 'tanh', 'relu'],
    # In scikit-learn, the 'logistic' activation function is equivalent to the sigmoid function.
    'solver': ['adam', 'sgd'],
}

# Perform grid search for Top-MLP (Penguins)
grid_search_penguins_mlp = GridSearchCV(MLPClassifier(), mlp_param_grid, cv=5)
grid_search_penguins_mlp.fit(X_train_penguins, Y_train_penguins)

# Perform grid search for Top-MLP (Abalone)
grid_search_abalone_mlp = GridSearchCV(MLPClassifier(), mlp_param_grid, cv=5)
grid_search_abalone_mlp.fit(W_train_abalone, Z_train_abalone)

# Train and evaluate Top-MLP with the best parameters found (Penguins)
topMLP_penguins = grid_search_penguins_mlp.best_estimator_
Y_pred_topMLP_penguins = topMLP_penguins.predict(X_test_penguins)

# write to file
with open("penguin-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Top-MLP with gridsearch\n")
    f.write(
        "Parameters: activation function: [sigmoid, tanh, relu], hidden layers: [(30, 50), (10, 10, 10)], solver: [adam, sgd]\n")
    f.write(f"Best parameters: {grid_search_penguins_mlp.best_params_}")
write_performance_metrics(topMLP_penguins, X_test_penguins, Y_test_penguins, Y_pred_topMLP_penguins,
                          'penguin-performance.txt')
repeated_evaluation(topMLP_penguins, X_penguins, Y_penguins, 'penguin-performance.txt')

# Train and evaluate Top-MLP with the best parameters found (Abalone)
topMLP_abalone = grid_search_abalone_mlp.best_estimator_
Y_pred_topMLP_abalone = topMLP_abalone.predict(W_test_abalone)

with open("abalone-performance.txt", 'a') as f:
    f.write(f"------------------------------------------\n")
    f.write("(A) Top-MLP with gridsearch\n")
    f.write(
        "Parameters: activation function: [sigmoid, tanh, relu], hidden layers: [(30, 50), (10, 10, 10)], solver: [adam, sgd]\n")
    f.write(f"Best parameters: {grid_search_abalone_mlp.best_params_}")
write_performance_metrics(topMLP_abalone, W_test_abalone, Z_test_abalone, Y_pred_topMLP_abalone,
                          'abalone-performance.txt')
repeated_evaluation(topMLP_abalone, W_abalone, Z_abalone, 'abalone-performance.txt')

print(f"Top-MLP Best Parameters for Penguins: {grid_search_penguins_mlp.best_params_}")
print(f"Top-MLP Accuracy for Penguins: {metrics.accuracy_score(Y_test_penguins, Y_pred_topMLP_penguins)}")

print(f"Top-MLP Best Parameters for Abalone: {grid_search_abalone_mlp.best_params_}")
print(f"Top-MLP Accuracy for Abalone: {metrics.accuracy_score(Z_test_abalone, Y_pred_topMLP_abalone)}")