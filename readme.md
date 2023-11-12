This assignment encompasses the development and comparison of machine learning models for classification tasks using two distinct datasets: the Abalone dataset and the Penguin dataset. The primary goal is to predict the species of penguins and the sex of abalones based on various input features.

The assignment implements and evaluates two types of classifiers: Decision Trees (Base-DT and Top-DT) and Multi-Layer Perceptrons (Base-MLP and Top-MLP). Base-DT and Base-MLP serve as baseline models with default parameters, while Top-DT and Top-MLP are optimized versions using GridSearchCV for hyperparameter tuning.

We perform data preprocessing, including one-hot encoding for categorical variables, and split the data into training and testing sets. Performance metrics such as accuracy, confusion matrix, F1 scores, and classification reports are calculated and written to separate performance files for in-depth analysis.

For visualization, we plot the class distributions and decision trees for both datasets. Additionally, we conduct repeated evaluations to assess the consistency of our models.

Key findings include the best parameters for the top-performing models obtained through grid search and the average performance metrics over multiple runs, providing insights into the models' reliability and predictive power.

For a detailed understanding, all code, figures, and performance metric files are available within the assignment repository.
