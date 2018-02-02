import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV


house_data = pd.read_csv("ml_house_data_set.csv")

# removing features (fields/columns) from the dataset that we don't need to build the model
del house_data['house_number']
del house_data['street_name']
del house_data['unit_number']
del house_data['zip_code']

# applying one-hot encoding to certain features in the dataset

features_df = pd.get_dummies(house_data, columns=['garage_type','city'])

# remove the column whose values need to be predicted using the model from the features data frame
del features_df['sale_price']

# get the features as a numPy matrix X and the predicted value 'sale_price' as a numpy matrix y

X = features_df.as_matrix()
y = house_data['sale_price'].as_matrix()

# split the above X,y data into training and testing data (70% : training and 30%: testing)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# build the model which uses the gradient-boosted regression algorithm
model = ensemble.GradientBoostingRegressor()

'''
Instead of putting in one value for each hyper-parameter, we try a list of values for each parameter
and, using grid search (a brute-force approach), we see which values for each hyper-parameter gives the best
possible prediction (minimizes the mean absolute error for both training and testing data) to give a good fit 
(and avoid overfitting and under-fitting).
'''
param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}

# apply Grid-Search cross-validation onto our model using these parameters and 1 CPU (we could use more than 1 CPU)
gs_cv = GridSearchCV(model, param_grid, n_jobs=1)

# fit only the training data for grid search 
'''
the grid search slices the training data to train the model using a subset of the training data and test the model
using the other data subsets.
'''
gs_cv.fit(X_train, y_train)

# print the best parameters computed by the grid search 
print("The best parameters: " + gs_cv.best_params_)

# compute and print the error rate on the training data when our model used the best parameters
error_rate = mean_absolute_error(y_train, gs_cv.predict(X_train))
print("Mean absolute error on the training data with the best parameters: %.4f" % error_rate)

# compute the error rate on the testing data when our model used the best parameters
error_rate = mean_absolute_error(y_test, gs_cv.predict(X_test))
print("Mean absolute error on the testing data with the best parameters: %.4f" % error_rate)
