import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

house_data = pd.read_csv("ml_house_data_set.csv")

# removing features (fields/columns) from the dataset that we don't need to build the model
del house_data['house_number']
del house_data['street_name']
del house_data['unit_number']
del house_data['zip_code']

# applying one-hot encoding to certain features in the dataset

'''
one-hot encoding is done using the pandas library's get_dummies function with parameters:
data frame and the columns in the data frame to apply one-hot encoding
'''
features_df = pd.get_dummies(house_data, columns=['garage_type','city'])

# remove the column whose values need to be predicted using the model from the features data frame
del features_df['sale_price']

# get the features as a numPy matrix X and the predicted value 'sale_price' as a numpy matrix y

X = features_df.as_matrix()
y = house_data['sale_price'].as_matrix()

# split the above X,y data into training and testing data (70% : training and 30%: testing)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# building a model using gradient boosting regression algorithm
'''
the hyper-parametes are:
n_estimators: number of decision trees
learning_rate: controls how each additional decision tree influences the overall prediction
max_depth: max depth of each decision tree
min_samples_leaf: the number of times a value appears in the training data before the decision tree
				  makes a decision based on it
max_features: the percentage of features randomly chosen to create a branch in the decision tree
loss: the 'huber' function in the scikit library handles errors
'''
model = ensemble.GradientBoostingRegressor(
	n_estimators = 1000, 
	learning_rate=0.1, 
	max_depth=6, 
	min_samples_leaf=9, 
	max_features=0.1,
	loss='huber'
	)

# fit the training data into our model
model.fit(X_train, y_train)

# save the model for future analyses
joblib.dump(model, 'trained_house_classifier_model.pkl')

'''
compare the values of X and y for both training and testing data and see how close we are 
in our prediction using the mean absolute error. 

So we have, for training and testing data, the y values which are known and we predict using the model 
what y values would the X values generate and see how close the predicted and actual values are using our model. 
'''

# find the error rate on the training data
error_rate = mean_absolute_error(y_train, model.predict(X_train))
print("Mean absolute error on the training data: %.4f" % error_rate)

# find the error rate on the testing data
error_rate = mean_absolute_error(y_test, model.predict(X_test))
print("Mean absolute error on the testing data: %.4f" % error_rate)

'''
Output:
Mean absolute error on the training data: 47678.6038
Mean absolute error on the testing data: 60314.8504

In the training data, the model predicted the sale price of each house to within $47,678 of the real price.
In the testing data, the price predicted was $60,314 of the real price, which is pretty far off.
'''
