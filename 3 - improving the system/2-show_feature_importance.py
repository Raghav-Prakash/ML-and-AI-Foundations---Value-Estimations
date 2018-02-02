import numpy as np
from sklearn.externals import joblib

# create a numPy array of the features from the input data set
feature_labels = np.array(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms', 'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft', 'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating', 'has_central_cooling', 'garage_type_attached', 'garage_type_detached', 'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad', 'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown', 'city_East Amychester', 'city_East Janiceville', 'city_East Justin', 'city_East Lucas', 'city_Fosterberg', 'city_Hallfort', 'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt', 'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn', 'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack', 'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven', 'city_Martinezfort', 'city_Morrisport', 'city_New Michele', 'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown', 'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough', 'city_Richardport', 'city_Rickytown', 'city_Scottberg', 'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire', 'city_Wendybury', 'city_West Ann', 'city_West Brittanyview', 'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia', 'city_West Terrence'])

# load the model created
model = joblib.load("trained_house_classifier_model.pkl")

'''
Not all features (columns from the input table) contribute to predicting accurately the sale price of the house.
So we calcuate the importance of each feature (value from 0 to 1) where each value is like a percentage of its
importance in our model.
'''
# create a numpy array of feature importances
importance = model.feature_importances_

# sorting this array which returns the array indices of these increasing values.
feature_indexes_by_importance = importance.argsort()

# print the feature along with its importance 
for index in feature_indexes_by_importance:
	print("{} - {:.2f}%".format(feature_labels[index], (importance[index])*100))

'''
Importance output:
city_Rickytown - 0.00%
city_New Robinton - 0.00%
city_New Michele - 0.00%
city_Martinezfort - 0.00%
city_Davidtown - 0.00%
city_Julieberg - 0.00%
city_West Brittanyview - 0.07%
city_Leahview - 0.10%
city_Fosterberg - 0.11%
city_Lake Jennifer - 0.13%
city_Amystad - 0.13%
city_Port Daniel - 0.14%
city_East Justin - 0.14%
city_Toddshire - 0.14%
city_South Stevenfurt - 0.17%
city_Brownport - 0.20%
city_West Terrence - 0.20%
city_Clarkberg - 0.20%
city_Jenniferberg - 0.21%
city_West Gerald - 0.21%
city_West Lydia - 0.22%
city_East Janiceville - 0.23%
city_Lake Carolyn - 0.24%
city_Port Adamtown - 0.26%
city_Joshuafurt - 0.27%
city_Wendybury - 0.28%
city_East Amychester - 0.28%
city_East Lucas - 0.29%
city_Lake Christinaport - 0.31%
city_Lake Dariusborough - 0.31%
city_Hallfort - 0.33%
city_Davidfort - 0.34%
city_Port Jonathanborough - 0.35%
city_Morrisport - 0.36%
city_West Gregoryview - 0.36%
city_Scottberg - 0.38%
city_Richardport - 0.39%
city_Justinport - 0.40%
city_Jeffreyhaven - 0.54%
city_North Erinville - 0.58%
has_central_cooling - 0.65%
city_Lewishaven - 0.66%
has_central_heating - 0.68%
city_Port Andrealand - 0.75%
city_South Anthony - 0.77%
city_Chadstad - 0.81%
city_Coletown - 0.84%
city_West Ann - 0.89%
garage_type_detached - 1.06%
garage_type_none - 1.26%
garage_type_attached - 1.30%
city_Lake Jack - 1.36%
has_pool - 1.71%
half_bathrooms - 1.89%
has_fireplace - 1.91%
stories - 2.11%
full_bathrooms - 3.90%
carport_sqft - 3.99%
num_bedrooms - 4.56%
year_built - 12.50%
garage_sqft - 13.08%
livable_sqft - 17.20%
total_sqft - 17.28%
'''