from sklearn.externals import joblib

# Load the model we trained previously
model = joblib.load('trained_house_classifier_model.pkl')

# Input features for a new house to predict its sale price
house_to_value = [
    # House features
    2006,   # year_built
    1,      # stories
    4,      # num_bedrooms
    3,      # full_bathrooms
    0,      # half_bathrooms 
    2200,   # livable_sqft
    2350,   # total_sqft
    0,      # garage_sqft
    0,      # carport_sqft
    True,   # has_fireplace
    False,  # has_pool
    True,   # has_central_heating
    True,   # has_central_cooling
    
    # Garage type: Choose only one
    0,      # attached
    0,      # detached
    1,      # none
    
    # City: Choose only one
    0,      # Amystad
    1,      # Brownport
    0,      # Chadstad
    0,      # Clarkberg
    0,      # Coletown
    0,      # Davidfort
    0,      # Davidtown
    0,      # East Amychester
    0,      # East Janiceville
    0,      # East Justin
    0,      # East Lucas
    0,      # Fosterberg
    0,      # Hallfort
    0,      # Jeffreyhaven
    0,      # Jenniferberg
    0,      # Joshuafurt
    0,      # Julieberg
    0,      # Justinport
    0,      # Lake Carolyn
    0,      # Lake Christinaport
    0,      # Lake Dariusborough
    0,      # Lake Jack
    0,      # Lake Jennifer
    0,      # Leahview
    0,      # Lewishaven
    0,      # Martinezfort
    0,      # Morrisport
    0,      # New Michele
    0,      # New Robinton
    0,      # North Erinville
    0,      # Port Adamtown
    0,      # Port Andrealand
    0,      # Port Daniel
    0,      # Port Jonathanborough
    0,      # Richardport
    0,      # Rickytown
    0,      # Scottberg
    0,      # South Anthony
    0,      # South Stevenfurt
    0,      # Toddshire
    0,      # Wendybury
    0,      # West Ann
    0,      # West Brittanyview
    0,      # West Gerald
    0,      # West Gregoryview
    0,      # West Lydia
    0       # West Terrence
]

'''
we can't predict using the model directly on the input above. 
The scikit-learn library can predict the model when given an array of lists(houses).
'''
# create a list with one house
houses_to_predict = [house_to_value]

# using the model and the input list, predict the price
predicted_house_prices = model.predict(houses_to_predict)

# since the function above returned a list with one value, we extract that value for simplification
predicted_house_price = predicted_house_prices[0]

# display the predicted house price
print("The predicted sale price of this house: ${:,.2f}".format(predicted_house_price))

'''
Output:
The predicted sale price of this house: $590,868.51
''' 
