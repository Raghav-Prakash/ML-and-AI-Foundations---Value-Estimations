# read data from a csv file and display it on a web browser

import pandas as pd
import webbrowser
import os

# read data from the csv file
house_data = pd.read_csv("ml_house_data_set.csv")

# take the first 100 houses and convert that data into html format
html_data = house_data[0:100].to_html()

# write this into an html file
with open("data.html","w") as f:
	f.write(html_data)

# get the full filename of this html file and display on a browser
full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))
