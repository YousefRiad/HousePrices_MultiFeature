import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math


#calling and storing dataframe in variable df
df = pd.read_csv('house_prices.csv')

#we calculate and floor the median for column 'bedrooms' because we have a missing value(in row #2):
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)

#we fill any null or NaN spot in the bedroom column with the median_bedroom value and update the dataframe:
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

#plotting the data (we have 3 features so this step with plotting 1 feature isn't necessary)
%matplotlib inline
plt.xlabel('area(sqr ft)')
plt.ylabel('price($)')
plt.scatter(df.area,df.price,color='red', marker='x')

#usning linear reg (we can see from the dataframe its a linear data)
rg = linear_model.LinearRegression()
rg.fit(df[['area','bedrooms','age']],df.price)

#w=
rg.coef_

#b=
rg.intercept_

#testing with prediction 1:
rg.predict([[3000,3,40]])

#testing with prediction 2:
rg.predict([[2500,4,5]])
