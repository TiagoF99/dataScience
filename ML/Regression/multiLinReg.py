import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

df = pd.read_csv("FuelConsumption.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
print(cdf.shape)

# Lets plot Emission values with respect to Engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# splits the data into train 80% and test 20%
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print(train.shape)


# predict model using x parameters
regr = linear_model.LinearRegression()
x = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = train['CO2EMISSIONS']
regr.fit(x, y)
# The coefficients
print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = test['CO2EMISSIONS']
print("Residual sum of squares: " + str(np.mean((y_hat - y) ** 2)))

# Explained variance score: 1 is perfect prediction
# R^2 value is approx .88 so very good prediction
print('Variance score: %.2f' % regr.score(x, y))

# ===========================================================================================
# predict using a different set of parameters
regr = linear_model.LinearRegression()
x = train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
y = train['CO2EMISSIONS']
regr.fit(x, y)
# The coefficients
print('Coefficients: ', regr.coef_)

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
y = test['CO2EMISSIONS']
print("Residual sum of squares: " + str(np.mean((y_hat - y) ** 2)))

# Explained variance score: 1 is perfect prediction
# R^2 value also approx .88 so very good prediction.
# notice that no real difference in change of parameters.
print('Variance score: ' + str(regr.score(x, y)))