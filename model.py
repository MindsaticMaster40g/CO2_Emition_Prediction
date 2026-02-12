import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('CO2 Emissions_Canada.csv')

x = dataset.iloc[: ,:-1].values
y = dataset.iloc[: ,-1].values

features = [0 ,1 ,2 ,5 ,6]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), features)], remainder='passthrough', sparse_threshold=0)

x = np.array(ct.fit_transform(x))

x_train ,x_test ,y_train ,y_test = train_test_split(x , y ,test_size=0.2 ,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train ,y_train)

y_predict = regressor.predict(x_test)
success_rate = r2_score(y_test ,y_predict)
y_train_predict = regressor.predict(x_train)
train_rate = r2_score(y_train, y_train_predict)

print(success_rate)
print(train_rate)

sns.scatterplot(x=y_test ,y=y_predict ,alpha=0.6 ,color='blue')
plt.show()