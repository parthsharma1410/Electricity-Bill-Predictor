#19BCE0761 | PARTH SHARMA

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
#%matplotlib inline
#loading the dataset
df = pd.read_csv('eBill.csv')

#Q1-Format the  date column with pandas to_datetime() funtion.
df["Date"] = pd.to_datetime(df["Date"]) 

#Q2-Create new data frame by splitting date column into month and year (Dataframe -> month, year, consumed_units,
#unit_price, total_price)

# Extracting all data Like Year Month Day Time etc
dataframe = df
dataframe["Month"] = pd.to_datetime(df["Date"]).dt.month
dataframe["Year"] = pd.to_datetime(df["Date"]).dt.year
dataframe["Date"] = pd.to_datetime(df["Date"]).dt.date
dataframe["Day"] = pd.to_datetime(df["Date"]).dt.day_name()
dataframe = df.set_index("Date")
dataframe.index = pd.to_datetime(dataframe.index)
dataframe.head(1)


#Q3-Visualize your monthly electric bill with matplotlib

#Plotting the monthly consumed units
from matplotlib import style

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

style.use('ggplot')

sns.lineplot(x=dataframe["Month"], y=dataframe["Consumed Units"], data=df)
sns.set(rc={'figure.figsize':(15,6)})

plt.title("Energy consumption per month")
plt.xlabel("Month")
plt.ylabel("Energy in units")
plt.grid(True)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)

plt.title("Monthly energy consumption")


#Q4-Predict the electric bill for 2021 first cycle using Regression.

X = data2[["Year",'Month',"Units Consumed",'Unit Price']]
y = data2['Total Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
cdf = pd.DataFrame(lm.coef_,X.columns,columns=["Coeff"])
cdf
predictions = lm.predict(X_test)
predictions
y_test
plt.scatter(y_test,predictions)
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
from sklearn.metrics import mean_squared_error
def plot_learning_curves(model,x,y):
 X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.4)
 val_errors = []

 for m in range(1,len(X_train)):
  model.fit(X_train[:m],y_train[:m])
  y_train_predict = model.predict(X_train[:m])
  y_val_predict = model.predict(X_val)
  val_errors.append(mean_squared_error(y_val_predict,y_val))
 plt.plot(np.sqrt(val_errors),"r-+",linewidth=2,label='validation')
 plt.legend()
lin_reg = LinearRegression()
plot_learning_curves(lin_reg,X,y)