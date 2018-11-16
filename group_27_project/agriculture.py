import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import random
import csv
import pickle
# import plotly.graph_objs as go

df1 = pd.read_csv("agriculture.csv",low_memory=False)

y=df1.iloc[7][2:]


# print(type(y))

x1=df1.iloc[27][2:]
x2=df1.iloc[28][2:]
x3=df1.iloc[29][2:]

x11=pd.concat([x1, x2 ], axis=1)
x11=pd.concat([x11, x3], axis=1)
# print(x11.shape)

# print(x11.values.shape)
# print(y)
clf=LinearRegression()
# print(y.values)
clf.fit(x11.values, y.values)
# print(clf.predict(np.array([1, 2, 3]).reshape(1, -1)))


y1=len(df1.columns)-2
X1=np.zeros(y1)
years=['1951-68','1968-81','1981-91','1991-97','1997-02','2002-07','2007-12']

for count in range(y1):
	X1[count]=int(count)


plt.plot(X1,y)
plt.title('Agricultural Productivity growth across different years')
plt.ylabel("% growth of agricultural productivity")
plt.xlabel("years")
plt.xticks(X1, years)
plt.savefig('agriculture.png')
plt.show()


z = np.polyfit(X1, x1, 3)
f = np.poly1d(z)
# print(f)

x_new = np.linspace(X1[0], X1[-1], 50)
y_new = f(x_new)

plt.plot(X1, x1, 'o', x_new, y_new)
plt.title('Fitting plot for growth in land')
plt.ylabel( "% growth of land" )
plt.xlabel( "years" )
plt.xticks(X1, years)
plt.savefig('growthland.png')
plt.show()

z2 = np.polyfit(X1, x2, 2)
f2 = np.poly1d(z2)
# print(f)

x_new_2 = np.linspace(X1[0], X1[-1], 50)
y_new_2 = f2(x_new_2)

plt.plot(X1, x2, 'o', x_new_2, y_new_2)
plt.title('Fitting plot for growth in labour')
plt.ylabel( "% growth of labour")
plt.xlabel( "years" )
plt.xticks(X1, years)
plt.savefig('growthlabour.png')
plt.show()


z3 = np.polyfit(X1, x3, 2)
f3 = np.poly1d(z3)
# print(f)

x_new_3 = np.linspace(X1[0], X1[-1], 50)
y_new_3 = f3(x_new_3)

plt.plot(X1, x3, 'o', x_new_3, y_new_3)
plt.title('Fitting plot for growth in Net Fixed Capital Stock')
plt.ylabel( "% growth of Net Fixed Capital Stock")
plt.xlabel( "years" )
plt.xticks(X1, years)
plt.savefig('growthcapital.png')
# plt.show()

x_final=np.array([f(7),f2(7),f3(7)])

print('% growth of total crops from 2012-present is estimated to be '+str(clf.predict(x_final.reshape(1, -1))[0]))
