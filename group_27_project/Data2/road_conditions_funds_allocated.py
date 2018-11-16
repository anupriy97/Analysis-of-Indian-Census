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
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import random
import csv
import math
import pickle

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
def col(x, z):
    return math.floor(x/z)
def row(x, z):
    return math.floor(x/z)
#df = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/Acc_Classified_according_to_Type_of_Weather_Condition_2014_and_2016.csv",low_memory=False)
df = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/ACC_road_cond.csv",low_memory=False)
df2  = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/road_funds2013_14_15_16.csv",low_memory=False)
df3   = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/population.csv",low_memory=False)
d = df.columns
popu = np.asarray(df3[df3.columns[2]].values)
#z = math.floor((max(df2['Funds Allocated (in Crores) - 2014-15'].values)/5)) + 1
z = math.floor((max(df2[df2.columns[2]].values)/5)) + 1
ind=["0-" + str(z),str(z)+"-"+str(2*z),str(2*z)+"-"+str(3*z),str(3*z)+"-"+str(4*z),str(4*z)+"-"+str(5*z)]
#df_con = [df.iloc[:][5], df.iloc[-1][8], df.iloc[-1][11], df.iloc[-1][20],df.iloc[-1][23],df.iloc[-1][29]]

acc=np.zeros(36)

for x in range(0,36):
    for y in range(2,72,3):
       acc[x] += df.iloc[x][y]
z1 = math.floor(max(acc) / 5) + 1
ind1 = ["0-" + str(z1),str(z1)+"-"+str(2*z1),str(2*z1)+"-"+str(3*z1),str(3*z1)+"-"+str(4*z1),str(4*z1)+"-"+str(5*z1)]
funds = df2[df2.columns[2]].values
mat = np.zeros([5,5])

for x in range(0,36):
    j = col( funds[x], z )
    i = row( acc[x], z1 )
    mat[i][j] += 1
r1 = np.sum(mat, axis = 1)
c1 = np.sum(mat, axis = 0)
s = np.sum(c1)
ans = 0
for i in range(0,5):
    for j in range(0,5):
        e = (r1[i] * c1[j]) / s
        ans += (e-mat[i][j])**2 / e
dof = 16
states = df[d[1]].values
x = np.asarray(df2[df2.columns[2]].values)
x *= 10000000
#x = (x - np.min(x))/(np.max(x) - np.min(x))
popu = popu[:36]
x = x / popu
y = np.asarray(acc)
#y = (y - np.min(y))/(np.max(y) - np.min(y))
y /= 1000
dd = {"funds" : y, "Accidents" : x}
data = pd.DataFrame(dd)
corr = data.corr()
X = []
for i in range(36):
    X.append(i)
#a, b = best_fit(x,y)
plt.xlabel('Funds in INR/Person for road maintainace')
plt.ylabel('Number of accidents in thousands')
plt.scatter(x, y, color = '#006652')
#plt.xticks(X, states,rotation=90)
#yfit = [a + b * xi for xi in x]
#plt.plot(x, yfit)
#plt.show()
print("By chi square test it is observed that number of accident on a particular road condition  is dependent on fund allocated to that road conditions at 5% and 10% significance level but independent for 1% significance level")
