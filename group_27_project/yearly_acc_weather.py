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
import pickle
#df = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/Acc_Classified_according_to_Type_of_Weather_Condition_2014_and_2016.csv",low_memory=False)
df = pd.read_csv("Acc_Classified_according_to_Type_of_Weather_Condition_2014_and_2016.csv",low_memory=False)
#df2  = pd.read_csv("C:/Users/Pranjal DADA/Desktop/DM_proj/road_funds2013_14_15_16.csv",low_memory=False)
d = df.columns
df[d[11]] = df[d[11]] + df[d[14]]
d1 = [ d[5], d[8], d[11],d[20], d[23], d[29] ]
d2 = [df.iloc[-1][5], df.iloc[-1][8], df.iloc[-1][11], df.iloc[-1][20],df.iloc[-1][23],df.iloc[-1][29]]
d3 = [ df.iloc[-1][44], df.iloc[-1][47], df.iloc[-1][50], df.iloc[-1][56],df.iloc[-1][53], df.iloc[-1][59]]
ind = [2014, 2016]
t1 = df.iloc[-1][2]
t2 = df.iloc[-1][41]
dof = 5
data = [d2,d3]
df = pd.DataFrame(data, index = ind, columns = d1)
col_s = sum(df.values)
s1 = sum(d2)
s2 = sum(d3)
s = s1 + s2
ans = 0
for i in range(6):
    for j in range(2):
        e = 0
        if (j == 0):
            e = (s1*col_s[i])/s
        else:
            e = (s2*col_s[i])/s
        ans += ((df.iloc[j][i] - e)**2) / e
print("NULL hypothsesis not accepted with 1, 5 and 10% level of significance. So, Yearwise road accidents and wheather are not independent")
print("~25% of accidents are due to bad weather")
print(df.sum(axis = 1).iloc[0] , "out of " , df.sum(axis = 1).iloc[0] + t1 , "are due to bad weather in 2014")
print(df.sum(axis = 1).iloc[1] , "out of " , df.sum(axis = 1).iloc[1] + t2 , "are due to bad weather in 2016")
labels = ['Mist/fog','Cloudy', 'Rain', 'Hail/sleet', 'Snow', 'Dust storm']
y = np.asarray(d3)
y = y/1000
X = []
for x in labels:
    if ( x == 'Mist/fog'):
        X.append(0)
    elif ( x == 'Cloudy' ):
        X.append(1)
    elif (x == 'Rain'):
        X.append(2)
    elif (x == 'Hail/sleet' ):
        X.append(3)
    elif (x == 'Snow' ):
        X.append(4)
    elif (x == 'Dust storm' ):
        X.append(5)
plt.xlabel('Weather Conditions')
plt.ylabel('Number of accidents in thousands')
plt.bar(X, y, color = '#006652')
plt.xticks(X, labels)
#fig.savefig('test.jpg')
#plt.show()
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ind = np.arange(2)
x3 =[87627, 83594]
y3 = [397748, 429162]    # the x locations for the groups
width = 0.25
X = [0,1]        
 # the width of the bars
states = ["2014", "2016"]
p1 = plt.bar(ind, x3, width, color='#ff6600')
p2 = plt.bar(ind+width, y3, width, color='#006652')
plt.xticks(X, states, rotation=0)
plt.ylabel('Number of accidents')
plt.xlabel('Year')
plt.legend((p1[0],p2[0]),('Accidents due to bad weather', 'Total accidents'))
#plt.title('SDP Growth and Manufacturing growth for FY10-11')
plt.savefig('weather_vs_total.png')
