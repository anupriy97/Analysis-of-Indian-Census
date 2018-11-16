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
df = pd.read_csv("Acc_classified_according_to_Vehicular_Defect_2012_and_2016.csv",low_memory=False)
d = df.columns
d1 = [d[2], d[5], d[8], d[11], d[14]]
d2 = [df.iloc[-1][2], df.iloc[-1][5], df.iloc[-1][8], df.iloc[-1][11], df.iloc[-1][14]]
d3 = [df.iloc[-1][17], df.iloc[-1][20], df.iloc[-1][23], df.iloc[-1][26], df.iloc[-1][29]]
ind = [2014, 2016]
data = [d2,d3]
df = pd.DataFrame(data, index = ind, columns = d1)
col_s = sum(df.values)
s1 = sum(d2)
s2 = sum(d3)
s = s1 + s2
ans = 0
for i in range(5):
    for j in range(2):
        e = 0
        if (j == 0):
            e = (s1*col_s[i])/s
        else:
            e = (s2*col_s[i])/s
        #print(e, df.iloc[j][i])
        ans += ((df.iloc[j][i] - e)**2) / e
labels = ['Other defects','Defective Steering','Punctured/burst Typres','Bald Tyres','Defective brakes']
y = d2
y1 = d3
X = []
XX = []
for x in labels:
    if (x == 'Other defects' ):
        X.append(0)
        XX.append(0)
    elif ( x == 'Defective Steering'):
        X.append(1)
        XX.append(1)
    elif ( x == 'Punctured/burst Typres' ):
        X.append(2)
        XX.append(2)
    elif (x == 'Bald Tyres' ):
        X.append(3)
        XX.append(3)
    elif (x == 'Defective brakes' ):
        X.append(4)
        XX.append(4)
explode = (0,0, 0, 0, 0.1) 

fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
axs[0].pie(y,labels=labels, explode=explode,shadow=True,autopct='%1.1f%%', startangle=90)
axs[1].pie(y1,labels=labels, explode=explode,shadow=True,autopct='%1.1f%%', startangle=90)
#fig.suptitle('Distribution of accidents due to Vehicular Defects')
#fig1 ,ax1 = plt.subplots(1,2, figsize=(9, 3), sharey=True)
#fig2 ,ax2 = plt.subplots()
#ax1.pie(y,labels=labels, explode=explode,shadow=True,autopct='%1.1f%%', startangle=90)
#ax2.pie(y1,labels=labels, explode=explode,shadow=True,autopct='%1.1f%%', startangle=90)
#ax1.axis('equal')
#ax2.axis('equal'))

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ind = np.arange(2)
x3 =[56801, 61495]
y3 = [110651, 96268]    # the x locations for the groups
width = 0.25
X = [0,1]        
 # the width of the bars
states = ["2014", "2016"]
p1 = plt.bar(ind, x3, width, color='#ff6600')
p2 = plt.bar(ind+width, y3, width, color='#006652')
plt.xticks(X, states, rotation=0)
plt.ylabel('Number of accidents')
plt.xlabel('Year')
plt.legend((p1[0],p2[0]),('Accidents due to break defect', 'Total accidents'))
#plt.title('SDP Growth and Manufacturing growth for FY10-11')
plt.savefig('break_vs_total.png')
#plt.show()


print("NULL hypothsesis not accepted with 1, 5 and 10% level of significance. So, Yearwise road accidents and type of vehicular defects are not independent")
print("more than  50% accidents are due to break failure")
print(df.max(axis = 1).iloc[0] , "out of " , df.sum(axis = 1).iloc[0] , "are due to break failure in 2014")
print(df.max(axis = 1).iloc[1] , "out of " , df.sum(axis = 1).iloc[1] , "are due to break failure in 2016")