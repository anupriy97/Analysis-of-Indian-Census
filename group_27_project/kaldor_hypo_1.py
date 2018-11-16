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
df1 = pd.read_csv("manufacturing_growth.csv",low_memory=False)
df1 = df1.sort_values(by = ["States/Uts"])

df2 = pd.read_csv("nsdp_state_wise.csv",low_memory=False)
df2 = df2.sort_values(by = ["States/Uts"])

df2=df2.drop(['2005-2006', '2006-2007', '2007-2008', '2011-2012', '2012-2013', '2013-2014', '2014-2015'], axis=1)
states=df1['States/Uts']
states.index = range(len(states.index))
# print(states)

x1=df2['2008-2009']
x2=df2['2009-2010']
x3=df2['2010-2011']
y1=df1['2008-2009']
y2=df1['2009-2010']
y3=df1['2010-2011']

x1.index=range(32)
x2.index=range(32)
x3.index=range(32)
y1.index=range(32)
y2.index=range(32)
y3.index=range(32)

X = []
y=0
for x in states:
    X.append(y)
    y+=1

font = {'weight' : 'bold',
        'size'   : 6}

matplotlib.rc('font', **font)

N=32
# fig, ax = plt.subplots()
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = plt.bar(ind, x1, width, color='m')
p2 = plt.bar(ind+width, y1, width, color='b')
plt.xticks(X, states, rotation=80)
plt.ylabel('Growth in %')
plt.legend((p1[0],p2[0]),('SDP Growth', 'Manufacturing Growth'))
plt.title('SDP Growth and Manufacturing growth for FY08-09')
plt.savefig('growth1.png')
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = plt.bar(ind, x2, width, color='m')
p2 = plt.bar(ind+width, y2, width, color='b')
plt.xticks(X, states, rotation=80)
plt.ylabel('Growth in %')
plt.legend((p1[0],p2[0]),('SDP Growth', 'Manufacturing Growth'))
plt.title('SDP Growth and Manufacturing growth for FY09-10')
plt.savefig('growth2.png')

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = plt.bar(ind, x3, width, color='m')
p2 = plt.bar(ind+width, y3, width, color='b')
plt.xticks(X, states, rotation=80)
plt.ylabel('Growth in %')
plt.legend((p1[0],p2[0]),('SDP Growth', 'Manufacturing Growth'))
plt.title('SDP Growth and Manufacturing growth for FY10-11')
plt.savefig('growth3.png')

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

for z in range(32):
	x=[x1[z],x2[z],x3[z]]
	y=[y1[z],y2[z],y3[z]]
	plt.plot(x, y,marker='o', alpha=0.5, label=states[z])
	plt.title('Average SDP growth vs Average Manufacturing growth')
	plt.xlabel( "Average SDP growth" )
	plt.ylabel( "Average Manufacturing growth" )
	plt.legend()
	if(states[z]=='Jharkhand'):
		print(x,y)
		plt.savefig('jharkhand.png')
	if(states[z]=='Nagaland'):
		plt.savefig('Nagaland.png')
	if(states[z]=='UP'):
		plt.savefig('UP.png')
	if(states[z]=='Andhra'):
		plt.savefig('Andhra.png')
	plt.show()