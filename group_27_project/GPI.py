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
df = pd.read_csv("IPC_Crime_headwise_distribution_of_inmates_who_convicted.csv",low_memory=False)
df2 = pd.read_csv("Statement_SES_2011-12-GPI.csv",low_memory=False)

crime = df.groupby("CRIME HEAD")
crime_rape = ( crime.get_group("RAPE").reset_index(drop=True ).iloc[:,0:3] )
crime_rape_cnt = crime.get_group("RAPE").reset_index(drop=True ).iloc[:,8]
rapes = pd.concat([crime_rape, crime_rape_cnt ], axis=1)

rape = []
dowry = []
cruelty_hus = []
molestation = []
GPI = []
years = []
for i in range(2001,2012):
    rape.append(sum(rapes.loc[rapes['YEAR'] == i]["Total 18+AC0-30 years"]))
    
crime_dowry = ( crime.get_group("DOWRY DEATHS").reset_index(drop=True ).iloc[:,0:3] )
crime_dowry_cnt = crime.get_group("DOWRY DEATHS").reset_index(drop=True ).iloc[:,8]
dowrys = pd.concat([crime_dowry, crime_dowry_cnt ], axis=1)
for i in range(2001,2012):
    dowry.append(sum(dowrys.loc[dowrys['YEAR'] == i]["Total 18+AC0-30 years"]))
    
crime_MOLESTATION = ( crime.get_group("MOLESTATION").reset_index(drop=True ).iloc[:,0:3] )
crime_MOLESTATION_cnt = crime.get_group("MOLESTATION").reset_index(drop=True ).iloc[:,8]
MOLESTATION = pd.concat([crime_MOLESTATION, crime_MOLESTATION_cnt ], axis=1)
for i in range(2001,2012):
    molestation.append(sum(MOLESTATION.loc[MOLESTATION['YEAR'] == i]["Total 18+AC0-30 years"]))
    
crime_cruelty_hus = ( crime.get_group("CRUELTY BY HUSBAND OR RELATIVE OF HUSBAND").reset_index(drop=True ).iloc[:,0:3] )
crime_cruelty_hus_cnt = crime.get_group("CRUELTY BY HUSBAND OR RELATIVE OF HUSBAND").reset_index(drop=True ).iloc[:,8]
cruelty = pd.concat([crime_cruelty_hus, crime_cruelty_hus_cnt ], axis=1)
for i in range(2001,2012):
    cruelty_hus.append(sum(cruelty.loc[cruelty['YEAR'] == i]["Total 18+AC0-30 years"]))
    years.append(i)

    
GPI = df2["Classes VI+AC0-VIII (11+AC0-13 Years) +AC0- SC"][5:16].values
temp = GPI[6:]
reg = linear_model.LinearRegression()
X_train = temp.reshape(-1,1)
reg.fit(X_train, rape[6:])
t = [1]
t = np.asarray(t)
test = t.reshape(-1,1)
ans  = reg.predict(test)
print("RAPE AT GPI = 1", ans)

reg.fit(X_train, molestation[6:])
ans  = reg.predict(test)
print("MOLESTATION AT GPI = 1", ans)

reg.fit(X_train, cruelty_hus[6:])
ans  = reg.predict(test)
print("CRUELTY BY HUSBAND AT GPI = 1", ans)

reg.fit(X_train, dowry[6:])
ans  = reg.predict(test)
print("DOWRY AT GPI = 1", ans)

X = []
labels = ['RAPE','MOLESTATION','CRUELTY BY HUSBAND','DOWRY']
for i in range(4):
    X.append(i)
    
p1 = plt.plot(GPI, rape, 'o-', label=labels[0])
p2 = plt.plot(GPI, molestation,'o-', label=labels[1])
p3 = plt.plot(GPI, cruelty_hus,'o-', label=labels[2])
p4 = plt.plot(GPI, dowry,'o-', label=labels[3])
plt.xlabel('GPI')
plt.ylabel('Number of Crimes')
plt.legend()
#plt.savefig("crime_vs_GPI.jpg")
plt.show()
plt.plot(years, GPI,'o-')
plt.xlabel('Years')
plt.ylabel('GPI')
plt.show()
crimes = []
for i in range(11):
    t = dowry[i] + molestation[i] + cruelty_hus[i] + rape[i]
    crimes.append(t)
plt.plot(years, crimes,'o-')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()
# 57.6 lac - 1999

df = pd.read_csv("primaryschool.csv" , low_memory=False)
y = [361088,	439234771,	548159652,	683329097,	846421039,	1028737436,	1210569573]
x_train = [1951,1961,1971,1981,1991,2001, 2011]
x_train = np.asarray(x_train)
x_train = x_train - 1951
x_train = x_train.reshape(-1,1)
reg.fit(x_train, y)
t = [47]
t = np.asarray(t)
test = t.reshape(-1,1)
ans  = reg.predict(test)
