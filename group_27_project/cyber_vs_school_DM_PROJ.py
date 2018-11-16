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
df1 = pd.read_csv("cyber_cases.csv",low_memory=False)
df = df1.sort_values(by = ["State/ Uts"])
df2 = pd.read_csv("percentage_schools_having_computer.csv",low_memory=False)
df2 = df2.iloc[:35, :]
a = df["Students"]
b = df2["Upper Primary Schools/ Sections 2012-13"]
d = [a,b]
data = pd.concat(d,axis = 1)
correlation = data.corr(method='pearson')
plt.scatter(a.values, b.values, c="b", alpha=0.5)
plt.title('Student_cyber_crime vs % of school with computes')
plt.xlabel( "Student_cyber_crime" )
plt.ylabel( "% of school with computes" )




