import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("DemographicData.csv")
X = dataset.iloc[:,2:4].values
z = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
z = lab.fit_transform(z)
lab.classes_

plt.scatter(X[:,0],X[:,1])
plt.show()

plt.scatter(X[z==0,0],X[z==0,1],c='r',label='High Income')
plt.scatter(X[z==1,0],X[z==1,1],c='g',label='Low Income')
plt.scatter(X[z==2,0],X[z==2,1],c='b',label='Lower Middle Income')
plt.scatter(X[z==3,0],X[z==3,1],c='y',label='Upper Middle Income')
plt.legend()
plt.xlabel('Birth Rate')
plt.ylabel('Internet Users')
plt.title('Analysis on the World Bank Dataset')
plt.show()




dataset = pd.read_csv('StudentsPerformance.csv')

import seaborn as sns

dataset.info()
dataset.describe()
dataset.isnull().any()
dataset.isnull().sum()

dataset.columns = ['gender',
                   'race',
                   'ped',
                   'lunch',
                   'test',
                   'math',
                   'reading',
                   'writing']

plt.hist(dataset['math'],bins = 100)
plt.hist(dataset['reading'],bins = 100)
plt.hist(dataset['writing'],bins = 100)

sns.barplot(x = dataset['gender'].value_counts().index,
            y = dataset['gender'].value_counts(),
            hue = ['female','male'])

sns.barplot(dataset['race'],dataset['math'],
            hue = dataset['gender'])

sns.barplot(dataset['ped'],dataset['math'],
            hue = dataset['gender'])

sns.barplot(dataset['lunch'],dataset['math'],
            hue = dataset['gender'])

sns.barplot(dataset['test'],dataset['math'],
             hue = dataset['gender'])
           

sns.boxplot(dataset['math'])
sns.boxplot(dataset['reading'])
sns.boxplot(dataset['writing'])

sns.boxplot(dataset['gender'],dataset['math'])
sns.boxplot(dataset['gender'],dataset['reading'])
sns.boxplot(dataset['gender'],dataset['writing'])  














































