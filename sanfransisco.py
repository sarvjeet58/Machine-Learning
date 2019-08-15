import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import folium
from wordcloud import WordCloud

df = pd.read_csv("I:\sarv\Police_Department_Incidents_-_Previous_Year__2016_.csv")

df.isnull().sum()
df.head()  
df.describe()
df.dtypes

df['PdDistrict'].fillna(df['PdDistrict'].mode()[0],inplace=True)
df.isnull().sum()

sns.countplot(df['Category'])
plt.title('Crime in Sanfransisco')
plt.xticks(rotation = 90)
plt.show()

a = df['Category'].value_counts().head(25)
squarify.plot(sizes = a.values, label = a.index, alpha=.8)
plt.title('Tree Map for Top 25 Crimes')
plt.axis('off')
plt.show()

wc = WordCloud(background_color='Blue',width=2000,height=2000).generate(str(df['Descript']))
plt.imshow(wc)
plt.title('Description of crime')
plt.axis('off')
plt.show()

df['PdDistrict'].value_counts().plot.bar(figsize=(15,10))
plt.title('District Vs crime')
plt.xticks(rotation=90)
plt.show()
















































  