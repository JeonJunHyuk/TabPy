import numpy as np
import pandas as pd


df = pd.read_csv('c:/users/junhyuk/downloads/TabPy materials/Cars.csv')
df.head()
pd.set_option('display.max_columns',50)
pd.set_option('display.width',300)



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df) #에러. df.info() 보면 object. label encoder


le = LabelEncoder()
df['2 Class Cluster'].unique()
df['Make'].unique()
df.Vehicle.unique()
df.Vehicle.value_counts()
df.Vehicle.value_counts().sort_values(ascending=False)
df.Make.value_counts()
df = df.set_index(['Make','Vehicle'])
df
le_df = le.fit_transform(df)  # 안돼.
le_df = le.fit_transform(df['2 Class Cluster'])
le_df
le_df.shape
len(le_df)
df['2 Class Cluster'] = le_df  # 해도 되지만 아래 것으로.
df['2 Class Cluster'] = le.fit_transform(df['2 Class Cluster'])

#인덱스로 만들고, le 했으니 이제 될걸?
scaled_df = scaler.fit_transform(df)

pca = PCA(n_components=10)
pca_scores = pca.fit_transform(scaled_df)
pca_scores
pca_scores.shape
x = pca_scores[:,0]  # 각 obs. 의 PC1 score
x
y = pca_scores[:,1]  # PC2 score
y
pca.components_.shape  # 가로: loading vector, 세로: 각 column 별 주성분 계수값?
pca.components_[0,0], pca.components_[0,1]  # PC1, PC2 scatter plot 에 찍을 1st column point 좌표
x0_pc1, x0_pc2 = pca.components_[0,0], pca.components_[0,1]
x1_pc1, x1_pc2 = pca.components_[1,0], pca.components_[1,1]
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

from matplotlib import pyplot as plt

fig1 = 
plt.scatter(x, y)
plt.scatter(x0_pc1, x0_pc2)
plt.scatter(x1_pc1, x1_pc2)
plt.grid()
plt.show()
