import numpy as np
import pandas as pd

df = pd.read_csv('c:/users/junhyuk/downloads/TabPy materials/Cars.csv')
df.head()
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
# 에러. df.info() 보면 object. label encoder
df.info()

df['2 Class Cluster'].unique()
df['Make'].unique()
df.Vehicle.unique()
df.Vehicle.value_counts()
df.Vehicle.value_counts().sort_values(ascending=False)
df.Make.value_counts()

df = df.set_index(['Make', 'Vehicle'])
df

le = LabelEncoder()
le_df = le.fit_transform(df)
le_df = le.fit_transform(df['2 Class Cluster'])
le_df
le_df.shape
len(le_df)
df['2 Class Cluster'] = le_df  # 해도 되지만 아래 것으로 한 번에.
df['2 Class Cluster'] = le.fit_transform(df['2 Class Cluster'])  # 이거시 결론

# 인덱스로 만들고, le 했으니 이제 될걸?
scaled_df = scaler.fit_transform(df)
scaled_df
scaled_df.shape
pca = PCA(n_components=5)
pca_scores = pca.fit_transform(scaled_df)
pca_scores
pca_scores.shape
PC1_score = pca_scores[:, 0]  # 각 obs. 의 PC1 score
PC2_score = pca_scores[:, 1]  # PC2 score

pca.components_.shape  # 가로: loading vector
pca.components_[0, 0], pca.components_[0, 1]  # PC1, PC2 scatter plot 에 찍을 1st column point 좌표
x0 = [pca.components_[0, 0], pca.components_[0, 1]]
x1 = [pca.components_[1, 0], pca.components_[1, 1]]
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

from matplotlib import pyplot as plt

plt.scatter(PC1_score, PC2_score)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('dkrnlcksgdk')

plt.plot(pca.explained_variance_ratio_)

def myplot(score, loading, labels=None):
    PC1s = score[:, 0]
    PC2s = score[:, 1]
    n = loading.shape[0]
    scalex = 1.0 / (PC1s.max() - PC1s.min())
    scaley = 1.0 / (PC2s.max() - PC2s.min())
    plt.scatter(PC1s * scalex, PC2s * scaley, c='b',alpha=0.3)
    for i in range(n):
        plt.arrow(0, 0, loading[i, 0], loading[i, 1], color='r', alpha=0.3)
        if labels is None:
            plt.text(loading[i, 0] * 1.15, loading[i, 1] * 1.15,
                     "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(loading[i, 0] * 1.15, loading[i, 1] * 1.15,
                     labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


myplot(pca_scores[:, 0:2], np.transpose(pca.components_[0:2, :]), df.columns)
