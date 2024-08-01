import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


names = ['Recente','Frequente','Monetario','Tempo','Doou_Sangue_em_Marco_de_2007']  
home_data = pd.read_csv(r'D:\Cauã Ernani\MineracaoDeDados_Blood-Transfusion_Service_Center-main\MineracaoDeDados_Blood-Transfusion_Service_Center-main\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names, usecols = ['Frequente', 'Monetario', 'Tempo'])
home_data.head()

sns.scatterplot(x = 'Frequente', y = 'Tempo', hue = 'Monetario', data = home_data)

X_train, X_test, y_train, y_test = train_test_split(home_data[['Frequente', 'Tempo']], home_data[['Monetario']], test_size=0.33, random_state=0)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'Frequente', y = 'Tempo', hue = kmeans.labels_)

sns.boxplot(x = kmeans.labels_, y = y_train['Monetario'])
plt.show()