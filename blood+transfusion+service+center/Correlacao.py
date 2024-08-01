import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

names = ['Recency','Frequency','Monetary','Time','Donated_in_March_2007']

df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

features = ['Recency','Frequency','Monetary','Time']
target = ['Donated_in_March_2007']

# Excluir a variável dependente 'Doou_Sangue_em_Marco_de_2007' do DataFrame
independente_df = df[features]

# Calcular a matriz de correlação
correlation = independente_df.corr()

# Imprimir a matriz de correlação
print(correlation)

# Plotar a matriz de correlação usando um heatmap
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 12})
sn.heatmap(correlation, cmap='viridis', vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", square=True, linewidths=.5)
plt.show()
