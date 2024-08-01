import pandas as pd
import math
import matplotlib.pyplot as plt

# Leitura da base de dados
names = ['Recente','Frequente','Monetario','Tempo','Doou_Sangue_em_Marco_de_2007']
df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

# Colunas numéricas para análise
numeric_columns = ['Recente','Frequente','Monetario','Tempo']

# Calcular a Amplitude Total (AT) e Amplitude da Classe (h) para cada coluna
at = df[numeric_columns].apply(lambda col: col.max() - col.min())
k = math.sqrt(len(df))
h = (at / k).apply(math.ceil)

# Calcular os Intervalos de Classe para cada coluna
intervals = {}
for col in numeric_columns:
    min_val = df[col].min()
    max_val = df[col].max()
    interval = [(min_val + i * h[col], min_val + (i + 1) * h[col]) for i in range(math.ceil((max_val - min_val) / h[col]))]
    intervals[col] = interval

# Calcular as Frequências Absolutas para a coluna 'Recency'
freq_abs = pd.cut(df['Recente'], bins=[interval[0] for interval in intervals['Recente']] + [df['Recente'].max()], include_lowest=True, right=True).value_counts()

print("Frequências Absolutas para a coluna 'Recente':")
print(freq_abs)

# Plotar um histograma da coluna 'Recency'
plt.hist(df['Recente'], bins='auto', color='blue', alpha=0.7)
plt.title('Histograma da Recente')
plt.xlabel('Recente')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()
