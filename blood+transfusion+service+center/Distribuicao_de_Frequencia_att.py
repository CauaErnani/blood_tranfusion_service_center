import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leitura da base de dados
names = ['Recency','Frequency','Monetary','Time','Donated_in_March_2007']
df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

# Definição do número de intervalos de classe usando o método de Sturges
n = len(df)
k = 1 + int(round(np.log2(n)))

# Lista de atributos
attributes = ['Recency', 'Frequency', 'Monetary', 'Time']

# Plotagem dos histogramas para cada atributo
for attribute in attributes:
    plt.figure(figsize=(8, 6))
    plt.hist(df[df['Donated_in_March_2007'] == 0][attribute], bins=k, alpha=0.5, color='blue', label='Não doou em Março de 2007')
    plt.hist(df[df['Donated_in_March_2007'] == 1][attribute], bins=k, alpha=0.5, color='red', label='Doou em Março de 2007')
    plt.title(f'Distribuição de Frequência de {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True)
    plt.show()
