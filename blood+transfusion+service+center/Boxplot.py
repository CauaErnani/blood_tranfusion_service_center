import pandas as pd
import matplotlib.pyplot as plt

# Leitura da base de dados
names = ['Recency','Frequency','Monetary','Time','Donated_in_March_2007']
df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

# Boxplot para cada atributo
plt.figure(figsize=(10, 6))
df.drop('Donated_in_March_2007', axis=1).boxplot()
plt.title('Boxplot para Cada Atributo')
plt.xlabel('Atributos')
plt.ylabel('Valores')
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para facilitar a leitura
plt.grid(True)
plt.show()
