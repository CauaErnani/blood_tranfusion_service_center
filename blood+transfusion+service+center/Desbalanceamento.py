import pandas as pd
import matplotlib.pyplot as plt

# Leitura da base de dados
names = ['Recency','Frequency','Monetary','Time','Donated_in_March_2007']
df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

# Contagem dos valores na variável dependente
count_0 = df['Donated_in_March_2007'].value_counts()[0]
count_1 = df['Donated_in_March_2007'].value_counts()[1]
total = len(df)

# Cálculo das porcentagens
percentage_0 = (count_0 / total) * 100
percentage_1 = (count_1 / total) * 100

# Plotagem do gráfico de pizza
labels = ['0', '1']
sizes = [percentage_0, percentage_1]
colors = ['lightcoral', 'lightskyblue']
explode = (0, 0.1)  # Destacar a segunda fatia (1)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribuição da Variável Dependente')
plt.axis('equal')  # Garantir que o gráfico seja exibido como um círculo
plt.show()
