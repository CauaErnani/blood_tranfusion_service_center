import pandas as pd

# Leitura da base de dados
names = ['Recency','Frequency','Monetary','Time','Donated_in_March_2007']
df = pd.read_csv(r'C:\Users\cauae\Downloads\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)

# Excluindo a variável dependente (última coluna)
features = df.iloc[:, :-1]

# Calculando a média das características (features)
mean_values = features.mean()

# Calculando as estatísticas descritivas para cada coluna
statistics = pd.DataFrame({
    'Média': mean_values.round(2),
    'Amplitude': (features.max() - features.min()).round(2),
    'Desvio Padrão': features.std().round(2),
    'Coeficiente de Variação (%)': (features.std() / features.mean() * 100).round(2)
})

# Exibindo as estatísticas descritivas
print("Estatísticas Descritivas das Features:")
print(statistics)
