import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv('my_datasets/Municipios.csv')

# Filtrar apenas as linhas que contêm "Município" na coluna "04. Âmbito Geográfico"
df_municipios = df[df['04. Âmbito Geográfico'] == 'Município']

# Salvar o arquivo filtrado
df_municipios.to_csv('my_datasets/Municipios.csv', index=False)

print(f"Filtrado! Total de linhas mantidas: {len(df_municipios)}")
print(f"Municípios únicos: {df_municipios['03. Nome Região (Portugal)'].nunique()}")
