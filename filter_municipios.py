import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv('my_datasets/Municipios.csv')

df_filtrado = df[(df['04. Âmbito Geográfico'] == 'Município') &
                  (df['05. Filtro 1'] == 'Total') &
                  (df['06. Filtro 2'] == 'Total')]

df_filtrado.to_csv('my_datasets/Municipios.csv', index=False)

print(f"Filtrado! Total de linhas mantidas: {len(df_filtrado)}")
print(f"Municípios únicos: {df_filtrado['03. Nome Região (Portugal)'].nunique()}")
