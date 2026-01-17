import pandas as pd

# Carregar dados de carbapenemes
df = pd.read_csv('antibioticos-carbapenemes.csv', sep=';')

print(f"Total de registros: {len(df)}")
print(f"Hospitais únicos: {df['Hospital'].nunique()}")
print(f"Regiões únicas: {df['ARS'].nunique()}")

print("\n=== TODOS OS HOSPITAIS ÚNICOS ===")
hospitais = sorted(df['Hospital'].unique())
for i, hospital in enumerate(hospitais, 1):
    print(f"{i:2d}. {hospital}")

# Contar registros por hospital
print("\n=== HOSPITAIS COM MAIS DADOS ===")
contagem = df['Hospital'].value_counts().head(10)
for hospital, count in contagem.items():
    print(f"{hospital}: {count} registros")