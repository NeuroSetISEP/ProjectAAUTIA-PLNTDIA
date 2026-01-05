import pandas as pd
import numpy as np
import re

# Nomes dos teus ficheiros (devem estar na mesma pasta)
files = [
    "atividade-de-internamento-hospitalar.csv",
    "atendimentos-por-tipo-de-urgencia-hospitalar-link.csv",
    "evolucao-da-prescricao-electronica-de-medicamentos.csv",
    "antibioticos-carbapenemes.csv"
]

# Ler os ficheiros
dfs = {}
for f in files:
    # Ajusta o 'sep' se necessário (os teus ficheiros parecem usar ';')
    dfs[f] = pd.read_csv(f, sep=';', encoding='utf-8')

# Funções de limpeza
def clean_region(val):
    if pd.isna(val): return np.nan
    val = str(val).lower()
    if 'alentejo' in val: return 'Alentejo'
    if 'algarve' in val: return 'Algarve'
    if 'centro' in val: return 'Centro'
    if 'norte' in val: return 'Norte'
    if 'lvt' in val or 'lisboa' in val: return 'LVT'
    return val

def clean_hospital(val):
    if pd.isna(val): return np.nan
    val = str(val).lower()
    val = re.sub(r'[,.\-]', ' ', val)      # Remove pontuação
    val = re.sub(r'\be\s?p\s?e\b', '', val) # Remove EPE
    val = re.sub(r'\bppp\b', '', val)       # Remove PPP
    val = re.sub(r'\s+', ' ', val).strip()  # Remove espaços extra
    return val

# --- 1. Processar Internamento ---
df1 = dfs[files[0]].copy()
df1['Region_Clean'] = df1['Região'].apply(clean_region)
df1['Hospital_Clean'] = df1['Instituição'].apply(clean_hospital)
# Pivot para transformar linhas de especialidade em colunas
df1_pivot = df1.pivot_table(
    index=['Período', 'Region_Clean', 'Hospital_Clean'],
    columns='Tipo de Especialidade',
    values=['Doentes Saídos', 'Dias de Internamento'],
    aggfunc='sum'
).reset_index()
# Aplanar nomes das colunas
df1_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df1_pivot.columns.values]

# --- 2. Processar Urgências ---
df2 = dfs[files[1]].copy()
df2['Region_Clean'] = df2['Região'].apply(clean_region)
df2['Hospital_Clean'] = df2['Instituição'].apply(clean_hospital)
df2_clean = df2.drop(columns=['Região', 'Instituição', 'Período.Format.2', 'Localização Geográfica'])
df2_clean = df2_clean.groupby(['Período', 'Region_Clean', 'Hospital_Clean']).sum().reset_index()

# --- 3. Processar Prescrições (Regional) ---
df3 = dfs[files[2]].copy()
df3['Region_Clean'] = df3['Região'].apply(clean_region)
df3_clean = df3.drop(columns=['Região']).groupby(['Período', 'Region_Clean']).sum().reset_index()

# --- 4. Processar Antibióticos ---
df4 = dfs[files[3]].copy()
df4['Region_Clean'] = df4['ARS'].apply(clean_region)
df4['Hospital_Clean'] = df4['Hospital'].apply(clean_hospital)
df4_clean = df4.drop(columns=['ARS', 'Hospital', 'Localização Geográfica', 'Grupo Hospitalar'])
df4_clean = df4_clean.groupby(['Período', 'Region_Clean', 'Hospital_Clean']).sum().reset_index()

# --- MERGE FINAL ---
# Criar lista mestra de chaves (Data + Região + Hospital)
all_keys = pd.concat([
    df1_pivot[['Período', 'Region_Clean', 'Hospital_Clean']],
    df2_clean[['Período', 'Region_Clean', 'Hospital_Clean']],
    df4_clean[['Período', 'Region_Clean', 'Hospital_Clean']]
]).drop_duplicates()

merged = pd.merge(all_keys, df1_pivot, on=['Período', 'Region_Clean', 'Hospital_Clean'], how='left')
merged = pd.merge(merged, df2_clean, on=['Período', 'Region_Clean', 'Hospital_Clean'], how='left')
merged = pd.merge(merged, df4_clean, on=['Período', 'Region_Clean', 'Hospital_Clean'], how='left')
merged = pd.merge(merged, df3_clean, on=['Período', 'Region_Clean'], how='left')

# Guardar ficheiro
merged.to_csv("dataset_sns_consolidado.csv", index=False, sep=';', encoding='utf-8-sig')
print("Ficheiro criado com sucesso!")