import pandas as pd

# 1. Carregar os datasets
# Substitui 'caminho/do/ficheiro.csv' pelos nomes reais dos teus ficheiros
df_antibioticos = pd.read_csv('antibioticos-carbapenemes.csv')
df_consultas = pd.read_csv('01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv')
df_urgencias = pd.read_csv('atendimentos-por-tipo-de-urgencia-hospitalar-link.csv')

# 2. Preparar Dataset 1: Antibióticos
# Selecionamos as colunas e já renomeamos para o padrão final
df1 = df_antibioticos[['Periodo', 'Hospital', 'Regiao', 'DDD Consumidas de Carbapenemes', 'DDD Consumidas dos Restantes Antibióticos']].copy()
df1.rename(columns={
    'DDD Consumidas de Carbapenemes': 'Consumo',
    'DDD Consumidas dos Restantes Antibióticos': 'Sobra'
}, inplace=True)

# 3. Preparar Dataset 2: Consultas
# Aqui a coluna chama-se 'Instituição', vamos manter para o merge e depois descartar
df2 = df_consultas[['Periodo', 'Instituição', 'Regiao', 'Nº Consultas Médicas Total']].copy()
df2.rename(columns={'Nº Consultas Médicas Total': 'nr consultas'}, inplace=True)

# 4. Preparar Dataset 3: Urgências
df3 = df_urgencias[['Periodo', 'Instituição', 'Regiao', 'Urgências Geral', 'Urgências Pediátricas', 'Urgência Obstetricia', 'Urgência Psiquiátrica', 'Total Urgências']].copy()

# --- PROCESSO DE JUNÇÃO (MERGE) ---

# Passo A: Juntar Antibióticos com Consultas
# Usamos 'Hospital' do primeiro e 'Instituição' do segundo
merged_df = pd.merge(
    df1, 
    df2, 
    left_on=['Periodo', 'Hospital', 'Regiao'], 
    right_on=['Periodo', 'Instituição', 'Regiao'], 
    how='left'
)

# Remover a coluna 'Instituição' que ficou duplicada após o merge
merged_df.drop(columns=['Instituição'], inplace=True)

# Passo B: Juntar o resultado anterior com as Urgências
final_df = pd.merge(
    merged_df, 
    df3, 
    left_on=['Periodo', 'Hospital', 'Regiao'], 
    right_on=['Periodo', 'Instituição', 'Regiao'], 
    how='left'
)

# Remover novamente a coluna 'Instituição' repetida
final_df.drop(columns=['Instituição'], inplace=True)

# 5. Guardar o resultado final
final_df.to_csv('dataset_hospitalar_combinado.csv', index=False, encoding='utf-8-sig')

print("Dataset combinado com sucesso! Colunas finais:")
print(final_df.columns.tolist())