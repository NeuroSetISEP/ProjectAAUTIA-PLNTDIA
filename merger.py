import pandas as pd

# Função para limpar e normalizar nomes de Hospitais/Instituições
def limpar_texto(txt):
    if pd.isna(txt): return ""
    txt = str(txt).upper().strip()
    # Remove termos comuns que variam entre bases
    termos_para_remover = ["E.P.E.", "E. P. E.", "EPE", ",", "-", "  "]
    for termo in termos_para_remover:
        txt = txt.replace(termo, " ")
    return " ".join(txt.split()) # Remove espaços duplos

# Função para detetar o separador e carregar o CSV com segurança
def carregar_csv(caminho):
    # Tenta detetar se é vírgula ou ponto e vírgula
    try:
        return pd.read_csv(caminho, sep=None, engine='python', encoding='utf-8-sig')
    except:
        return pd.read_csv(caminho, sep=',', encoding='latin1')

# 1. Carregar os ficheiros
df_ant = carregar_csv('antibioticos-carbapenemes.csv')
df_con = carregar_csv('01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv')
df_urg = carregar_csv('atendimentos-por-tipo-de-urgencia-hospitalar-link.csv')

# 2. Normalizar nomes de colunas (remover acentos e espaços para evitar erros)
# Vamos garantir que 'Periodo', 'Hospital' e 'Regiao' existem sem acentos no código
for df in [df_ant, df_con, df_urg]:
    df.columns = [c.replace('Período', 'Periodo').replace('Região', 'Regiao').replace('Instituição', 'Hospital') for c in df.columns]

# 3. Criar uma coluna "Hospital_Key" simplificada para fazer o Merge
# Isto ignora se está escrito "Hospital de Braga" ou "Hosp. Braga EPE"
for df in [df_ant, df_con, df_urg]:
    df['Hospital_Key'] = df['Hospital'].apply(limpar_texto)
    df['Periodo'] = df['Periodo'].astype(str).str.strip()

# 4. Preparar colunas específicas
df1 = df_ant[['Periodo', 'Hospital', 'Hospital_Key', 'Região', 'DDD Consumidas de Carbapenemes', 'DDD Consumidas dos Restantes Antibióticos']].copy()
df1.rename(columns={'DDD Consumidas de Carbapenemes': 'Consumo', 'DDD Consumidas dos Restantes Antibióticos': 'Sobra'}, inplace=True)

df2 = df_con[['Periodo', 'Hospital_Key', 'Nº Consultas Médicas Total']].copy()
df2.rename(columns={'Nº Consultas Médicas Total': 'nr consultas'}, inplace=True)

df3 = df_urg[['Periodo', 'Hospital_Key', 'Urgências Geral', 'Urgências Pediátricas', 'Urgência Obstetricia', 'Urgência Psiquiátrica', 'Total Urgências']].copy()

# 5. Realizar o Merge usando a chave limpa (Hospital_Key)
# Cruzamos apenas por Periodo e Hospital_Key para maior precisão
merged = pd.merge(df1, df2, on=['Periodo', 'Hospital_Key'], how='left')
final_df = pd.merge(merged, df3, on=['Periodo', 'Hospital_Key'], how='left')

# 6. Limpeza final: remover a chave auxiliar e organizar
final_df.drop(columns=['Hospital_Key'], inplace=True)

# Opcional: Padronizar a coluna Região (ex: transformar "Região de Saúde LVT" em apenas "LVT")
def padronizar_regiao(r):
    r = str(r).upper()
    if 'LVT' in r: return 'LVT'
    if 'NORTE' in r: return 'Norte'
    if 'CENTRO' in r: return 'Centro'
    if 'ALENTEJO' in r: return 'Alentejo'
    if 'ALGARVE' in r: return 'Algarve'
    return r

final_df['Regiao'] = final_df['Regiao'].apply(padronizar_regiao)

# Guardar
final_df.to_csv('dataset_hospitalar_final_corrigido.csv', index=False, encoding='utf-8-sig')

print("Ficheiro processado com sucesso!")
print(f"Total de linhas: {len(final_df)}")
print(f"Colunas: {final_df.columns.tolist()}")