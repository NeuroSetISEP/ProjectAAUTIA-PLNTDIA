#!/usr/bin/env python3
"""
Script otimizado para criar dataset de ML para previsão de uso de Carbapenemes
Inclui features adicionais críticas para melhor performance do modelo
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

def normalize_period(period):
    """Normaliza formato de período para YYYY-MM"""
    if pd.isna(period):
        return period
    period = str(period).strip()
    if re.match(r'^\d{4}-\d{2}$', period):
        return period
    if re.match(r'^\d{4}/\d{2}/\d{2}$', period):
        return period[:7].replace('/', '-')
    return period

def normalize_regiao(regiao):
    """Normaliza nomes de regiões de saúde"""
    if pd.isna(regiao):
        return regiao
    regiao = str(regiao).strip()
    mapeamento = {
        'Centro': 'Região de Saúde do Centro',
        'Norte': 'Região de Saúde Norte',
        'LVT': 'Região de Saúde LVT',
        'Alentejo': 'Região de Saúde do Alentejo',
        'Algarve': 'Região de Saúde do Algarve',
    }
    if regiao.startswith('Região de Saúde'):
        return regiao
    if regiao in mapeamento:
        return mapeamento[regiao]
    return regiao

def normalize_hospital_name(hospital):
    """Normaliza nomes de hospitais para matching entre datasets"""
    if pd.isna(hospital):
        return hospital

    hospital = str(hospital).strip()
    import unicodedata
    hospital = unicodedata.normalize('NFD', hospital)
    hospital = ''.join(char for char in hospital if unicodedata.category(char) != 'Mn')
    hospital = hospital.replace('-', ' ')
    hospital = re.sub(r'\bProf\.?\s', 'Professor ', hospital, flags=re.IGNORECASE)
    hospital = re.sub(r'\bDr\.?\s', 'Doutor ', hospital, flags=re.IGNORECASE)
    hospital = re.sub(r'\bDra\.?\s', 'Doutora ', hospital, flags=re.IGNORECASE)
    hospital = re.sub(r',?\s*[E\.]\.?\s*[P\.]\.?\s*[E\.]\.?\s*$', '', hospital, flags=re.IGNORECASE)
    hospital = re.sub(r',?\s*EPE\s*$', '', hospital, flags=re.IGNORECASE)
    hospital = re.sub(r',?\s*PPP\s*$', '', hospital, flags=re.IGNORECASE)
    hospital = hospital.replace('/', ' ')
    hospital = re.sub(r'\b(do|da|de|dos|das|e|os)\b', ' ', hospital, flags=re.IGNORECASE)
    hospital = hospital.replace('.', '')
    hospital = re.sub(r'[,\-\s]+', ' ', hospital)
    hospital = hospital.strip()
    return hospital

def load_and_aggregate_carbapenemes():
    """Carrega dados de Carbapenemes"""
    print("Carregando dados de Carbapenemes...")
    df = pd.read_csv('antibioticos-carbapenemes.csv', sep=';')
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['ARS'].apply(normalize_regiao)
    df['Instituicao'] = df['Hospital'].apply(normalize_hospital_name)

    df_agg = df.groupby(['Periodo', 'Regiao', 'Instituicao']).agg({
        'DDD Consumidas de Carbapenemes': 'sum',
        'DDD Consumidas dos Restantes Antibióticos': 'sum',
        'Peso': 'mean'
    }).reset_index()

    df_agg = df_agg.rename(columns={
        'DDD Consumidas de Carbapenemes': 'Consumo_Carbapenemes',
        'DDD Consumidas dos Restantes Antibióticos': 'Consumo_Outros_Antibioticos',
        'Peso': 'Peso_Medio_Carbapenemes'
    })

    df_agg['Consumo_Total_Antibioticos'] = (
        df_agg['Consumo_Carbapenemes'] + df_agg['Consumo_Outros_Antibioticos']
    )

    print(f"  - {len(df_agg)} registros de Carbapenemes")
    return df_agg

def load_and_aggregate_cefalosporinas():
    """Carrega dados de Cefalosporinas (NOVO - importante para resistência)"""
    print("Carregando dados de Cefalosporinas...")
    df = pd.read_csv('my_datasets/antibioticos-cefalosporinas.csv', sep=';')
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['ARS'].apply(normalize_regiao)
    df['Instituicao'] = df['Hospital'].apply(normalize_hospital_name)

    df_agg = df.groupby(['Periodo', 'Regiao', 'Instituicao']).agg({
        'DDD Consumidas de Cefalosporinas': 'sum',
        'Peso': 'mean'
    }).reset_index()

    df_agg = df_agg.rename(columns={
        'DDD Consumidas de Cefalosporinas': 'Consumo_Cefalosporinas',
        'Peso': 'Peso_Medio_Cefalosporinas'
    })

    print(f"  - {len(df_agg)} registros de Cefalosporinas")
    return df_agg

def load_and_aggregate_urgencias():
    """Carrega dados de urgências"""
    print("Carregando dados de urgências...")
    df = pd.read_csv('atendimentos-por-tipo-de-urgencia-hospitalar-link.csv', sep=';')
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['Região']
    df['Instituicao'] = df['Instituição'].apply(normalize_hospital_name)

    df_agg = df.groupby(['Periodo', 'Regiao', 'Instituicao']).agg({
        'Total Urgências': 'sum',
        'Urgências Geral': 'sum',
        'Urgências Pediátricas': 'sum',
        'Urgência Obstetricia': 'sum',
        'Urgência Psiquiátrica': 'sum'
    }).reset_index()

    df_agg = df_agg.rename(columns={
        'Total Urgências': 'Total_Urgencias',
        'Urgências Geral': 'Urgencias_Geral',
        'Urgências Pediátricas': 'Urgencias_Pediatricas',
        'Urgência Obstetricia': 'Urgencias_Obstetricia',
        'Urgência Psiquiátrica': 'Urgencias_Psiquiatrica'
    })

    print(f"  - {len(df_agg)} registros de urgências")
    return df_agg

def load_and_aggregate_triagem_manchester():
    """Carrega dados de Triagem Manchester (NOVO - crítico para severidade)"""
    print("Carregando dados de Triagem Manchester...")
    df = pd.read_csv('my_datasets/atendimentos-em-urgencia-triagem-manchester.csv', sep=';')
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['Região']
    df['Instituicao'] = df['Instituição'].apply(normalize_hospital_name)

    # Colunas de triagem
    colunas_triagem = {
        'Nº Atendimentos em Urgência SU Triagem Manchester -Vermelha': 'Triagem_Vermelha',
        'Nº Atendimentos em Urgência SU Triagem Manchester -Laranja': 'Triagem_Laranja',
        'Nº Atendimentos em Urgência SU Triagem Manchester -Amarela': 'Triagem_Amarela',
        'Nº Atendimentos em Urgência SU Triagem Manchester -Verde': 'Triagem_Verde',
        'Nº Atendimentos em Urgência SU Triagem Manchester -Azul': 'Triagem_Azul',
        'Nº Atendimentos em Urgência SU Triagem Manchester -Branca': 'Triagem_Branca'
    }

    agg_dict = {col: 'sum' for col in colunas_triagem.keys() if col in df.columns}

    df_agg = df.groupby(['Periodo', 'Regiao', 'Instituicao']).agg(agg_dict).reset_index()
    df_agg = df_agg.rename(columns=colunas_triagem)

    # Calcular total de triagens
    cols_triagem = ['Triagem_Vermelha', 'Triagem_Laranja', 'Triagem_Amarela',
                    'Triagem_Verde', 'Triagem_Azul', 'Triagem_Branca']
    df_agg['Total_Triagens'] = df_agg[cols_triagem].sum(axis=1)

    print(f"  - {len(df_agg)} registros de Triagem Manchester")
    return df_agg

def load_and_aggregate_consultas():
    """Carrega dados de consultas"""
    print("Carregando dados de consultas...")
    df = pd.read_csv('01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv', sep=';')
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['Região']
    df['Instituicao'] = df['Instituição'].apply(normalize_hospital_name)

    df_agg = df.groupby(['Periodo', 'Regiao', 'Instituicao']).agg({
        'Nº Consultas Médicas Total': 'sum',
        'Nº Primeiras Consultas': 'sum',
        'Nº Consultas Subsequentes': 'sum'
    }).reset_index()

    df_agg = df_agg.rename(columns={
        'Nº Consultas Médicas Total': 'Total_Consultas',
        'Nº Primeiras Consultas': 'Primeiras_Consultas',
        'Nº Consultas Subsequentes': 'Consultas_Subsequentes'
    })

    print(f"  - {len(df_agg)} registros de consultas")
    return df_agg

def load_populacao():
    """Carrega dados de população por região"""
    print("Carregando dados de população...")
    df = pd.read_csv('my_datasets/Municipios.csv')
    df = df.rename(columns={
        '01. Ano': 'Ano',
        '04. Âmbito Geográfico': 'Municipio',
        '10. Valor': 'Populacao',
        'Região de Saúde': 'Regiao'
    })

    df_pop = df.groupby(['Ano', 'Regiao'])['Populacao'].sum().reset_index()
    df_pop = df_pop.rename(columns={'Populacao': 'Populacao_Regiao'})

    df_municipios = df.groupby(['Ano', 'Regiao'])['Municipio'].count().reset_index()
    df_municipios = df_municipios.rename(columns={'Municipio': 'Num_Municipios'})

    df_pop = df_pop.merge(df_municipios, on=['Ano', 'Regiao'])

    print(f"  - {len(df_pop)} registros de população")
    return df_pop

def create_temporal_features(df):
    """Cria features temporais avançadas (NOVO)"""
    print("\nCriando features temporais avançadas...")

    # Ordenar por instituição, região e período
    df = df.sort_values(['Instituicao', 'Regiao', 'Periodo'])

    # Lag features (valores do mês anterior) - importantes para prever tendências
    lag_cols = ['Consumo_Carbapenemes', 'Total_Urgencias', 'Total_Consultas',
                'Triagem_Vermelha', 'Triagem_Laranja']

    for col in lag_cols:
        if col in df.columns:
            df[f'{col}_Lag1'] = df.groupby(['Instituicao', 'Regiao'])[col].shift(1)
            df[f'{col}_Lag2'] = df.groupby(['Instituicao', 'Regiao'])[col].shift(2)
            df[f'{col}_Lag3'] = df.groupby(['Instituicao', 'Regiao'])[col].shift(3)

    # Rolling means (médias móveis de 3 e 6 meses)
    rolling_cols = ['Consumo_Carbapenemes', 'Total_Urgencias', 'Triagem_Vermelha']

    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_Rolling_Mean_3'] = df.groupby(['Instituicao', 'Regiao'])[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df[f'{col}_Rolling_Mean_6'] = df.groupby(['Instituicao', 'Regiao'])[col].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )

    # Taxa de crescimento mensal
    if 'Consumo_Carbapenemes' in df.columns:
        df['Taxa_Crescimento_Carbapenemes'] = df.groupby(['Instituicao', 'Regiao'])['Consumo_Carbapenemes'].pct_change() * 100

    print("  - Features temporais criadas com sucesso")
    return df

def create_ratio_features(df):
    """Cria features de proporções e ratios (NOVO)"""
    print("Criando features de proporções...")

    # Proporções de urgências por tipo
    if 'Total_Urgencias' in df.columns:
        for col in ['Urgencias_Geral', 'Urgencias_Pediatricas', 'Urgencias_Obstetricia', 'Urgencias_Psiquiatrica']:
            if col in df.columns:
                df[f'Prop_{col}'] = (df[col] / df['Total_Urgencias'] * 100).fillna(0)

    # Proporções de triagem Manchester (severidade)
    if 'Total_Triagens' in df.columns:
        triagem_cols = ['Triagem_Vermelha', 'Triagem_Laranja', 'Triagem_Amarela',
                       'Triagem_Verde', 'Triagem_Azul', 'Triagem_Branca']
        for col in triagem_cols:
            if col in df.columns:
                df[f'Prop_{col}'] = (df[col] / df['Total_Triagens'] * 100).fillna(0)

        # Índice de severidade (weighted score baseado nas cores)
        # Vermelha=5, Laranja=4, Amarela=3, Verde=2, Azul=1, Branca=0
        df['Indice_Severidade'] = (
            (df.get('Triagem_Vermelha', 0) * 5 +
             df.get('Triagem_Laranja', 0) * 4 +
             df.get('Triagem_Amarela', 0) * 3 +
             df.get('Triagem_Verde', 0) * 2 +
             df.get('Triagem_Azul', 0) * 1 +
             df.get('Triagem_Branca', 0) * 0) /
            df['Total_Triagens']
        ).fillna(0)

    # Ratio de primeiras consultas vs subsequentes
    if 'Total_Consultas' in df.columns and 'Primeiras_Consultas' in df.columns:
        df['Ratio_Primeiras_Consultas'] = (df['Primeiras_Consultas'] / df['Total_Consultas'] * 100).fillna(0)

    # Ratio Carbapenemes / Cefalosporinas (indicador de resistência)
    if 'Consumo_Carbapenemes' in df.columns and 'Consumo_Cefalosporinas' in df.columns:
        df['Ratio_Carbapenemes_Cefalosporinas'] = (
            df['Consumo_Carbapenemes'] / (df['Consumo_Cefalosporinas'] + 1)  # +1 para evitar divisão por zero
        ).round(4)

    print("  - Features de proporções criadas com sucesso")
    return df

def merge_datasets():
    """Faz o merge de todos os datasets"""
    print("\n=== Iniciando merge dos datasets OTIMIZADOS ===\n")

    # Carregar todos os datasets
    df_carbapenemes = load_and_aggregate_carbapenemes()
    df_cefalosporinas = load_and_aggregate_cefalosporinas()
    df_urgencias = load_and_aggregate_urgencias()
    df_triagem = load_and_aggregate_triagem_manchester()
    df_consultas = load_and_aggregate_consultas()
    df_populacao = load_populacao()

    print("\n--- Fazendo merge ---")

    # Começar com carbapenemes como base
    df_final = df_carbapenemes.copy()

    # Merge com cefalosporinas
    print("Merging com Cefalosporinas...")
    df_final = df_final.merge(
        df_cefalosporinas,
        on=['Periodo', 'Regiao', 'Instituicao'],
        how='outer'
    )

    # Merge com urgências
    print("Merging com urgências...")
    df_final = df_final.merge(
        df_urgencias,
        on=['Periodo', 'Regiao', 'Instituicao'],
        how='outer'
    )

    # Merge com triagem Manchester
    print("Merging com Triagem Manchester...")
    df_final = df_final.merge(
        df_triagem,
        on=['Periodo', 'Regiao', 'Instituicao'],
        how='outer'
    )

    # Merge com consultas
    print("Merging com consultas...")
    df_final = df_final.merge(
        df_consultas,
        on=['Periodo', 'Regiao', 'Instituicao'],
        how='outer'
    )

    # Extrair ano para merge com população
    df_final['Ano'] = df_final['Periodo'].str[:4].astype(float)

    # Merge com população
    print("Merging com população...")
    df_final = df_final.merge(
        df_populacao,
        on=['Ano', 'Regiao'],
        how='left'
    )

    # Preencher NaN com 0 para colunas de contagem
    print("\nProcessando valores vazios...")
    colunas_contagem = [col for col in df_final.columns if any(x in col for x in
                        ['Consumo', 'Total', 'Urgencias', 'Triagem', 'Consultas', 'Peso', 'Primeiras'])]

    for col in colunas_contagem:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    # Features temporais
    df_final['Ano'] = df_final['Periodo'].str[:4].astype(int)
    df_final['Mes'] = df_final['Periodo'].str[5:7].astype(int)
    df_final['Trimestre'] = ((df_final['Mes'] - 1) // 3) + 1
    df_final['Semestre'] = ((df_final['Mes'] - 1) // 6) + 1
    df_final['Dia_Do_Ano'] = df_final.apply(
        lambda row: datetime(int(row['Ano']), int(row['Mes']), 1).timetuple().tm_yday, axis=1
    )

    # Calcular percentuais e per capita
    if 'Consumo_Total_Antibioticos' in df_final.columns:
        df_final['Percentual_Carbapenemes'] = (
            df_final['Consumo_Carbapenemes'] / (df_final['Consumo_Total_Antibioticos'] + 1) * 100
        ).round(2)

    if 'Populacao_Regiao' in df_final.columns:
        df_final['Consumo_Carbapenemes_Per_Capita'] = (
            df_final['Consumo_Carbapenemes'] / df_final['Populacao_Regiao'] * 100000
        ).round(2)
        df_final['Total_Urgencias_Per_Capita'] = (
            df_final['Total_Urgencias'] / df_final['Populacao_Regiao'] * 100000
        ).round(2)
        df_final['Total_Consultas_Per_Capita'] = (
            df_final['Total_Consultas'] / df_final['Populacao_Regiao'] * 100000
        ).round(2)

    # Criar features temporais avançadas
    df_final = create_temporal_features(df_final)

    # Criar features de proporções
    df_final = create_ratio_features(df_final)

    # Ordenar por região, instituição e período
    df_final = df_final.sort_values(['Regiao', 'Instituicao', 'Periodo'])

    print(f"\n=== Dataset OTIMIZADO criado com {len(df_final)} registros e {len(df_final.columns)} features ===")

    # Estatísticas
    print(f"\n--- Estatísticas ---")
    print(f"  Períodos: {df_final['Periodo'].min()} até {df_final['Periodo'].max()}")
    print(f"  Regiões: {df_final['Regiao'].nunique()}")
    print(f"  Instituições: {df_final['Instituicao'].nunique()}")
    print(f"  Registros com Carbapenemes: {(df_final['Consumo_Carbapenemes'] > 0).sum()}")
    print(f"  Registros com Triagem Manchester: {(df_final.get('Total_Triagens', pd.Series([0])) > 0).sum()}")
    print(f"  Registros com Cefalosporinas: {(df_final.get('Consumo_Cefalosporinas', pd.Series([0])) > 0).sum()}")

    return df_final

def main():
    """Função principal"""
    print("="*80)
    print("CRIAÇÃO DE DATASET OTIMIZADO PARA ML - PREVISÃO DE CARBAPENEMES")
    print("="*80)

    # Fazer merge
    df_final = merge_datasets()

    # Salvar resultado
    output_file = 'dataset_medicamentos_optimized.csv'
    df_final.to_csv(output_file, sep=';', index=False)
    print(f"\n✓ Dataset otimizado salvo em: {output_file}")

    # Estatísticas detalhadas
    print("\n" + "="*80)
    print("ESTATÍSTICAS DO DATASET OTIMIZADO")
    print("="*80)

    print(f"\nDimensões: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")
    print(f"\n--- Categorias de Features ---")

    feature_categories = {
        'Temporais': [col for col in df_final.columns if any(x in col for x in ['Ano', 'Mes', 'Trimestre', 'Semestre', 'Periodo', 'Dia'])],
        'Antibióticos': [col for col in df_final.columns if any(x in col for x in ['Consumo', 'Carbapenemes', 'Cefalosporinas', 'Antibioticos', 'Peso'])],
        'Urgências': [col for col in df_final.columns if 'Urgencias' in col or 'Urgencia' in col],
        'Triagem Manchester': [col for col in df_final.columns if 'Triagem' in col],
        'Consultas': [col for col in df_final.columns if 'Consultas' in col],
        'População': [col for col in df_final.columns if 'Populacao' in col or 'Municipios' in col],
        'Proporções/Ratios': [col for col in df_final.columns if 'Prop_' in col or 'Ratio_' in col or 'Percentual' in col],
        'Features Temporais Avançadas': [col for col in df_final.columns if 'Lag' in col or 'Rolling' in col or 'Taxa_Crescimento' in col],
        'Índices': [col for col in df_final.columns if 'Indice' in col]
    }

    for categoria, features in feature_categories.items():
        if features:
            print(f"\n{categoria}: {len(features)} features")
            for feat in features[:5]:  # Mostrar apenas as primeiras 5
                print(f"  - {feat}")
            if len(features) > 5:
                print(f"  ... e mais {len(features) - 5}")

    print("\n--- Valores nulos (apenas colunas com missing data) ---")
    nulos = df_final.isnull().sum()
    nulos_pct = (nulos / len(df_final) * 100).round(2)
    nulos_df = pd.DataFrame({
        'Valores Nulos': nulos[nulos > 0],
        'Percentual (%)': nulos_pct[nulos > 0]
    })
    if len(nulos_df) > 0:
        print(nulos_df.head(10))
    else:
        print("Nenhum valor nulo!")

    print("\n--- Estatísticas descritivas (variáveis principais) ---")
    colunas_principais = ['Consumo_Carbapenemes', 'Consumo_Cefalosporinas',
                         'Total_Urgencias', 'Total_Triagens', 'Indice_Severidade']
    colunas_existentes = [col for col in colunas_principais if col in df_final.columns]
    if colunas_existentes:
        print(df_final[colunas_existentes].describe())

    print("\n" + "="*80)
    print("✓ DATASET OTIMIZADO CRIADO COM SUCESSO!")
    print("="*80)
    print("\nMelhorias implementadas:")
    print("  ✓ Dados de Cefalosporinas (resistência antibiótica)")
    print("  ✓ Triagem Manchester (severidade dos casos)")
    print("  ✓ Features temporais (lag, rolling means, crescimento)")
    print("  ✓ Proporções e ratios (severidade, tipos de urgência)")
    print("  ✓ Índice de severidade ponderado")
    print("  ✓ Ratio Carbapenemes/Cefalosporinas")

if __name__ == "__main__":
    main()
