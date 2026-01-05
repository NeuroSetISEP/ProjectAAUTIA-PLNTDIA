#!/usr/bin/env python3


import pandas as pd
import numpy as np
import re

def normalize_period(period):
    """
    Normaliza formato de período para YYYY-MM
    """
    if pd.isna(period):
        return period

    period = str(period).strip()

    if re.match(r'^\d{4}-\d{2}$', period):
        return period

    # convert para o formato YYYY-MM
    if re.match(r'^\d{4}/\d{2}/\d{2}$', period):
        return period[:7].replace('/', '-')

    return period

def normalize_regiao(regiao):
    """
    Normaliza nomes de regiões de saúde
    """
    if pd.isna(regiao):
        return regiao

    regiao = str(regiao).strip()

    # Mapear variações conhecidas
    mapeamento = {
        'Centro': 'Região de Saúde do Centro',
        'Norte': 'Região de Saúde Norte',
        'LVT': 'Região de Saúde LVT',
        'Alentejo': 'Região de Saúde do Alentejo',
        'Algarve': 'Região de Saúde do Algarve',
    }

    if regiao.startswith('Região de Saúde'):
        return regiao

    # mapear abreviações
    if regiao in mapeamento:
        return mapeamento[regiao]

    return regiao

def load_and_aggregate_antibioticos():
    """Carrega e agrega dados de antibióticos por Região + Período"""
    print("Carregando dados de antibióticos...")
    df = pd.read_csv('antibioticos-carbapenemes.csv', sep=';')

    # Norm periodo e regiao
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['ARS'].apply(normalize_regiao)

    # Agregar
    df_agg = df.groupby(['Periodo', 'Regiao']).agg({
        'DDD Consumidas de Carbapenemes': 'sum',
        'DDD Consumidas dos Restantes Antibióticos': 'sum',
        'Peso': 'mean',
        'Hospital': 'count'
    }).reset_index()


    df_agg = df_agg.rename(columns={
        'DDD Consumidas de Carbapenemes': 'Consumo_Carbapenemes',
        'DDD Consumidas dos Restantes Antibióticos': 'Consumo_Outros_Antibioticos',
        'Peso': 'Peso_Medio_Carbapenemes',
        'Hospital': 'Num_Hospitais_Antibioticos'
    })

    # Calcular total de antibióticos
    df_agg['Consumo_Total_Antibioticos'] = (
        df_agg['Consumo_Carbapenemes'] +
        df_agg['Consumo_Outros_Antibioticos']
    )

    # Calcular percentual de carbapenemes
    df_agg['Percentual_Carbapenemes'] = (
        df_agg['Consumo_Carbapenemes'] /
        df_agg['Consumo_Total_Antibioticos'] * 100
    ).round(2)

    print(f"  - {len(df_agg)} registros agregados (Região + Período)")
    return df_agg

def load_and_aggregate_urgencias():
    """Carrega e agrega dados de urgências por Região + Período"""
    print("Carregando dados de urgências...")
    df = pd.read_csv('atendimentos-por-tipo-de-urgencia-hospitalar-link.csv', sep=';')

    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['Região']

    df_agg = df.groupby(['Periodo', 'Regiao']).agg({
        'Total Urgências': 'sum',
        'Urgências Geral': 'sum',
        'Urgências Pediátricas': 'sum',
        'Urgência Obstetricia': 'sum',
        'Urgência Psiquiátrica': 'sum',
        'Instituição': 'count'
    }).reset_index()

    df_agg = df_agg.rename(columns={
        'Total Urgências': 'Total_Urgencias',
        'Urgências Geral': 'Urgencias_Geral',
        'Urgências Pediátricas': 'Urgencias_Pediatricas',
        'Urgência Obstetricia': 'Urgencias_Obstetricia',
        'Urgência Psiquiátrica': 'Urgencias_Psiquiatrica',
        'Instituição': 'Num_Hospitais_Urgencias'
    })

    print(f"  - {len(df_agg)} registros agregados (Região + Período)")
    return df_agg

def load_and_aggregate_consultas():
    """Carrega e agrega dados de consultas por Região + Período"""
    print("Carregando dados de consultas...")
    df = pd.read_csv('01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv', sep=';')

    # Normalizar período
    df['Periodo'] = df['Período'].apply(normalize_period)
    df['Regiao'] = df['Região']  # Já está no formato correto

    # Agregar por Região + Período
    df_agg = df.groupby(['Periodo', 'Regiao']).agg({
        'Nº Consultas Médicas Total': 'sum',
        'Nº Primeiras Consultas': 'sum',
        'Nº Consultas Subsequentes': 'sum',
        'Instituição': 'count'
    }).reset_index()

    # Renomear colunas
    df_agg = df_agg.rename(columns={
        'Nº Consultas Médicas Total': 'Total_Consultas',
        'Nº Primeiras Consultas': 'Primeiras_Consultas',
        'Nº Consultas Subsequentes': 'Consultas_Subsequentes',
        'Instituição': 'Num_Hospitais_Consultas'
    })

    print(f"  - {len(df_agg)} registros agregados (Região + Período)")
    return df_agg

def load_populacao():
    """Carrega e processa dados de população por região"""
    print("Carregando dados de população...")
    df = pd.read_csv('my_datasets/Municipios.csv')

    # Renomear colunas
    df = df.rename(columns={
        '01. Ano': 'Ano',
        '04. Âmbito Geográfico': 'Municipio',
        '10. Valor': 'Populacao',
        'Região de Saúde': 'Regiao'
    })

    # Agregar população por região e ano
    df_pop = df.groupby(['Ano', 'Regiao'])['Populacao'].sum().reset_index()
    df_pop = df_pop.rename(columns={'Populacao': 'Populacao_Regiao'})

    # Contar quantos municípios por região
    df_municipios = df.groupby(['Ano', 'Regiao'])['Municipio'].count().reset_index()
    df_municipios = df_municipios.rename(columns={'Municipio': 'Num_Municipios'})

    df_pop = df_pop.merge(df_municipios, on=['Ano', 'Regiao'])

    print(f"  - {len(df_pop)} registros agregados por região")
    return df_pop

def merge_datasets():
    """Faz o merge de todos os datasets agregados por região"""
    print("\n=== Iniciando merge dos datasets (agregação por região) ===\n")

    # Carregar todos os datasets agregados
    df_antibioticos = load_and_aggregate_antibioticos()
    df_urgencias = load_and_aggregate_urgencias()
    df_consultas = load_and_aggregate_consultas()
    df_populacao = load_populacao()

    print("\n--- Fazendo merge ---")

    # Começar com antibióticos como base
    df_final = df_antibioticos.copy()

    # Merge com urgências
    print("Merging com urgências (por Região + Período)...")
    df_final = df_final.merge(
        df_urgencias,
        on=['Periodo', 'Regiao'],
        how='outer'
    )

    # Merge com consultas
    print("Merging com consultas (por Região + Período)...")
    df_final = df_final.merge(
        df_consultas,
        on=['Periodo', 'Regiao'],
        how='outer'
    )

    # Extrair ano do período para merge com população
    df_final['Ano'] = df_final['Periodo'].str[:4].astype(float)

    # Merge com população
    print("Merging com população (por Região + Ano)...")
    df_final = df_final.merge(
        df_populacao,
        on=['Ano', 'Regiao'],
        how='left'
    )

    # Criar features adicionais úteis para ML
    print("\nCriando features adicionais...")

    # Taxas per capita (por 100k habitantes)
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

    # Features temporais
    df_final['Ano'] = df_final['Periodo'].str[:4].astype(int)
    df_final['Mes'] = df_final['Periodo'].str[5:7].astype(int)
    df_final['Trimestre'] = ((df_final['Mes'] - 1) // 3) + 1
    df_final['Semestre'] = ((df_final['Mes'] - 1) // 6) + 1

    # Ordenar colunas de forma lógica
    colunas_ordem = [
        'Periodo', 'Ano', 'Mes', 'Trimestre', 'Semestre', 'Regiao',
        'Populacao_Regiao', 'Num_Municipios',
        'Consumo_Carbapenemes', 'Consumo_Outros_Antibioticos',
        'Consumo_Total_Antibioticos', 'Percentual_Carbapenemes',
        'Consumo_Carbapenemes_Per_Capita',
        'Total_Urgencias', 'Urgencias_Geral', 'Urgencias_Pediatricas',
        'Urgencias_Obstetricia', 'Urgencias_Psiquiatrica',
        'Total_Urgencias_Per_Capita',
        'Total_Consultas', 'Primeiras_Consultas', 'Consultas_Subsequentes',
        'Total_Consultas_Per_Capita',
        'Num_Hospitais_Antibioticos', 'Num_Hospitais_Urgencias', 'Num_Hospitais_Consultas',
        'Peso_Medio_Carbapenemes'
    ]

    # Manter apenas colunas que existem
    colunas_existentes = [col for col in colunas_ordem if col in df_final.columns]
    df_final = df_final[colunas_existentes]

    # Ordenar por região e período
    df_final = df_final.sort_values(['Regiao', 'Periodo'])

    print(f"\n=== Dataset final criado com {len(df_final)} registros ===")
    print(f"Colunas: {len(df_final.columns)}")

    return df_final

def main():
    """Função principal"""
    print("="*70)
    print("CRIAÇÃO DE DATASET AGREGADO POR REGIÃO PARA PREVISÃO DE MEDICAMENTOS")
    print("="*70)

    # Fazer merge
    df_final = merge_datasets()

    # Salvar resultado
    output_file = 'dataset_medicamentos_por_regiao.csv'
    df_final.to_csv(output_file, sep=';', index=False)
    print(f"\n✓ Dataset salvo em: {output_file}")

    # Estatísticas do dataset
    print("\n" + "="*70)
    print("ESTATÍSTICAS DO DATASET")
    print("="*70)

    print(f"\nDimensões: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")
    print(f"Períodos: {df_final['Periodo'].min()} a {df_final['Periodo'].max()}")
    print(f"Períodos únicos: {df_final['Periodo'].nunique()}")
    print(f"Regiões únicas: {df_final['Regiao'].nunique()}")

    print("\n--- Distribuição por Região ---")
    print(df_final['Regiao'].value_counts().sort_index())

    print("\n--- Valores nulos por coluna ---")
    nulos = df_final.isnull().sum()
    nulos_pct = (nulos / len(df_final) * 100).round(2)
    nulos_df = pd.DataFrame({
        'Valores Nulos': nulos[nulos > 0],
        'Percentual (%)': nulos_pct[nulos > 0]
    })
    if len(nulos_df) > 0:
        print(nulos_df)
    else:
        print("Nenhum valor nulo!")

    print("\n--- Estatísticas descritivas (variáveis principais) ---")
    colunas_principais = ['Consumo_Carbapenemes', 'Total_Urgencias',
                          'Total_Consultas', 'Populacao_Regiao']
    colunas_existentes = [col for col in colunas_principais if col in df_final.columns]
    print(df_final[colunas_existentes].describe())

    print("\n--- Amostra do dataset (primeiras 10 linhas) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_final.head(10))

    print("\n" + "="*70)
    print("✓ Processo concluído com sucesso!")
    print("="*70)

if __name__ == "__main__":
    main()
