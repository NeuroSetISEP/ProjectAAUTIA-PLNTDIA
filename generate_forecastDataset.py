import pandas as pd
import numpy as np

def generate_forecasting_dataset(file_path):
    # 1. Load data
    df = pd.read_csv(file_path, sep=';')

    # Ensure Periodo is datetime and sort for time-series calculations
    df['Periodo'] = pd.to_datetime(df['Periodo'])
    df = df.sort_values(['Instituicao', 'Periodo'])

    # Target column
    target = 'Consumo_Carbapenemes'

    print("Gerando métricas temporais...")

    # 2. Base Sazonal (Value from the same month previous year)
    df['valor_base_sazonal'] = df.groupby('Instituicao')[target].shift(12)

    # 3. Moving Averages
    df['media_3m'] = df.groupby('Instituicao')[target].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['media_6m'] = df.groupby('Instituicao')[target].transform(lambda x: x.rolling(window=6, min_periods=1).mean())

    # 4. Trends (Variação)
    df['prev_month'] = df.groupby('Instituicao')[target].shift(1)
    df['tendencia_mom'] = (df[target] - df['prev_month']) / df['prev_month']
    df['tendencia_yoy'] = (df[target] - df['valor_base_sazonal']) / df['valor_base_sazonal']

    # 5. Seasonal Index
    annual_avg = df.groupby(['Instituicao', 'Ano'])[target].transform('mean')
    df['indice_sazonal'] = df[target] / annual_avg

    # 6. Forecast Híbrido (Weighted combination)
    df['forecast_hibrido'] = (
        (df['media_3m'] * 0.5) +
        (df['media_6m'] * 0.3) +
        (df.groupby('Instituicao')['valor_base_sazonal'].shift(1).fillna(0) * 0.2)
    )

    # 7. Variação Prevista Pct
    df['variacao_prevista_pct'] = (df['forecast_hibrido'] / df['prev_month']) - 1

    # Replace infinite values and NaNs with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- NEW: ADD EXTRA COLUMNS FROM ORIGINAL DATASET ---
    # We include these so they are available for the Genetic Algorithm Fitness Function
    extra_cols = [
        'Populacao_Regiao',
        'Total_Urgencias',
        'Urgencias_Geral', 'Urgencias_Pediatricas',
        'Urgencias_Obstetricia', 'Urgencias_Psiquiatrica',
        'Total_Consultas'
    ]

    # Combine all columns to keep
    cols_to_keep = [
        'Instituicao', 'Regiao', 'Periodo', 'Ano', 'Mes', target
    ] + extra_cols + [
        'valor_base_sazonal', 'media_3m', 'media_6m', 'tendencia_mom',
        'tendencia_yoy', 'indice_sazonal', 'forecast_hibrido', 'variacao_prevista_pct'
    ]

    new_df = df[cols_to_keep]

    # Save the new dataset
    output_name = 'dataset_forecast_preparado.csv'
    new_df.to_csv(output_name, index=False, sep=';')
    print(f"✅ Novo dataset criado: {output_name}")
    return new_df

# Run the function
new_dataset = generate_forecasting_dataset('dataset_medicamentos_por_regiao.csv')