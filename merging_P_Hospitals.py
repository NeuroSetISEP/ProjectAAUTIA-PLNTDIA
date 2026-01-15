import pandas as pd

def merge_hospital_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file, sep=';')

    # Define mappings for hospitals with 'P' to their canonical names
    # Based on the pattern observed in the file
    name_mapping = {
        'Hospital Espirito Santo P Evora': 'Hospital Espirito Santo Evora',
        'Instituto Portugues Oncologia P Lisboa': 'Instituto Portugues Oncologia Lisboa',
        'Hospital Santa Maria Maior P Barcelos': 'Hospital Santa Maria Maior',
        'Instituto Portugues Oncologia P Porto': 'Instituto Portugues Oncologia Porto',
        'Instituto Portugues Oncologia P Coimbra': 'Instituto Portugues Oncologia Coimbra',
        # Add other mappings if similar patterns exist for other hospitals
    }

    # Apply the mapping to the 'Instituicao' column
    # If the name is not in the mapping, it keeps the original name
    df['Instituicao'] = df['Instituicao'].replace(name_mapping)

    # Identify columns to group by (Identifiers)
    group_cols = ['Instituicao', 'Regiao', 'Periodo', 'Ano', 'Mes']

    # Identify all other columns as value columns
    # We want to ensure they are numeric and included in the aggregation
    value_cols = [col for col in df.columns if col not in group_cols]

    # Convert value columns to numeric, coercing errors to NaN, then filling with 0
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Define aggregation dictionary
    agg_dict = {}
    for col in value_cols:
        if col == 'Populacao_Regiao':
            agg_dict[col] = 'max'  # Population should be max (constant for the region)
        else:
            agg_dict[col] = 'sum'  # Other metrics should be summed

    # Group by identifiers and aggregate
    df_merged = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Save the merged dataset
    df_merged.to_csv(output_file, sep=';', index=False)
    print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    input_csv = '/home/Projetos/ProjectAAUTIA-PLNTDIA/dataset_forecast_preparado.csv'
    output_csv = '/home/Projetos/ProjectAAUTIA-PLNTDIA/dataset_forecast_merged.csv'
    merge_hospital_data(input_csv, output_csv)
