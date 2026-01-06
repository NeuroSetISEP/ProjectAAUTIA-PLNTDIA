import pandas as pd
import numpy as np
import pygad

# 1. Configuração dos Dados
# Carrega o dataset (usando o delimitador ';' como no seu ficheiro)
df = pd.read_csv('dataset_medicamentos_por_regiao.csv', sep=';')
df = df.fillna(0)

# --- Inputs do Utilizador ---
print("="*40)
print("How much medicmentos total para distribuir: ")
TOTAL_STOCK = int(input())

print("Para que mes calcular (1-12): ")
PLANNING_MONTH = int(input())
print("="*40)

# Obter a lista de hospitais únicos
hospitals = df['Instituicao'].unique()
num_hospitals = len(hospitals)

# Pré-cálculo de médias históricas para acelerar o Algoritmo Genético
historical_data = []
for hosp in hospitals:
    hosp_df = df[df['Instituicao'] == hosp]
    
    # Regra: Consumo Sazonal (Média histórica do mês escolhido)
    seasonal_avg = hosp_df[hosp_df['Mes'] == PLANNING_MONTH]['Consumo_Carbapenemes'].mean()
    if np.isnan(seasonal_avg) or seasonal_avg == 0: 
        seasonal_avg = hosp_df['Consumo_Carbapenemes'].mean()
    
    # Regra: Prioridade por Urgência (Obstetrícia e Geral com pesos maiores)
    urgency_weight = (hosp_df['Urgencias_Obstetricia'].mean() * 1.5) + (hosp_df['Urgencias_Geral'].mean() * 1.2)
    
    # Regra: Escala populacional
    pop = hosp_df['Populacao_Regiao'].iloc[0] if not hosp_df.empty else 0
    
    historical_data.append({
        'seasonal_avg': seasonal_avg,
        'urgency_weight': urgency_weight,
        'pop': pop
    })

# --- Função de Fitness ---
def fitness_func(ga_instance, solution, solution_idx):
    # 'solution' é um array com os valores sugeridos para cada hospital
    if np.sum(solution) == 0: return -99999
    
    # Normalização: Garante que a soma testada corresponde ao TOTAL_STOCK
    allocation = (solution / np.sum(solution)) * TOTAL_STOCK
    
    fitness = 0
    for i in range(num_hospitals):
        data = historical_data[i]
        
        # Objetivo: A alocação deve ser proporcional à Necessidade Sazonal + Peso de Urgência
        target_need = data['seasonal_avg'] + (data['urgency_weight'] * 0.1)
        
        # O algoritmo tenta minimizar a diferença (erro) entre o alocado e o ideal
        error = abs(allocation[i] - target_need)
        fitness -= error 

    return fitness

# --- Execução do GA ---
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=num_hospitals,
    init_range_low=1000,
    init_range_high=50000,
    mutation_percent_genes=10
)

print(f"Treinando GA para distribuir {TOTAL_STOCK:,} unidades no mês {PLANNING_MONTH}...")
ga_instance.run()

# --- Resultados ---
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Normalização final para garantir que a soma é exatamente o stock total
final_allocation = (solution / np.sum(solution)) * TOTAL_STOCK

# CÁLCULO DA PERCENTAGEM (Nova parte solicitada)
percentages = (final_allocation / TOTAL_STOCK) * 100

# Criar o DataFrame de resultados para o CSV
results_df = pd.DataFrame({
    'Hospital': hospitals,
    'Allocated_Units': np.round(final_allocation, 2),
    'Percentage (%)': np.round(percentages, 4) # Adicionada a percentagem
})

# Guardar em CSV
results_df.to_csv('optimized_distribution.csv', index=False)

print("\n" + "="*40)
print("DISTRIBUIÇÃO OPTIMIZADA")
print(f"Total Distribuído: {TOTAL_STOCK:,}")
print("="*40)

print("\nTop 5 Hospitais com maior alocação:")
print(results_df.sort_values(by='Allocated_Units', ascending=False).head())

print("\nO ficheiro 'optimized_distribution.csv' foi gerado com sucesso.")

# Verificação final no terminal
print(f"\nVerificação: Soma Total = {results_df['Allocated_Units'].sum():,.2f}")
print(f"Verificação: Soma Percentagem = {results_df['Percentage (%)'].sum():.2f}%")