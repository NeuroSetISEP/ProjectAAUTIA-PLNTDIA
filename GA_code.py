import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygad
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Configurações de visualização e alertas
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ==============================================================================
# MÓDULO 1: PREVISÃO AVANÇADA (AutoML + Features de Tendência)
# ==============================================================================
class CarbapenemesPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.le_regiao = LabelEncoder()
        self.le_instituicao = LabelEncoder()

    def load_and_prepare_data(self):
        print(f"\n📂 Carregando dataset: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Arquivo {self.data_path} não encontrado!")

        self.df = pd.read_csv(self.data_path, sep=';')

        # Manter todos os hospitais (preencher nulos em vez de filtrar)
        self.df['Consumo_Carbapenemes'] = self.df['Consumo_Carbapenemes'].fillna(0)

        print(f"✅ Sucesso: {self.df['Instituicao'].nunique()} hospitais detectados.")
        print(f"✅ Registos Totais: {self.df.shape[0]}")

    def engineer_features(self):
        print("🛠️  Executando Engenharia de Features Avançada...")
        df_model = self.df.copy()

        # Encodings de Categorias
        df_model['Regiao_Encoded'] = self.le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = self.le_instituicao.fit_transform(df_model['Instituicao'])

        # Ciclos Sazonais (Seno/Cosseno do Mês)
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        # Lista Completa de Features (Básicas + As sugeridas pelo Professor)
        self.feature_names = [
            'Ano', 'Mes', 'Mes_Sin', 'Mes_Cos',
            'Regiao_Encoded', 'Instituicao_Encoded',
            'valor_base_sazonal', 'media_3m', 'media_6m',
            'tendencia_mom', 'tendencia_yoy', 'indice_sazonal',
            'forecast_hibrido', 'variacao_prevista_pct'
        ]

        # Tratamento de Nulos para as colunas de tendência
        X = df_model[self.feature_names].fillna(0)
        y = df_model['Consumo_Carbapenemes']

        return X, y

    def train_auto_ml(self, X, y):
        print("🤖 AutoML: Avaliando modelos para previsão de alta precisão...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge Regression': Ridge()
        }

        best_score = -np.inf
        winner_name = ""

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            print(f"   🔹 {name:<20} | R² Score: {score:.4f}")

            if score > best_score:
                best_score = score
                self.best_model = model
                winner_name = name

        print(f"🏆 Modelo Vencedor: {winner_name} (R²: {best_score:.2%})")
        return self.best_model

    def predict_month(self, month, year):
        # Pegar o estado mais recente de cada instituição
        latest_data = self.df.sort_values('Periodo').groupby('Instituicao').tail(1).copy()

        latest_data['Mes'] = month
        latest_data['Ano'] = year
        latest_data['Mes_Sin'] = np.sin(2 * np.pi * month / 12)
        latest_data['Mes_Cos'] = np.cos(2 * np.pi * month / 12)

        latest_data['Regiao_Encoded'] = self.le_regiao.transform(latest_data['Regiao'])
        latest_data['Instituicao_Encoded'] = self.le_instituicao.transform(latest_data['Instituicao'])

        X_future = latest_data[self.feature_names].fillna(0)
        X_future_scaled = self.scaler.transform(X_future)

        predictions = np.maximum(self.best_model.predict(X_future_scaled), 0)
        return dict(zip(latest_data['Instituicao'], predictions))

# ==============================================================================
# MÓDULO 2: OTIMIZAÇÃO (Algoritmo Genético)
# ==============================================================================
class GeneticDistributor:
    def __init__(self, demand_dict, total_stock):
        self.demand_dict = demand_dict
        self.hospitals = list(demand_dict.keys())
        self.targets = list(demand_dict.values())
        self.total_stock = total_stock

    def fitness_func(self, ga_instance, solution, solution_idx):
        if np.sum(solution) == 0: return -99999
        factor = self.total_stock / np.sum(solution)
        allocation = solution * factor

        penalty = 0
        for i in range(len(allocation)):
            needed = self.targets[i]
            given = allocation[i]
            if given < needed:
                penalty += (needed - given) ** 2  # Penalidade grave para falta
            else:
                penalty += (given - needed) * 0.1 # Penalidade leve para excesso

        return 1.0 / (penalty + 1.0)

    def run(self):
        ga_instance = pygad.GA(
            num_generations=200,
            num_parents_mating=10,
            fitness_func=self.fitness_func,
            sol_per_pop=40,
            num_genes=len(self.hospitals),
            init_range_low=10,
            init_range_high=1000,
            mutation_percent_genes=15,
            suppress_warnings=True
        )
        ga_instance.run()
        solution, _, _ = ga_instance.best_solution()
        factor = self.total_stock / np.sum(solution)
        return np.round(solution * factor, 0)

# ==============================================================================
# MÓDULO 3: EXECUÇÃO PRINCIPAL (Menu Multi-Input + Relatório)
# ==============================================================================
def main():
    print("="*85)
    print("      SNS AI: SISTEMA INTEGRADO DE PREVISÃO SENSÍVEL À TENDÊNCIA (97 HOSPITAIS)")
    print("="*85)

    file_path = 'dataset_forecast_preparado.csv'
    ml_system = CarbapenemesPredictionModel(file_path)
    ml_system.load_and_prepare_data()
    X, y = ml_system.engineer_features()
    ml_system.train_auto_ml(X, y)

    print("\n" + "-"*30)
    print("CONFIGURAÇÃO DE PERÍODO")
    print("1. Inserir meses específicos (ex: 1, 2, 3)")
    print("2. Inserir Trimestre (Quarter 1-4)")
    print("3. Ano Completo (1-12)")
    print("-"*30)

    opcao = input("Escolha a opção (1-3): ")
    ano_alvo = int(input("Informe o Ano (Ex: 2025): "))

    months_list = []
    if opcao == '1':
        months_list = [int(x.strip()) for x in input("Meses (separados por vírgula): ").split(',')]
    elif opcao == '2':
        q = int(input("Qual o Trimestre? (1-4): "))
        months_list = list(range((q-1)*3 + 1, q*3 + 1))
    elif opcao == '3':
        months_list = list(range(1, 13))

    perc_stock = float(input("\n👉 % de stock disponível para cada mês (ex: 0.8 para 80%): "))

    all_rows = []

    # Processamento em Loop para gerar resultados mês a mês
    for mes in months_list:
        label = f"{ano_alvo}-{mes:02d}"
        print(f"\n⏳ Processando {label}...")

        # 1. Prever
        demand_map = ml_system.predict_month(mes, ano_alvo)
        total_needed = sum(demand_map.values())

        # 2. Otimizar
        stock_m = int(total_needed * perc_stock)
        optimizer = GeneticDistributor(demand_map, stock_m)
        allocation = optimizer.run()

        # 3. Armazenar
        total_dist = np.sum(allocation)
        for i, inst in enumerate(demand_map.keys()):
            amt = allocation[i]
            all_rows.append({
                'periodo': label,
                'Instituicao': inst,
                'Amount': int(amt),
                'Percentage': round((amt/total_dist*100), 4) if total_dist > 0 else 0
            })

    # Exportação e Visualização
    df_final = pd.DataFrame(all_rows)
    csv_name = f"distribuicao_completa_{ano_alvo}.csv"
    df_final.to_csv(csv_name, index=False, sep=';')

    print("\n" + "="*85)
    print(f"{'PERÍODO':<10} | {'HOSPITAL (Top 10 p/ Mês)':<40} | {'ALOCADO':<10} | {'% TOTAL'}")
    print("-" * 85)

    # Mostrar um resumo visual (Top 10 do primeiro mês processado)
    first_month = f"{ano_alvo}-{months_list[0]:02d}"
    summary = df_final[df_final['periodo'] == first_month].sort_values('Amount', ascending=False).head(10)

    for _, row in summary.iterrows():
        print(f"{row['periodo']:<10} | {row['Instituicao'][:40]:<40} | {row['Amount']:<10} | {row['Percentage']}%")

    print("-" * 85)
    print(f"✅ SUCESSO! Relatório consolidado com {len(df_final)} linhas gerado em: {csv_name}")
    print(f"📈 Foram processados {len(months_list)} meses para {ml_system.df['Instituicao'].nunique()} hospitais.")

if __name__ == "__main__":
    main()