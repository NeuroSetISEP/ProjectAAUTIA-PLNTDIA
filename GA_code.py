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

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ==============================================================================
# MÓDULO 1: PREVISÃO AVANÇADA
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
        self.df = pd.read_csv(self.data_path, sep=';')
        self.df['Consumo_Carbapenemes'] = self.df['Consumo_Carbapenemes'].fillna(0)
        print(f"✅ Sucesso: {self.df['Instituicao'].nunique()} hospitais detectados.")

    def engineer_features(self):
        print("🛠️  Executando Engenharia de Features Avançada...")
        df_model = self.df.copy()
        df_model['Regiao_Encoded'] = self.le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = self.le_instituicao.fit_transform(df_model['Instituicao'])
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        self.feature_names = [
            'Ano', 'Mes', 'Mes_Sin', 'Mes_Cos', 'Regiao_Encoded', 'Instituicao_Encoded',
            'valor_base_sazonal', 'media_3m', 'media_6m', 'tendencia_mom',
            'tendencia_yoy', 'indice_sazonal', 'forecast_hibrido', 'variacao_prevista_pct'
        ]
        X = df_model[self.feature_names].fillna(0)
        y = df_model['Consumo_Carbapenemes']
        return X, y

    def train_auto_ml(self, X, y):
        print("🤖 AutoML: Treinando modelos...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge': Ridge()
        }

        best_score = -np.inf
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = r2_score(y_test, model.predict(X_test_scaled))
            if score > best_score:
                best_score = score
                self.best_model = model
        print(f"🏆 Melhor Modelo: {type(self.best_model).__name__} (R²: {best_score:.2%})")

    def predict_month_with_context(self, month, year):
        # Captura os dados mais recentes incluindo as novas colunas para o GA
        latest_data = self.df.sort_values('Periodo').groupby('Instituicao').tail(1).copy()
        latest_data['Mes'] = month
        latest_data['Ano'] = year
        latest_data['Mes_Sin'] = np.sin(2 * np.pi * month / 12)
        latest_data['Mes_Cos'] = np.cos(2 * np.pi * month / 12)
        latest_data['Regiao_Encoded'] = self.le_regiao.transform(latest_data['Regiao'])
        latest_data['Instituicao_Encoded'] = self.le_instituicao.transform(latest_data['Instituicao'])

        X_future_scaled = self.scaler.transform(latest_data[self.feature_names].fillna(0))
        predictions = np.maximum(self.best_model.predict(X_future_scaled), 0)

        # Criamos um dicionário que contém a previsão + dados contextuais
        context_map = {}
        for i, row in latest_data.reset_index().iterrows():
            context_map[row['Instituicao']] = {
                'pred': predictions[i],
                'pop': row['Populacao_Regiao'],
                'consultas': row['Total_Consultas'],
                'urg_geral': row['Urgencias_Geral'],
                'urg_ped': row['Urgencias_Pediatricas'],
                'urg_obs': row['Urgencias_Obstetricia'],
                'urg_psi': row['Urgencias_Psiquiatrica']
            }
        return context_map

# ==============================================================================
# MÓDULO 2: OTIMIZAÇÃO (Algoritmo Genético com Regras Sociais)
# ==============================================================================
class GeneticDistributor:
    def __init__(self, context_dict, total_stock):
        self.context_dict = context_dict
        self.hospitals = list(context_dict.keys())
        self.total_stock = total_stock
        self.priority_weights = self._calculate_priority_weights()

    def _calculate_priority_weights(self):
        """ Calcula o peso de prioridade (0.0 a 1.0) para cada hospital baseado nas novas regras """
        weights = {}

        # Extrair valores para normalização (evitar que escalas diferentes quebrem o GA)
        all_pop = [v['pop'] for v in self.context_dict.values()]
        all_cons = [v['consultas'] for v in self.context_dict.values()]

        max_pop = max(all_pop) if max(all_pop) > 0 else 1
        max_cons = max(all_cons) if max(all_cons) > 0 else 1

        for h, data in self.context_dict.items():
            # 1. Regra Urgências (Pesos definidos por você)
            score_urg = (data['urg_geral'] * 0.5 +
                         data['urg_ped'] * 0.25 +
                         data['urg_obs'] * 0.2 +
                         data['urg_psi'] * 0.05)

            # 2. Regra Consultas (Normalizada)
            score_cons = data['consultas'] / max_cons

            # 3. Regra População (Normalizada)
            score_pop = data['pop'] / max_pop

            # Score Final de Prioridade (Média ponderada das regras)
            # Damos 40% peso para Urgências, 30% Consultas e 30% População
            total_priority = (score_urg * 0.4) + (score_cons * 0.3) + (score_pop * 0.3)
            weights[h] = total_priority

        return weights

    def fitness_func(self, ga_instance, solution, solution_idx):
        if np.sum(solution) == 0: return -99999
        allocation = solution * (self.total_stock / np.sum(solution))

        total_penalty = 0
        for i, h in enumerate(self.hospitals):
            needed = self.context_dict[h]['pred']
            given = allocation[i]
            priority = self.priority_weights[h] # O multiplicador de "importância"

            if given < needed:
                # PENALIDADE DE FALTA: Multiplicada pela prioridade social
                # Se o hospital é prioritário, a falta dói muito mais no score
                total_penalty += ((needed - given) ** 2) * (1 + priority)
            else:
                # PENALIDADE DE EXCESSO: Leve
                total_penalty += (given - needed) * 0.1

        return 1.0 / (total_penalty + 1.0)

    def run(self):
        ga_instance = pygad.GA(
            num_generations=250, num_parents_mating=10, fitness_func=self.fitness_func,
            sol_per_pop=50, num_genes=len(self.hospitals), init_range_low=1,
            init_range_high=500, mutation_percent_genes=15, suppress_warnings=True
        )
        ga_instance.run()
        solution, _, _ = ga_instance.best_solution()
        return np.round(solution * (self.total_stock / np.sum(solution)), 0)

# ==============================================================================
# MÓDULO 3: EXECUÇÃO
# ==============================================================================
def main():
    print("="*85)
    print("      SNS AI: DISTRIBUIÇÃO OTIMIZADA POR PRIORIDADE SOCIAL E CLÍNICA")
    print("="*85)

    ml_system = CarbapenemesPredictionModel('dataset_forecast_preparado.csv')
    ml_system.load_and_prepare_data()
    X, y = ml_system.engineer_features()
    ml_system.train_auto_ml(X, y)

    # Seleção de Período
    print("\n1. Meses Específicos | 2. Trimestre | 3. Ano Completo")
    opcao = input("Escolha: ")
    ano = int(input("Ano: "))
    months = []
    if opcao == '1': months = [int(x) for x in input("Meses: ").split(',')]
    elif opcao == '2':
        q = int(input("Q (1-4): "))
        months = list(range((q-1)*3+1, q*3+1))
    else: months = list(range(1, 13))

    perc = float(input("👉 % de stock disponível (ex: 0.7 para 70%): "))
    all_results = []

    for m in months:
        label = f"{ano}-{m:02d}"
        print(f"⏳ Otimizando {label} com base em Consultas, Urgências e População...")

        context_map = ml_system.predict_month_with_context(m, ano)
        total_needed = sum([v['pred'] for v in context_map.values()])
        stock_m = int(total_needed * perc)

        optimizer = GeneticDistributor(context_map, stock_m)
        allocation = optimizer.run()

        for i, h in enumerate(optimizer.hospitals):
            all_results.append({
                'periodo': label, 'Instituicao': h, 'Amount': int(allocation[i]),
                'Priority_Weight': round(optimizer.priority_weights[h], 4)
            })

    df_final = pd.DataFrame(all_results)
    df_final.to_csv(f"distribuicao_priorizada_{ano}.csv", index=False, sep=';')
    print(f"\n✅ Concluído! Relatório salvo. Top 5 por Prioridade no primeiro mês:")
    print(df_final.sort_values(['periodo', 'Priority_Weight'], ascending=False).head(5))

if __name__ == "__main__":
    main()