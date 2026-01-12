import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygad
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ==============================================================================
# MÓDULO 1: PREVISÃO AVANÇADA (A classe do teu colega, ligeiramente adaptada)
# ==============================================================================
class CarbapenemesPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        # Encoders globais para poder usar na previsão futura
        self.le_regiao = LabelEncoder()
        self.le_instituicao = LabelEncoder()

    def load_and_prepare_data(self):
        print("📊 Carregando dados...")
        self.df = pd.read_csv(self.data_path, sep=';')

        # Filtro de qualidade (remover anos sem consumo real)
        if 'Consumo_Carbapenemes' in self.df.columns:
             self.df = self.df[self.df['Consumo_Carbapenemes'] > 0].copy()

        print(f"   Dados limpos: {self.df.shape[0]} registos.")

    def engineer_features(self):
        print("\n🔧 Engenharia de features...")
        df_model = self.df.copy()

        # Encoding
        df_model['Regiao_Encoded'] = self.le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = self.le_instituicao.fit_transform(df_model['Instituicao'])

        # Features Temporais
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        # Features de Lag (Inteligência Temporal Simples)
        # Assumimos que para treino temos colunas completas, para previsão futura teremos de construir
        feature_columns = [
            'Ano', 'Mes', 'Mes_Sin', 'Mes_Cos',
            'Regiao_Encoded', 'Instituicao_Encoded',
            'Populacao_Regiao', 'Total_Urgencias', 'Total_Consultas'
        ]

        # Tratamento de Nulos
        df_model = df_model.fillna(0)

        self.feature_names = feature_columns
        X = df_model[feature_columns]
        y = df_model['Consumo_Carbapenemes']

        return X, y

    def train_auto_ml(self, X, y):
        print("\n🤖 AutoML: A testar múltiplos modelos...")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge': Ridge()
        }

        best_score = -np.inf
        best_name = ""

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"   Model: {name:<20} | R² Teste: {score:.2%}")

            if score > best_score:
                best_score = score
                self.best_model = model
                best_name = name

        print(f"\n🏆 Vencedor: {best_name} (R²: {best_score:.2%})")
        return self.best_model

    def predict_demand(self, month, year):
        """ Gera previsões para todos os hospitais para um mês específico """
        print(f"\n🔮 A gerar previsões para {month}/{year}...")

        # Obter lista única de hospitais e os seus dados mais recentes (estáticos)
        latest_data = self.df.groupby('Instituicao').tail(1).copy()

        # Atualizar tempo
        latest_data['Mes'] = month
        latest_data['Ano'] = year
        latest_data['Mes_Sin'] = np.sin(2 * np.pi * month / 12)
        latest_data['Mes_Cos'] = np.cos(2 * np.pi * month / 12)

        # Recriar Encodings (seguro)
        try:
            latest_data['Regiao_Encoded'] = self.le_regiao.transform(latest_data['Regiao'])
            latest_data['Instituicao_Encoded'] = self.le_instituicao.transform(latest_data['Instituicao'])
        except:
            # Fallback se houver algum hospital novo
            latest_data['Regiao_Encoded'] = 0
            latest_data['Instituicao_Encoded'] = 0

        # Preparar X
        X_future = latest_data[self.feature_names].fillna(0)
        X_future_scaled = self.scaler.transform(X_future)

        # Prever
        predictions = self.best_model.predict(X_future_scaled)

        # Garantir que não há valores negativos
        predictions = np.maximum(predictions, 0)

        return dict(zip(latest_data['Instituicao'], predictions))

# ==============================================================================
# MÓDULO 2: OTIMIZAÇÃO COM ALGORITMO GENÉTICO (A nossa parte de Distribuição)
# ==============================================================================
class GeneticDistributor:
    def __init__(self, demand_dict, total_stock):
        self.demand_dict = demand_dict
        self.hospitals = list(demand_dict.keys())
        self.targets = list(demand_dict.values())
        self.total_stock = total_stock

    def fitness_func(self, ga_instance, solution, solution_idx):
        # Solução vazia é inválida
        if np.sum(solution) == 0: return -99999

        # Normalização forçada (Gene -> Stock Real)
        # O AG gera números, nós transformamos em % do stock total
        factor = self.total_stock / np.sum(solution)
        allocation = solution * factor

        penalty = 0
        # Calcular a "Dor" de cada hospital
        for i in range(len(allocation)):
            needed = self.targets[i]
            given = allocation[i]

            if given < needed:
                # Penalidade Quadrática: Faltar stock é MUITO grave
                # Ex: Faltar 100 é 100x pior que faltar 10 (100^2 vs 10^2)
                penalty += (needed - given) ** 2
            else:
                # Penalidade por Excesso (Leve): Desperdício
                penalty += (given - needed) * 0.1

        # Fitness = Inverso da Penalidade (Quanto menor a penalidade, maior o fitness)
        return 1.0 / (penalty + 1.0)

    def run(self):
        print(f"🧬 A iniciar Algoritmo Genético para distribuir {self.total_stock} unidades...")

        ga_instance = pygad.GA(
            num_generations=100,
            num_parents_mating=5,
            fitness_func=self.fitness_func,
            sol_per_pop=20,
            num_genes=len(self.hospitals),
            init_range_low=10,
            init_range_high=1000,
            mutation_percent_genes=15,
            keep_parents=2,
            suppress_warnings=True
        )

        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()

        # Converter solução abstrata para números reais
        factor = self.total_stock / np.sum(solution)
        final_allocation = np.round(solution * factor, 0)

        return final_allocation

# ==============================================================================
# MÓDULO 3: EXECUÇÃO INTEGRADA (MAIN)
# ==============================================================================
def main():
    print("="*60)
    print("🏥 SISTEMA INTEGRADO DE PREVISÃO E DISTRIBUIÇÃO SNS")
    print("="*60)

    file_path = 'dataset_medicamentos_por_regiao.csv'

    # 1. Pipeline de Machine Learning
    ml_system = CarbapenemesPredictionModel(file_path)
    ml_system.load_and_prepare_data()
    X, y = ml_system.engineer_features()
    ml_system.train_auto_ml(X, y)

    # 2. Previsão de Cenário Futuro
    mes_alvo = 11
    ano_alvo = 2025
    demand_map = ml_system.predict_demand(mes_alvo, ano_alvo)

    total_needed = sum(demand_map.values())
    print(f"\n🔮 Necessidade Nacional Prevista (Nov 2025): {int(total_needed):,} unidades")

    # 3. Definição de Cenário de Escassez
    # Vamos simular que só temos 80% do necessário
    stock_disponivel = int(total_needed * 0.80)
    print(f"⚠️  Stock Real Disponível (Cenário de Crise): {stock_disponivel:,} unidades")

    # 4. Pipeline de Otimização (Genetic Algorithm)
    optimizer = GeneticDistributor(demand_map, stock_disponivel)
    final_allocation = optimizer.run()

    # 5. Relatório Final
    print("\n" + "="*60)
    print(f"{'HOSPITAL':<40} | {'NECESSÁRIO':<10} | {'ALOCADO':<10} | {'DIFERENÇA'}")
    print("-" * 75)

    results = []
    hospitals = list(demand_map.keys())
    needs = list(demand_map.values())

    # Ordenar por maior défice para mostrar os casos críticos primeiro
    diffs = final_allocation - needs
    sorted_indices = np.argsort(diffs) # Indices dos que sofreram mais cortes

    for i in sorted_indices[:10]: # Mostrar Top 10 Críticos
        h = hospitals[i]
        n = int(needs[i])
        a = int(final_allocation[i])
        d = a - n
        status = "CRÍTICO" if d < -100 else "Ajustado"
        print(f"{h[:40]:<40} | {n:<10} | {a:<10} | {d}")

    print("-" * 75)
    print(f"Total Alocado: {int(sum(final_allocation))} / {stock_disponivel}")
    print("Conclusão: O sistema utilizou ML para prever a procura e AG para minimizar o impacto da rutura.")

if __name__ == "__main__":
    main()