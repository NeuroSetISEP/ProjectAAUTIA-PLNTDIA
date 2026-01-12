"""
Sistema Integrado de Previsão e Otimização do Consumo de Carbapenemes
=======================================================================

ARQUITETURA DO SISTEMA:
----------------------
1. MOTOR DE PREVISÃO (Machine Learning):
   - Random Forest / Gradient Boosting
   - Prevê o consumo futuro de carbapenemes por instituição e região

2. MOTOR DE OTIMIZAÇÃO (Algoritmo Genético):
   - Usa as previsões do ML como input
   - Otimiza a distribuição de stock escasso
   - Considera múltiplos objetivos: necessidade prevista, urgências, população

3. INTERFACE DE DECISÃO:
   - Permite ao utilizador definir cenários
   - Compara diferentes estratégias de alocação
   - Gera relatórios e visualizações

Autores: [Teu Nome]
Data: Janeiro 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pygad
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class MLPredictionEngine:
    """
    Motor de Previsão usando Machine Learning
    Treina modelos para prever o consumo futuro de carbapenemes
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_and_prepare_data(self):
        """Carrega e prepara os dados"""
        print("📊 [ML ENGINE] Carregando dados históricos...")
        self.df = pd.read_csv(self.data_path, sep=';', decimal='.')
        print(f"   ✅ {len(self.df)} registos carregados ({self.df['Periodo'].min()} a {self.df['Periodo'].max()})")

    def engineer_features(self):
        """Cria features para o modelo ML"""
        from sklearn.preprocessing import LabelEncoder

        df_model = self.df.copy().dropna(subset=['Consumo_Carbapenemes'])

        # Encoding
        le_regiao = LabelEncoder()
        le_instituicao = LabelEncoder()
        df_model['Regiao_Encoded'] = le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = le_instituicao.fit_transform(df_model['Instituicao'])

        # Features temporais
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        # Features derivadas
        df_model['Urgencias_Por_Pop'] = df_model['Total_Urgencias'] / df_model['Populacao_Regiao']
        df_model['Consultas_Por_Pop'] = df_model['Total_Consultas'] / df_model['Populacao_Regiao']
        df_model['Ratio_Primeiras_Consultas'] = df_model['Primeiras_Consultas'] / (df_model['Total_Consultas'] + 1)
        df_model['Ratio_Urgencias_Geral'] = df_model['Urgencias_Geral'] / (df_model['Total_Urgencias'] + 1)

        # Selecionar features
        self.feature_names = [
            'Ano', 'Mes', 'Trimestre', 'Semestre',
            'Mes_Sin', 'Mes_Cos',
            'Regiao_Encoded', 'Instituicao_Encoded',
            'Populacao_Regiao', 'Num_Municipios',
            'Total_Urgencias', 'Urgencias_Geral', 'Urgencias_Pediatricas',
            'Total_Urgencias_Per_Capita',
            'Total_Consultas', 'Primeiras_Consultas', 'Consultas_Subsequentes',
            'Total_Consultas_Per_Capita',
            'Urgencias_Por_Pop', 'Consultas_Por_Pop',
            'Ratio_Primeiras_Consultas', 'Ratio_Urgencias_Geral',
            'Consumo_Outros_Antibioticos', 'Consumo_Total_Antibioticos'
        ]

        X = df_model[self.feature_names].fillna(df_model[self.feature_names].median())
        y = df_model['Consumo_Carbapenemes']

        return X, y, df_model

    def train_model(self, X, y):
        """Treina o modelo de ML"""
        print("\n🤖 [ML ENGINE] Treinando modelo de previsão...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Treinar Gradient Boosting (geralmente o melhor)
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Avaliar
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"   ✅ Modelo treinado: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        return r2, rmse, mae

    def predict_future_consumption(self, target_month, target_year=2024):
        """
        Prevê o consumo futuro para cada instituição num determinado mês

        Returns:
            DataFrame com previsões por instituição
        """
        print(f"\n🔮 [ML ENGINE] Gerando previsões para {target_month}/{target_year}...")

        # Obter dados das instituições
        latest_data = self.df.groupby('Instituicao').last().reset_index()

        predictions = []

        for _, row in latest_data.iterrows():
            # Criar features para o mês alvo
            features = {
                'Ano': target_year,
                'Mes': target_month,
                'Trimestre': (target_month - 1) // 3 + 1,
                'Semestre': 1 if target_month <= 6 else 2,
                'Mes_Sin': np.sin(2 * np.pi * target_month / 12),
                'Mes_Cos': np.cos(2 * np.pi * target_month / 12),
            }

            # Usar dados históricos da instituição
            for col in self.feature_names:
                if col not in features:
                    features[col] = row.get(col, 0)

            # Preparar para previsão
            X_pred = pd.DataFrame([features])[self.feature_names]
            X_pred = X_pred.fillna(X_pred.median())
            X_pred_scaled = self.scaler.transform(X_pred)

            # Fazer previsão
            predicted_consumption = self.model.predict(X_pred_scaled)[0]
            predicted_consumption = max(0, predicted_consumption)  # Não pode ser negativo

            predictions.append({
                'Instituicao': row['Instituicao'],
                'Regiao': row['Regiao'],
                'Populacao_Regiao': row['Populacao_Regiao'],
                'Total_Urgencias': row['Total_Urgencias'],
                'Urgencias_Geral': row['Urgencias_Geral'],
                'Consumo_Previsto': predicted_consumption
            })

        predictions_df = pd.DataFrame(predictions)
        print(f"   ✅ Previsões geradas para {len(predictions_df)} instituições")
        print(f"   📊 Consumo total previsto: {predictions_df['Consumo_Previsto'].sum():.2f} unidades")

        return predictions_df


class GAOptimizationEngine:
    """
    Motor de Otimização usando Algoritmo Genético
    Otimiza a distribuição de stock com base nas previsões do ML
    """

    def __init__(self, predictions_df, total_stock):
        self.predictions_df = predictions_df
        self.total_stock = total_stock
        self.hospitals = predictions_df['Instituicao'].values
        self.num_hospitals = len(self.hospitals)
        self.best_solution = None

    def calculate_weights(self):
        """Calcula os pesos para otimização multi-objetivo"""
        weights = []

        for _, row in self.predictions_df.iterrows():
            # Componente 1: Necessidade prevista pelo ML (peso: 50%)
            predicted_need = row['Consumo_Previsto']

            # Componente 2: Urgências (peso: 30%)
            urgency_factor = row['Urgencias_Geral'] * 0.3

            # Componente 3: População (peso: 20%)
            pop_factor = (row['Populacao_Regiao'] / 100000) * 0.2

            # Peso final normalizado
            total_weight = predicted_need + urgency_factor + pop_factor
            weights.append(total_weight)

        return np.array(weights)

    def fitness_function(self, ga_instance, solution, solution_idx):
        """
        Função de fitness multi-objetivo:
        1. Minimiza diferença entre alocação e necessidade prevista
        2. Penaliza desigualdades extremas
        3. Recompensa eficiência na distribuição
        """
        if np.sum(solution) == 0:
            return -999999

        # Normalizar para o stock total
        allocation = (solution / np.sum(solution)) * self.total_stock

        fitness = 0
        weights = self.calculate_weights()

        for i in range(self.num_hospitals):
            target_need = weights[i]

            # Objetivo 1: Minimizar erro de alocação (70% do fitness)
            error = abs(allocation[i] - target_need)
            fitness -= error * 0.7

            # Objetivo 2: Penalizar sub-alocação crítica (20% do fitness)
            if allocation[i] < target_need * 0.5:  # Se receber menos de 50% do necessário
                critical_penalty = (target_need * 0.5 - allocation[i]) * 2
                fitness -= critical_penalty * 0.2

            # Objetivo 3: Recompensar distribuição proporcional (10% do fitness)
            proportion = allocation[i] / (target_need + 1)
            if 0.8 <= proportion <= 1.2:  # Alocação dentro de ±20% do ideal
                fitness += 100 * 0.1

        # Penalizar variância excessiva (evitar concentração em poucos hospitais)
        variance_penalty = np.std(allocation) * 0.1
        fitness -= variance_penalty

        return fitness

    def optimize_distribution(self):
        """Executa o Algoritmo Genético para otimizar a distribuição"""
        print(f"\n🧬 [GA ENGINE] Otimizando distribuição de {self.total_stock:,} unidades...")
        print(f"   População: {self.num_hospitals} instituições")

        ga_instance = pygad.GA(
            num_generations=300,
            num_parents_mating=8,
            fitness_func=self.fitness_function,
            sol_per_pop=30,
            num_genes=self.num_hospitals,
            init_range_low=100,
            init_range_high=self.total_stock / self.num_hospitals * 2,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=15,
            random_seed=42
        )

        ga_instance.run()

        # Obter melhor solução
        solution, solution_fitness, _ = ga_instance.best_solution()
        self.best_solution = (solution / np.sum(solution)) * self.total_stock

        print(f"   ✅ Otimização concluída (Fitness: {solution_fitness:.2f})")

        return self.create_results_dataframe()

    def create_results_dataframe(self):
        """Cria DataFrame com os resultados da otimização"""
        results = self.predictions_df.copy()
        results['Alocacao_Otimizada'] = np.round(self.best_solution, 2)
        results['Percentagem (%)'] = np.round((self.best_solution / self.total_stock) * 100, 2)
        results['Diferenca_vs_Previsto'] = results['Alocacao_Otimizada'] - results['Consumo_Previsto']
        results['Taxa_Cobertura (%)'] = np.round(
            (results['Alocacao_Otimizada'] / results['Consumo_Previsto']) * 100, 2
        )

        return results.sort_values('Alocacao_Otimizada', ascending=False)


class IntegratedSystem:
    """
    Sistema Integrado que coordena ML e GA
    Interface principal para o utilizador
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.ml_engine = MLPredictionEngine(data_path)
        self.ga_engine = None
        self.predictions = None
        self.optimization_results = None

    def train_prediction_model(self):
        """Treina o motor de previsão"""
        self.ml_engine.load_and_prepare_data()
        X, y, _ = self.ml_engine.engineer_features()
        metrics = self.ml_engine.train_model(X, y)
        return metrics

    def run_full_pipeline(self, target_month, target_year, total_stock):
        """
        Executa o pipeline completo: Previsão → Otimização → Relatório

        Args:
            target_month: Mês para previsão (1-12)
            target_year: Ano para previsão
            total_stock: Stock total disponível para distribuir
        """
        print("\n" + "="*70)
        print("🏥 SISTEMA INTEGRADO DE PREVISÃO E OTIMIZAÇÃO")
        print("="*70)

        # Passo 1: Previsão com ML
        self.predictions = self.ml_engine.predict_future_consumption(target_month, target_year)

        # Passo 2: Otimização com GA
        self.ga_engine = GAOptimizationEngine(self.predictions, total_stock)
        self.optimization_results = self.ga_engine.optimize_distribution()

        # Passo 3: Análise de resultados
        self.analyze_results(target_month, target_year, total_stock)

        return self.optimization_results

    def analyze_results(self, target_month, target_year, total_stock):
        """Analisa e imprime resultados do sistema"""
        print("\n" + "="*70)
        print("📊 ANÁLISE DE RESULTADOS")
        print("="*70)

        df = self.optimization_results

        print(f"\n🎯 Cenário: {target_month}/{target_year} | Stock Disponível: {total_stock:,} unidades")
        print(f"\n📈 Necessidade vs Disponibilidade:")
        print(f"   Consumo Total Previsto (ML): {df['Consumo_Previsto'].sum():,.2f}")
        print(f"   Stock Disponível: {total_stock:,}")

        deficit = df['Consumo_Previsto'].sum() - total_stock
        if deficit > 0:
            print(f"   ⚠️  DÉFICE: {deficit:,.2f} unidades ({(deficit/df['Consumo_Previsto'].sum())*100:.1f}%)")
        else:
            print(f"   ✅ EXCEDENTE: {abs(deficit):,.2f} unidades")

        print(f"\n🏆 Top 5 Instituições (Maior Alocação):")
        print(df[['Instituicao', 'Regiao', 'Consumo_Previsto', 'Alocacao_Otimizada', 'Taxa_Cobertura (%)']].head(5).to_string(index=False))

        print(f"\n⚠️  Top 5 Instituições (Menor Taxa de Cobertura):")
        print(df[['Instituicao', 'Regiao', 'Consumo_Previsto', 'Alocacao_Otimizada', 'Taxa_Cobertura (%)']].nsmallest(5, 'Taxa_Cobertura (%)').to_string(index=False))

        print(f"\n📊 Estatísticas de Distribuição:")
        print(f"   Média de alocação: {df['Alocacao_Otimizada'].mean():,.2f}")
        print(f"   Desvio padrão: {df['Alocacao_Otimizada'].std():,.2f}")
        print(f"   Taxa de cobertura média: {df['Taxa_Cobertura (%)'].mean():.2f}%")
        print(f"   Instituições com >90% cobertura: {(df['Taxa_Cobertura (%)'] >= 90).sum()}")
        print(f"   Instituições com <70% cobertura: {(df['Taxa_Cobertura (%)'] < 70).sum()}")

    def generate_visualizations(self):
        """Gera visualizações completas dos resultados"""
        print("\n📊 Gerando visualizações...")

        df = self.optimization_results

        fig = plt.figure(figsize=(18, 12))

        # 1. Comparação Previsto vs Alocado (Top 15)
        ax1 = plt.subplot(2, 3, 1)
        top15 = df.head(15)
        x = np.arange(len(top15))
        width = 0.35
        ax1.barh(x - width/2, top15['Consumo_Previsto'], width, label='Previsto (ML)', alpha=0.8, color='steelblue')
        ax1.barh(x + width/2, top15['Alocacao_Otimizada'], width, label='Alocado (GA)', alpha=0.8, color='coral')
        ax1.set_yticks(x)
        ax1.set_yticklabels(top15['Instituicao'], fontsize=8)
        ax1.set_xlabel('Unidades')
        ax1.set_title('Top 15: Consumo Previsto vs Alocação Otimizada')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribuição por Região
        ax2 = plt.subplot(2, 3, 2)
        region_data = df.groupby('Regiao').agg({
            'Consumo_Previsto': 'sum',
            'Alocacao_Otimizada': 'sum'
        })
        region_data.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Distribuição por Região')
        ax2.set_ylabel('Unidades')
        ax2.set_xlabel('Região')
        ax2.legend(['Previsto (ML)', 'Alocado (GA)'])
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Taxa de Cobertura
        ax3 = plt.subplot(2, 3, 3)
        coverage_bins = [0, 50, 70, 90, 100, 150]
        coverage_labels = ['<50%', '50-70%', '70-90%', '90-100%', '>100%']
        df['Coverage_Range'] = pd.cut(df['Taxa_Cobertura (%)'], bins=coverage_bins, labels=coverage_labels)
        coverage_counts = df['Coverage_Range'].value_counts().sort_index()
        colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
        ax3.pie(coverage_counts, labels=coverage_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Distribuição da Taxa de Cobertura')

        # 4. Scatter: População vs Alocação
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(df['Populacao_Regiao'], df['Alocacao_Otimizada'],
                             c=df['Taxa_Cobertura (%)'], cmap='RdYlGn', s=100, alpha=0.6)
        ax4.set_xlabel('População da Região')
        ax4.set_ylabel('Alocação Otimizada')
        ax4.set_title('Alocação vs População (cor = taxa cobertura)')
        plt.colorbar(scatter, ax=ax4, label='Taxa Cobertura (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Diferença vs Previsto
        ax5 = plt.subplot(2, 3, 5)
        df_sorted = df.sort_values('Diferenca_vs_Previsto').head(20)
        colors_diff = ['red' if x < 0 else 'green' for x in df_sorted['Diferenca_vs_Previsto']]
        ax5.barh(range(len(df_sorted)), df_sorted['Diferenca_vs_Previsto'], color=colors_diff, alpha=0.7)
        ax5.set_yticks(range(len(df_sorted)))
        ax5.set_yticklabels(df_sorted['Instituicao'], fontsize=8)
        ax5.set_xlabel('Diferença (Alocado - Previsto)')
        ax5.set_title('Top 20: Maior Défice vs Excedente')
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax5.grid(True, alpha=0.3)

        # 6. Box Plot por Região
        ax6 = plt.subplot(2, 3, 6)
        df.boxplot(column='Taxa_Cobertura (%)', by='Regiao', ax=ax6)
        ax6.set_title('Distribuição da Taxa de Cobertura por Região')
        ax6.set_xlabel('Região')
        ax6.set_ylabel('Taxa de Cobertura (%)')
        plt.suptitle('')  # Remove título automático
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sistema_integrado_resultados.png', dpi=300, bbox_inches='tight')
        print("   ✅ Visualizações salvas: sistema_integrado_resultados.png")

    def save_results(self, filename='otimizacao_final.csv'):
        """Guarda os resultados em CSV"""
        self.optimization_results.to_csv(filename, index=False)
        print(f"\n💾 Resultados salvos: {filename}")

    def compare_scenarios(self, target_month, target_year, stock_scenarios):
        """
        Compara diferentes cenários de stock

        Args:
            stock_scenarios: lista de valores de stock para comparar
        """
        print("\n" + "="*70)
        print("🔄 COMPARAÇÃO DE CENÁRIOS")
        print("="*70)

        comparison_results = []

        for stock in stock_scenarios:
            print(f"\n→ Cenário: Stock = {stock:,} unidades")
            results = self.run_full_pipeline(target_month, target_year, stock)

            comparison_results.append({
                'Stock': stock,
                'Consumo_Previsto': results['Consumo_Previsto'].sum(),
                'Taxa_Cobertura_Media': results['Taxa_Cobertura (%)'].mean(),
                'Instituicoes_Cobertas_90pct': (results['Taxa_Cobertura (%)'] >= 90).sum(),
                'Instituicoes_Criticas': (results['Taxa_Cobertura (%)'] < 50).sum()
            })

        comparison_df = pd.DataFrame(comparison_results)
        print("\n📊 RESUMO COMPARATIVO:")
        print(comparison_df.to_string(index=False))

        return comparison_df


def main():
    """
    Função principal - Interface do utilizador
    """
    print("="*70)
    print("🏥 SISTEMA INTEGRADO ML + GA: CARBAPENEMES")
    print("   Machine Learning (Previsão) + Algoritmo Genético (Otimização)")
    print("="*70)

    # Configuração
    DATA_PATH = 'dataset_medicamentos_por_regiao.csv'

    # Inputs do utilizador
    print("\n📋 CONFIGURAÇÃO DO CENÁRIO:")
    print("-" * 70)

    try:
        target_month = int(input("Mês para planeamento (1-12): "))
        target_year = int(input("Ano para planeamento (ex: 2024): "))
        total_stock = int(input("Stock total disponível (unidades): "))

        # Validações
        if not 1 <= target_month <= 12:
            print("❌ Mês inválido! Usando mês 6 (Junho)")
            target_month = 6

        if total_stock <= 0:
            print("❌ Stock inválido! Usando 500000 unidades")
            total_stock = 500000

    except ValueError:
        print("\n⚠️  Inputs inválidos! Usando valores padrão:")
        target_month = 6
        target_year = 2024
        total_stock = 500000

    print(f"\n✅ Cenário configurado: {target_month}/{target_year} | Stock: {total_stock:,} unidades")

    # Inicializar sistema
    system = IntegratedSystem(DATA_PATH)

    # Treinar modelo ML
    print("\n" + "="*70)
    print("FASE 1: TREINO DO MODELO DE PREVISÃO")
    print("="*70)
    system.train_prediction_model()

    # Executar pipeline completo
    print("\n" + "="*70)
    print("FASE 2: PREVISÃO E OTIMIZAÇÃO")
    print("="*70)
    results = system.run_full_pipeline(target_month, target_year, total_stock)

    # Visualizações
    print("\n" + "="*70)
    print("FASE 3: ANÁLISE E VISUALIZAÇÃO")
    print("="*70)
    system.generate_visualizations()
    system.save_results('otimizacao_final.csv')

    # Comparação de cenários (opcional)
    compare = input("\n❓ Deseja comparar diferentes cenários de stock? (s/n): ").lower()
    if compare == 's':
        print("\nDefina 3 cenários de stock (separados por vírgula):")
        scenarios_input = input("Ex: 300000,500000,700000: ")
        try:
            scenarios = [int(s.strip()) for s in scenarios_input.split(',')]
            system.compare_scenarios(target_month, target_year, scenarios)
        except:
            print("❌ Formato inválido, pulando comparação.")

    print("\n" + "="*70)
    print("✅ SISTEMA EXECUTADO COM SUCESSO!")
    print("="*70)
    print("\n📁 Ficheiros gerados:")
    print("   1. otimizacao_final.csv - Resultados detalhados da otimização")
    print("   2. sistema_integrado_resultados.png - Visualizações completas")
    print("\n💡 Este sistema combina:")
    print("   ✓ Machine Learning para PREVER o consumo futuro")
    print("   ✓ Algoritmo Genético para OTIMIZAR a distribuição")
    print("   ✓ Análise multi-objetivo considerando necessidade, urgências e população")


if __name__ == "__main__":
    main()
