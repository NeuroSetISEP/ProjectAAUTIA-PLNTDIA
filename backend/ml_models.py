"""
Módulo refatorado das classes de ML para melhor integração com a API
Baseado no GA_code.py original mas organizado para uso em produção
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pygad
from typing import Dict, List, Tuple, Any
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class MLPredictionModel:
    """
    Classe refatorada para previsão de consumo de carbapenemes
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.le_regiao = LabelEncoder()
        self.le_instituicao = LabelEncoder()
        self.is_trained = False
        self.model_metrics = {}

    def load_data(self) -> bool:
        """Carrega e prepara os dados"""
        try:
            if not self.data_path or not os.path.exists(self.data_path):
                print(f"⚠️ Dataset não encontrado: {self.data_path}")
                return False

            self.df = pd.read_csv(self.data_path, sep=';')
            self.df['Consumo_Carbapenemes'] = self.df['Consumo_Carbapenemes'].fillna(0)

            print(f"✅ Dataset carregado: {len(self.df)} registros, {self.df['Instituicao'].nunique()} hospitais")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False

    def engineer_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Engenharia de features avançada"""
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        df_model = self.df.copy()

        # Encoding categórico
        df_model['Regiao_Encoded'] = self.le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = self.le_instituicao.fit_transform(df_model['Instituicao'])

        # Features temporais
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        # Definir features para o modelo
        self.feature_names = [
            'Ano', 'Mes', 'Mes_Sin', 'Mes_Cos', 'Regiao_Encoded', 'Instituicao_Encoded',
            'valor_base_sazonal', 'media_3m', 'media_6m', 'tendencia_mom',
            'tendencia_yoy', 'indice_sazonal', 'forecast_hibrido', 'variacao_prevista_pct'
        ]

        # Preparar X e y
        X = df_model[self.feature_names].fillna(0)
        y = df_model['Consumo_Carbapenemes']

        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Treina múltiplos modelos e seleciona o melhor"""

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Modelos candidatos
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }

        best_score = -np.inf
        best_model_name = None
        results = {}

        # Treinar e avaliar cada modelo
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

            # Predições
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Métricas
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            results[name] = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'model': model
            }

            # Selecionar melhor modelo (evita overfitting)
            score = r2_test - abs(r2_train - r2_test) * 0.1  # Penaliza overfitting
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = model

        self.is_trained = True
        self.model_metrics = results

        print(f"🎯 Melhor modelo: {best_model_name} (R² test: {results[best_model_name]['r2_test']:.3f})")

        return results

    def predict_monthly(self, month: int, year: int) -> Dict[str, Dict[str, Any]]:
        """Prediz consumo para um mês específico de todos os hospitais"""

        if not self.is_trained:
            raise ValueError("Modelo não treinado. Execute train_model() primeiro.")

        if self.df is None or self.df.empty:
            raise ValueError("Dataset não carregado. Execute load_data() primeiro.")

        # Obter dados mais recentes de cada hospital
        latest_data = self.df.sort_values('Periodo').groupby('Instituicao').tail(1).copy()

        if latest_data.empty:
            raise ValueError("Nenhum dado disponível para predição.")

        # Ajustar para o mês/ano desejado
        latest_data['Mes'] = month
        latest_data['Ano'] = year
        latest_data['Mes_Sin'] = np.sin(2 * np.pi * month / 12)
        latest_data['Mes_Cos'] = np.cos(2 * np.pi * month / 12)
        latest_data['Regiao_Encoded'] = self.le_regiao.transform(latest_data['Regiao'])
        latest_data['Instituicao_Encoded'] = self.le_instituicao.transform(latest_data['Instituicao'])

        # Preparar features e fazer predições
        X_future = latest_data[self.feature_names].fillna(0)
        X_future_scaled = self.scaler.transform(X_future)
        predictions = np.maximum(self.best_model.predict(X_future_scaled), 0)

        # Organizar resultados
        results = {}
        for i, (_, row) in enumerate(latest_data.iterrows()):
            hospital = row['Instituicao']
            results[hospital] = {
                'pred': predictions[i],
                'region': row['Regiao'],
                'urgencies': row.get('Total_Urgencias', 0),
                'consultations': row.get('Total_Consultas', 0),
                'population': row.get('Populacao_Regiao', 0)
            }

        return results

    def save_model(self, filepath: str):
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo não treinado.")

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'le_regiao': self.le_regiao,
            'le_instituicao': self.le_instituicao,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics,
            'trained_at': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Modelo salvo em: {filepath}")

    def load_model(self, filepath: str) -> bool:
        """Carrega modelo pré-treinado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.le_regiao = model_data['le_regiao']
            self.le_instituicao = model_data['le_instituicao']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data.get('metrics', {})
            self.is_trained = True

            print(f"📂 Modelo carregado: {filepath}")

            # IMPORTANTE: Recarregar os dados para permitir predições
            print(f"📊 Carregando dados de: {self.data_path}")
            if self.data_path and self.load_data():
                print(f"✅ Dados carregados com sucesso para predições")
            else:
                print(f"⚠️ AVISO: Não foi possível carregar dados. Predições podem falhar.")
                print(f"   Data path: {self.data_path}")
                print(f"   Existe? {os.path.exists(self.data_path) if self.data_path else 'N/A'}")

            return True

        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            import traceback
            print(traceback.format_exc())
            return False


class OptimizedDistributor:
    """
    Algoritmo genético otimizado para distribuição de medicamentos
    """

    def __init__(self, hospital_data: Dict[str, Dict], total_stock: int,
                 generations: int = 200, population_size: int = 80):
        self.hospital_data = hospital_data
        self.total_stock = total_stock
        self.generations = generations
        self.population_size = population_size
        self.hospitals = list(hospital_data.keys())
        self.num_hospitals = len(self.hospitals)
        self.best_allocation = None
        self.best_fitness_score = None
        self.optimization_history = []
        # --- NOVO: calcular fator médio pred/urgencies ANTES de priority_weights ---
        ratios = [
            data['pred'] / data['urgencies']
            for data in self.hospital_data.values()
            if data['pred'] > 0 and data.get('urgencies', 0) > 0
        ]
        self.proxy_factor = np.mean(ratios) if ratios else 0.0
        self.priority_weights = self._compute_priority_weights()

    def _get_effective_pred(self, data):
        pred = data['pred']
        urg = data.get('urgencies', 0)
        if pred == 0 and urg > 0 and self.proxy_factor > 0:
            return self.proxy_factor * urg
        return pred

    def _compute_priority_weights(self):
        weights = {}
        consumptions = [self._get_effective_pred(data) for data in self.hospital_data.values()]
        urgencies = [data.get('urgencies', 0) for data in self.hospital_data.values()]
        populations = [data.get('population', 0) for data in self.hospital_data.values()]
        max_consumption = max(consumptions) if max(consumptions) > 0 else 1
        max_urgency = max(urgencies) if max(urgencies) > 0 else 1
        max_population = max(populations) if max(populations) > 0 else 1
        for hospital in self.hospitals:
            data = self.hospital_data[hospital]
            eff_pred = self._get_effective_pred(data)
            if eff_pred == 0:
                weights[hospital] = 0.0
                continue
            consumption_factor = eff_pred / max_consumption
            urgency_factor = data.get('urgencies', 0) / max_urgency
            population_factor = data.get('population', 0) / max_population
            priority = (
                consumption_factor * 0.5 +
                urgency_factor * 0.30 +
                population_factor * 0.20
            )
            weights[hospital] = priority
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {h: w/total_weight for h, w in weights.items()}
        return weights

    def fitness_function(self, ga_instance, solution, solution_idx):
        allocation = np.maximum(solution.astype(int), 0)
        total_allocated = np.sum(allocation)
        if total_allocated > self.total_stock:
            excess = total_allocated - self.total_stock
            return 1.0 / (1.0 + excess * 10)
        total_error = 0.0
        total_needs = 0.0
        for i, hospital in enumerate(self.hospitals):
            data = self.hospital_data[hospital]
            eff_pred = self._get_effective_pred(data)
            allocated = allocation[i]
            priority = self.priority_weights[hospital]
            if eff_pred == 0:
                if allocated > 0:
                    total_error += allocated * 100.0
                continue
            total_needs += eff_pred
            if allocated < eff_pred:
                shortage = eff_pred - allocated
                shortage_pct = shortage / eff_pred
                if shortage_pct < 0.20:
                    error = shortage_pct * priority * 100
                else:
                    error = (shortage_pct ** 2) * priority * 500
            else:
                excess = allocated - eff_pred
                excess_pct = excess / eff_pred
                if excess_pct <= 0.20:
                    error = excess_pct * 10
                elif excess_pct <= 0.20:
                    error = excess_pct * 50
                else:
                    error = (excess_pct ** 2) * 300
            total_error += error
        utilization = total_allocated / self.total_stock if self.total_stock > 0 else 0
        if utilization >= 0.95:
            utilization_bonus = 200
        elif utilization >= 0.90:
            utilization_bonus = 150
        elif utilization >= 0.80:
            utilization_bonus = 100
        else:
            utilization_bonus = utilization * 100
        fitness = 10000.0 / (1.0 + total_error) + utilization_bonus
        return max(fitness, 1.0)

    def _create_initial_population(self) -> np.ndarray:
        population = []
        predictions = np.array([self._get_effective_pred(self.hospital_data[h]) for h in self.hospitals])
        total_predicted = np.sum(predictions)
        if total_predicted > 0:
            base_allocation = (predictions / total_predicted * self.total_stock).astype(int)
        else:
            base_allocation = np.zeros(len(self.hospitals), dtype=int)
        for _ in range(self.population_size):
            noise = np.random.normal(0, 0.05, size=len(self.hospitals))
            variation = np.clip(base_allocation * (1 + noise), 0, None).astype(int)
            while np.sum(variation) > self.total_stock:
                max_idx = np.argmax(variation)
                if variation[max_idx] > 0:
                    variation[max_idx] -= 1
            population.append(variation)
        return np.array(population)

    def optimize(self) -> List[int]:
        initial_population = self._create_initial_population()
        gene_space = []
        for hospital in self.hospitals:
            data = self.hospital_data[hospital]
            eff_pred = self._get_effective_pred(data)
            if eff_pred == 0:
                gene_space.append({'low': 0, 'high': 0})
            else:
                min_alloc = int(eff_pred * 0.8)
                max_alloc = int(eff_pred * 1.2)
                gene_space.append({'low': min_alloc, 'high': max_alloc})
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=self.population_size // 2,
            fitness_func=self.fitness_function,
            sol_per_pop=self.population_size,
            num_genes=self.num_hospitals,
            gene_type=int,
            gene_space=gene_space,
            initial_population=initial_population,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            stop_criteria="saturate_10"
        )
        print(f"🧬 Iniciando otimização genética: {self.generations} gerações, {self.population_size} indivíduos...")
        ga_instance.run()
        self.best_allocation = ga_instance.best_solution()[0].astype(int)
        self.best_fitness_score = ga_instance.best_solution()[1]
        self.optimization_history = ga_instance.best_solutions_fitness
        return self.best_allocation.tolist()


# Função utilitária para carregar modelo treinado ou treinar novo
def load_or_train_model(data_path: str, model_path: str = None) -> MLPredictionModel:
    """
    Carrega modelo existente ou treina um novo
    """
    model = MLPredictionModel(data_path)

    # Tentar carregar modelo existente
    if model_path and os.path.exists(model_path):
        if model.load_model(model_path):
            print("📂 Modelo pré-treinado carregado com sucesso")
            return model

    # Se não conseguir carregar, treinar novo modelo
    print("🔄 Treinando novo modelo...")

    if not model.load_data():
        raise Exception("Não foi possível carregar os dados para treinamento")

    X, y = model.engineer_features()
    model.train_model(X, y)

    # Salvar modelo treinado
    if model_path:
        model.save_model(model_path)

    return model