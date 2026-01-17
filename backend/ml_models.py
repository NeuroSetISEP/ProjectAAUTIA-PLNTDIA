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

            # Previsões
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
            raise ValueError("Nenhum dado disponível para Previsão.")

        # Ajustar para o mês/ano desejado
        latest_data['Mes'] = month
        latest_data['Ano'] = year
        latest_data['Mes_Sin'] = np.sin(2 * np.pi * month / 12)
        latest_data['Mes_Cos'] = np.cos(2 * np.pi * month / 12)
        latest_data['Regiao_Encoded'] = self.le_regiao.transform(latest_data['Regiao'])
        latest_data['Instituicao_Encoded'] = self.le_instituicao.transform(latest_data['Instituicao'])

        # Preparar features e fazer Previsões
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
                # Detailed urgency types for weighted priority
                'urg_geral': row.get('Urgencias_Geral', 0),
                'urg_ped': row.get('Urgencias_Pediatricas', 0),
                'urg_obs': row.get('Urgencias_Obstetricia', 0),
                'urg_psi': row.get('Urgencias_Psiquiatrica', 0),
                'urgencies': row.get('Total_Urgencias', 0), # Fallback
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

            # IMPORTANTE: Recarregar os dados para permitir Previsões
            print(f"📊 Carregando dados de: {self.data_path}")
            if self.data_path and self.load_data():
                print(f"✅ Dados carregados com sucesso para Previsões")
            else:
                print(f"⚠️ AVISO: Não foi possível carregar dados. Previsões podem falhar.")
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
                 generations: int = 400, population_size: int = 100):
        self.hospital_data = hospital_data
        self.total_stock = total_stock
        self.generations = generations
        self.population_size = population_size
        self.hospitals = list(hospital_data.keys())
        self.num_hospitals = len(self.hospitals)
        self.best_allocation = None
        self.best_fitness_score = None
        self.optimization_history = []

        # 1. Calcular Proxy Factor (relação Consumo/Urgência)
        ratios = [
            data['pred'] / data['urgencies']
            for data in self.hospital_data.values()
            if data['pred'] > 0 and data.get('urgencies', 0) > 0
        ]
        self.proxy_factor = np.mean(ratios) if ratios else 0.0

        # 2. Calcular Previsões Base (corrigindo zeros)
        self.effective_predictions = {}
        for h in self.hospitals:
            data = self.hospital_data[h]
            pred = data['pred']
            urg = data.get('urgencies', 0)
            if pred == 0 and urg > 0 and self.proxy_factor > 0:
                self.effective_predictions[h] = self.proxy_factor * urg
            else:
                self.effective_predictions[h] = pred

        # 3. Verificar Escassez Extrema e Consolidar Demanda
        total_needed = sum(self.effective_predictions.values())
        self.coverage_ratio = (total_stock / total_needed) if total_needed > 0 else 1.0
        self.is_extreme_shortage = self.coverage_ratio < 0.50

        if self.is_extreme_shortage:
            print(f"⚠️ ESCASSEZ EXTREMA DETECTADA ({self.coverage_ratio:.1%}). Consolidando demanda nos hospitais chave.")
            self._consolidate_regional_demand()

        # NOVA LÓGICA: Dynamic Safety Stock
        self.safety_stocks = self._calculate_safety_stocks()

        # 4. Inicializar pesos e chaves
        self.priority_weights = self._compute_priority_weights()
        self.key_hospitals_indices = self._identify_key_hospitals()

    def get_effective_pred(self, hospital: str) -> float:
        return self.effective_predictions.get(hospital, 0.0)

    def _identify_key_hospitals(self) -> List[int]:
        """Identifica o maior hospital de cada região (baseado em previsão) para garantia de stock"""
        region_map = {} # region -> (index, prediction)

        for i, h in enumerate(self.hospitals):
            data = self.hospital_data[h]
            region = data.get('region', 'Unknown')
            pred = self.get_effective_pred(h)

            if region not in region_map:
                region_map[region] = (i, pred)
            else:
                # Se este hospital tem maior previsão que o atual campeão da região
                if pred > region_map[region][1]:
                    region_map[region] = (i, pred)

        # Retorna lista de índices dos hospitais chave
        return [val[0] for val in region_map.values()]

    def _consolidate_regional_demand(self):
        """Move a demanda de urgência dos hospitais satélites para o Hub regional em caso de escassez"""
        # Agrupar por região
        regions = {}
        for h in self.hospitals:
            reg = self.hospital_data[h].get('region', 'Unknown')
            if reg not in regions: regions[reg] = []
            regions[reg].append(h)

        for reg, hospitals in regions.items():
            if not hospitals: continue
            # Identificar Hub (maior Previsão base)
            hub = max(hospitals, key=lambda h: self.effective_predictions[h])

            for h in hospitals:
                if h == hub: continue

                # Calcular parcela de urgência (Life-Saving portion)
                urgencies = self.hospital_data[h].get('urgencies', 0)
                urgency_demand = urgencies * self.proxy_factor

                # Não podemos mover mais do que a Previsão total do hospital
                current_pred = self.effective_predictions[h]
                shift_amount = min(urgency_demand, current_pred)

                # Mover demanda: Hub assume a carga de urgência do satélite
                self.effective_predictions[h] -= shift_amount
                self.effective_predictions[hub] += shift_amount

    def _calculate_safety_stocks(self) -> Dict[str, int]:
        """Calcula stock de segurança baseado no tamanho do hospital e cobertura global"""
        # Métricas de tamanho
        vals_cons = [d.get('consultations', 0) for d in self.hospital_data.values()]
        vals_urg = [d.get('urgencies', 0) for d in self.hospital_data.values()]

        max_cons = max(vals_cons) if vals_cons and max(vals_cons) > 0 else 1.0
        max_urg = max(vals_urg) if vals_urg and max(vals_urg) > 0 else 1.0

        # Multiplicador dinâmico baseado na cobertura
        if self.coverage_ratio >= 0.80:
            multiplier = 1.0
        elif self.coverage_ratio >= 0.50:
            multiplier = 0.5
        else:
            multiplier = 0.0 # Desativa em escassez extrema para priorizar Hubs

        safety_stocks = {}
        # Base máxima de segurança (ex: 50 unidades para o maior hospital)
        MAX_SAFETY_FLOOR = 50

        for h in self.hospitals:
            d = self.hospital_data[h]
            # Fator de tamanho combinando Consultas e Urgências
            size_factor = (
                (d.get('consultations', 0) / max_cons) +
                (d.get('urgencies', 0) / max_urg)
            ) / 2

            safety_stocks[h] = int(MAX_SAFETY_FLOOR * size_factor * multiplier)

        return safety_stocks

    def _compute_priority_weights(self):
        weights = {}
        consumptions = [self.get_effective_pred(h) for h in self.hospitals]

        # Calculate weighted urgency score for each hospital
        urgency_scores = []
        for data in self.hospital_data.values():
            score = (
                data.get('urg_geral', 0) * 0.50 +
                data.get('urg_ped', 0) * 0.25 +
                data.get('urg_obs', 0) * 0.20 +
                data.get('urg_psi', 0) * 0.05
            )
            # Fallback if specific types are missing but total exists
            if score == 0 and data.get('urgencies', 0) > 0:
                score = data.get('urgencies', 0) * 0.25 # Average weight
            urgency_scores.append(score)

        consultations = [data.get('consultations', 0) for data in self.hospital_data.values()]
        populations = [data.get('population', 0) for data in self.hospital_data.values()]

        max_consumption = max(consumptions) if max(consumptions) > 0 else 1
        max_urgency_score = max(urgency_scores) if max(urgency_scores) > 0 else 1
        max_consultations = max(consultations) if max(consultations) > 0 else 1
        max_population = max(populations) if max(populations) > 0 else 1

        for i, hospital in enumerate(self.hospitals):
            data = self.hospital_data[hospital]
            eff_pred = self.get_effective_pred(hospital)
            if eff_pred == 0:
                weights[hospital] = 0.0
                continue
            consumption_factor = eff_pred / max_consumption
            urgency_factor = urgency_scores[i] / max_urgency_score
            consultation_factor = data.get('consultations', 0) / max_consultations
            population_factor = data.get('population', 0) / max_population
            priority = (
                consumption_factor * 0.40 +  # Predicted Need
                urgency_factor * 0.30 +      # Weighted Urgencies
                consultation_factor * 0.15 + # Total Consultations
                population_factor * 0.15     # Regional Population
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
            eff_pred = self.get_effective_pred(hospital)
            allocated = allocation[i]
            priority = self.priority_weights[hospital]
            if eff_pred == 0:
                if allocated > 0:
                    total_error += allocated * 100.0
                continue

            # Check Safety Stock (Floor)
            min_stock = self.safety_stocks.get(hospital, 0)
            if allocated < min_stock:
                deficit = min_stock - allocated
                # Aumentar drasticamente a penalidade para evitar "quase lá" (faltando 1 ou 2)
                # Multiplicador linear alto (2000) garante que cada unidade faltante custe muito caro
                total_error += (deficit * 2000) + (deficit ** 2) * 100

            total_needs += eff_pred
            if allocated < eff_pred:
                shortage = eff_pred - allocated
                shortage_pct = shortage / eff_pred
                if shortage_pct < 0.20:
                    error = shortage_pct * priority * 100
                else:
                    error = (shortage_pct ** 2) * priority * 500

                # NOVA REGRA: Se for hospital chave da região, penalidade massiva se não estiver cheio
                if i in self.key_hospitals_indices:
                    # Penalidade extra severa para garantir stock total nestes hospitais
                    error += (shortage_pct ** 2) * 5000
            else:
                excess = allocated - eff_pred
                excess_pct = excess / eff_pred
                if excess_pct <= 0.20:
                    error = excess_pct * 10
                elif excess_pct <= 0.50:
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

        # 1. Criar alocação base inteligente (Safety Stock + Proporcional)
        base_allocation = np.zeros(self.num_hospitals, dtype=int)
        remaining_stock = self.total_stock

        # A. Garantir Safety Stock primeiro na população inicial
        for i, h in enumerate(self.hospitals):
            s_stock = self.safety_stocks.get(h, 0)
            give = min(s_stock, remaining_stock)
            base_allocation[i] = give
            remaining_stock -= give

        # B. Distribuir restante proporcionalmente à Previsão efetiva
        if remaining_stock > 0:
            predictions = np.array([self.get_effective_pred(h) for h in self.hospitals])
            total_pred = np.sum(predictions)

            if total_pred > 0:
                proportions = predictions / total_pred
                extra = (proportions * remaining_stock).astype(int)
                base_allocation += extra

                # Distribuir sobras de arredondamento para o maior hospital
                current_total = np.sum(base_allocation)
                diff = self.total_stock - current_total
                if diff > 0:
                    max_idx = np.argmax(predictions)
                    base_allocation[max_idx] += diff
            else:
                # Se não há Previsões mas sobrou stock, alocar ao primeiro
                base_allocation[0] += remaining_stock

        # 2. Gerar população variando dessa base otimizada
        for _ in range(self.population_size):
            noise = np.random.normal(0, 0.05, size=self.num_hospitals)
            variation = base_allocation.astype(float) * (1 + noise)
            variation = np.maximum(variation, 0).astype(int)

            # Reforçar safety stock na população inicial para evitar que o ruído o quebre
            for i, h in enumerate(self.hospitals):
                s_stock = self.safety_stocks.get(h, 0)
                if variation[i] < s_stock:
                    variation[i] = s_stock

            while np.sum(variation) > self.total_stock:
                max_idx = np.argmax(variation)
                if variation[max_idx] > 0:
                    variation[max_idx] -= 1
            population.append(variation)
        return np.array(population)

    def optimize(self) -> List[int]:
        initial_population = self._create_initial_population()
        gene_space = []

        # Expandir limite superior se houver excedente (coverage > 1.0)
        # Garante que o GA possa alocar o excedente em vez de ficar preso no teto de 1.2x
        upper_bound_mult = max(1.5, self.coverage_ratio * 2.0)

        for hospital in self.hospitals:
            eff_pred = self.get_effective_pred(hospital)
            safety = self.safety_stocks.get(hospital, 0)

            # Limite inferior 0 permite ao fitness function penalizar faltas corretamente
            # Limite superior deve acomodar: Previsão expandida OU Safety Stock (o que for maior)
            high = max(int(eff_pred * upper_bound_mult), int(safety * 2.0), 50)

            gene_space.append({'low': 0, 'high': high})

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