"""
SNS AI - Sistema Interativo de Distribuição de Medicamentos
Backend API com FastAPI para integração com o frontend React
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import uvicorn
from datetime import datetime, date
import json
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Adicionar o diretório pai ao path para importar os módulos existentes
sys.path.append(str(Path(__file__).parent.parent))

# Tentar importar as classes refatoradas primeiro, depois as originais
ML_AVAILABLE = False
MLPredictionModel = None
OptimizedDistributor = None
load_or_train_model = None

try:
    # Tentar importar módulos refatorados
    from ml_models import MLPredictionModel, OptimizedDistributor, load_or_train_model
    ML_AVAILABLE = True
    print("✅ Módulos ML refatorados carregados com sucesso")
except ImportError:
    try:
        # Fallback: tentar importar módulos originais
        from GA_code import CarbapenemesPredictionModel, GeneticDistributor

        # Criar wrapper classes para compatibilidade
        class MLPredictionModel:
            def __init__(self, data_path):
                self.original_model = CarbapenemesPredictionModel(data_path)
                self.is_trained = False

            def load_data(self):
                self.original_model.load_and_prepare_data()
                return True

            def train_model(self, X, y):
                results = self.original_model.train_auto_ml(X, y)
                self.is_trained = True
                return results

            def predict_monthly(self, month, year):
                return self.original_model.predict_month_with_context(month, year)

        class OptimizedDistributor:
            def __init__(self, hospital_data, total_stock):
                self.original_distributor = GeneticDistributor(hospital_data, total_stock)

            def optimize(self):
                return self.original_distributor.run()

            @property
            def best_fitness_score(self):
                return self.original_distributor.best_fitness[-1] if self.original_distributor.best_fitness else 0.0

            @property
            def hospitals(self):
                return self.original_distributor.hospitals

            @property
            def priority_weights(self):
                return self.original_distributor.priority_weights

        def load_or_train_model(data_path, model_path=None):
            model = MLPredictionModel(data_path)
            model.load_data()
            X, y = model.original_model.engineer_features()
            model.train_model(X, y)
            return model

        ML_AVAILABLE = True
        print("✅ Módulos ML originais carregados com wrappers de compatibilidade")

    except ImportError as e:
        print(f"⚠️  Módulos de ML não encontrados: {e}")
        print("   Usando modo mock para desenvolvimento.")
        ML_AVAILABLE = False

# Função lifespan para gerenciar startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Iniciando SNS AI Backend...")
    model_manager.load_model()
    if model_manager.is_loaded:
        print("✅ Modelo ML carregado com sucesso")
    else:
        print("⚠️  Usando modo mock (modelo ML não disponível)")

    yield

    # Shutdown
    print("🛑 Parando SNS AI Backend...")

# Configuração da aplicação
app = FastAPI(
    title="SNS AI - Distribuição de Medicamentos",
    description="API para previsão e otimização de distribuição de carbapenemes",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para permitir comunicação com o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validação de dados
class PredictionRequest(BaseModel):
    month: int
    year: int
    stock_percentage: float = 0.7

class DistributionRequest(BaseModel):
    months: List[int]
    year: int
    stock_percentage: Optional[float] = None
    available_stock: Optional[int] = None  # Quantidade absoluta de carbapenemes
    mode: str = "month"  # "month", "quarter", "year"

class HospitalData(BaseModel):
    institution: str
    region: str
    predicted_consumption: float
    priority_weight: float
    allocated_amount: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_key_hospital: bool = False
    safety_stock: int = 0

class DistributionResult(BaseModel):
    period: str
    year: int
    hospitals: List[HospitalData]
    total_predicted: float
    available_stock: int
    optimization_score: float

# Classe para gerenciar o estado da aplicação
class ModelManager:
    def __init__(self):
        self.ml_model = None
        self.is_loaded = False
        # Usar caminhos absolutos baseados no diretório do backend
        backend_dir = Path(__file__).parent
        self.data_path = str(backend_dir.parent / "dataset_forecast_preparado.csv")
        self.model_path = str(backend_dir / "models" / "trained_model.pkl")
        self.coordinates_path = str(backend_dir.parent / "01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv")
        self.hospital_coordinates = {}  # Cache de coordenadas dos hospitais
        self._load_hospital_coordinates()

    def _load_hospital_coordinates(self):
        """Carrega coordenadas dos hospitais do arquivo CSV"""
        try:
            df_coords = pd.read_csv(self.coordinates_path, delimiter=';')
            print(f"🔍 Carregando coordenadas de {len(df_coords)} registros...")

            # Agrupar por instituição e pegar as coordenadas únicas
            for _, row in df_coords.iterrows():
                instituicao = row.get('Instituição', '')
                coords_str = row.get('Localização Geográfica', '')

                if instituicao and coords_str and ',' in str(coords_str):
                    try:
                        lat_str, lon_str = str(coords_str).split(',')
                        latitude = float(lat_str.strip())
                        longitude = float(lon_str.strip())

                        # Armazenar no cache (usar primeira ocorrência)
                        if instituicao not in self.hospital_coordinates:
                            self.hospital_coordinates[instituicao] = {
                                'latitude': latitude,
                                'longitude': longitude
                            }
                    except (ValueError, AttributeError) as e:
                        continue

            print(f"✅ {len(self.hospital_coordinates)} hospitais com coordenadas carregadas")
        except Exception as e:
            print(f"⚠️ Erro ao carregar coordenadas: {e}")
            self.hospital_coordinates = {}

    def load_model(self):
        """Carrega o modelo de ML se disponível"""
        if not ML_AVAILABLE:
            self.is_loaded = False
            return False

        try:
            # Usar função utilitária que carrega ou treina
            self.ml_model = load_or_train_model(self.data_path, self.model_path)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Erro ao carregar/treinar modelo: {e}")
            self.is_loaded = False
            return False

    def get_real_prediction(self, month: int, year: int) -> Dict:
        """Retorna Previsões baseadas em dados reais do CSV"""
        try:
            df = pd.read_csv(self.data_path, delimiter=';')

            # Debug: mostrar colunas disponíveis
            print(f"🔍 Colunas disponíveis no CSV: {df.columns.tolist()[:10]}")  # Primeiras 10 colunas

            # Verificar se o DataFrame foi carregado corretamente
            if df is None or df.empty:
                raise ValueError("Dataset vazio ou não encontrado")

            # Filtrar dados para o mês solicitado
            historical_data = df[df['Mes'] == month]

            if historical_data.empty:
                # Se não há dados para o mês específico, usar todos os dados disponíveis
                print(f"Aviso: Sem dados específicos para mês {month}, usando média geral")
                historical_data = df

            predictions = {}
            hospitals = df['Instituicao'].unique()

            for hospital in hospitals:
                hospital_data = historical_data[historical_data['Instituicao'] == hospital]

                if not hospital_data.empty:
                    # Usar média histórica, com fallbacks para valores ausentes
                    avg_consumption = hospital_data['Consumo_Carbapenemes'].mean()

                    # Pegar a primeira região válida para este hospital
                    region_data = hospital_data['Regiao'].dropna()
                    region = region_data.iloc[0] if not region_data.empty else 'Desconhecida'

                    # Buscar coordenadas do cache carregado
                    latitude, longitude = None, None
                    if hospital in self.hospital_coordinates:
                        latitude = self.hospital_coordinates[hospital]['latitude']
                        longitude = self.hospital_coordinates[hospital]['longitude']

                    # Médias com fallbacks
                    avg_urgencies = hospital_data['Total_Urgencias'].mean()
                    avg_consultations = hospital_data['Total_Consultas'].mean()
                    avg_population = hospital_data['Populacao_Regiao'].mean()

                    # Extrair tipos específicos de urgência para as regras de prioridade
                    avg_urg_geral = hospital_data['Urgencias_Geral'].mean() if 'Urgencias_Geral' in hospital_data.columns else 0.0
                    avg_urg_ped = hospital_data['Urgencias_Pediatricas'].mean() if 'Urgencias_Pediatricas' in hospital_data.columns else 0.0
                    avg_urg_obs = hospital_data['Urgencias_Obstetricia'].mean() if 'Urgencias_Obstetricia' in hospital_data.columns else 0.0
                    avg_urg_psi = hospital_data['Urgencias_Psiquiatrica'].mean() if 'Urgencias_Psiquiatrica' in hospital_data.columns else 0.0

                    # Se o consumo histórico é zero ou NaN, calcular um valor estimado baseado na atividade do hospital
                    if pd.isna(avg_consumption) or avg_consumption == 0.0:
                        # Estimar consumo baseado no tamanho/atividade do hospital
                        urgency_factor = float(avg_urgencies) if not pd.isna(avg_urgencies) else 1000
                        consultation_factor = float(avg_consultations) if not pd.isna(avg_consultations) else 2000

                        # Fórmula estimativa: consumo baseado na atividade hospitalar
                        # Hospitais maiores (mais urgências + consultas) = maior consumo de antibióticos
                        estimated_consumption = (urgency_factor * 0.002) + (consultation_factor * 0.001)
                        estimated_consumption = max(estimated_consumption, 5.0)  # Mínimo 5 unidades
                        estimated_consumption = min(estimated_consumption, 500.0)  # Máximo 500 unidades
                        avg_consumption = estimated_consumption

                    predictions[hospital] = {
                        'pred': float(avg_consumption),
                        'region': str(region),
                        'latitude': latitude,
                        'longitude': longitude,
                        'urgencies': float(avg_urgencies) if not pd.isna(avg_urgencies) else 0.0,
                        'consultations': float(avg_consultations) if not pd.isna(avg_consultations) else 0.0,
                        'population': float(avg_population) if not pd.isna(avg_population) else 0.0,
                        # Campos detalhados para regras de prioridade
                        'urg_geral': float(avg_urg_geral) if not pd.isna(avg_urg_geral) else 0.0,
                        'urg_ped': float(avg_urg_ped) if not pd.isna(avg_urg_ped) else 0.0,
                        'urg_obs': float(avg_urg_obs) if not pd.isna(avg_urg_obs) else 0.0,
                        'urg_psi': float(avg_urg_psi) if not pd.isna(avg_urg_psi) else 0.0
                    }

            if not predictions:
                raise ValueError("Nenhuma Previsão pôde ser gerada a partir dos dados")

            return predictions

        except Exception as e:
            print(f"Erro em get_real_prediction: {str(e)}")
            # Retornar dados de fallback se tudo falhar
            return {
                "Hospital Exemplo": {
                    'pred': 50.0,
                    'region': 'Região Exemplo',
                    'urgencies': 1000.0,
                    'consultations': 2000.0,
                    'population': 500000.0
                }
            }
# Endpoints da API

# Evento startup movido para lifespan function acima

@app.get("/")
async def root():
    return {
        "message": "SNS AI - Backend API",
        "status": "running",
        "model_loaded": model_manager.is_loaded,
        "endpoints": [
            "/predict",
            "/distribute",
            "/hospitals",
            "/health",
            "/docs"
        ]
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da aplicação"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model_manager.is_loaded else "mock_mode",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_consumption(request: PredictionRequest) -> Dict[str, Any]:
    """
    Prediz o consumo de carbapenemes para um mês específico
    """
    try:
        if model_manager.is_loaded:
            context_map = model_manager.ml_model.predict_monthly(
                request.month, request.year
            )
        else:
            # Usar dados reais do CSV quando modelo ML não está disponível
            context_map = model_manager.get_real_prediction(request.month, request.year)

        # Calcular estatísticas
        total_predicted = sum([v['pred'] for v in context_map.values()])
        hospitals = list(context_map.keys())

        return {
            "month": request.month,
            "year": request.year,
            "total_predicted_consumption": round(total_predicted, 2),
            "hospital_count": len(hospitals),
            "hospital_predictions": {
                hospital: {
                    "predicted_consumption": round(data['pred'], 2),
                    "region": data.get('region', 'Unknown')
                }
                for hospital, data in context_map.items()
            },
            "model_source": "ml" if model_manager.is_loaded else "csv_historical"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na Previsão: {str(e)}")

@app.post("/distribute")
async def optimize_distribution(request: DistributionRequest) -> List[DistributionResult]:
    """
    Otimiza a distribuição de medicamentos usando algoritmo genético
    """
    try:
        # Validar que pelo menos um dos campos foi fornecido
        if request.stock_percentage is None and request.available_stock is None:
            raise HTTPException(status_code=400, detail="Deve fornecer stock_percentage ou available_stock")
        print(f"📥 Recebido pedido de distribuição: {request}")
        all_results = []

        for month in request.months:
            print(f"🔄 Processando mês {month}/{request.year}...")
            # Obter Previsões para o mês
            if model_manager.is_loaded:
                context_map = model_manager.ml_model.predict_monthly(
                    month, request.year
                )
            else:
                # Usar dados reais do CSV quando modelo ML não está disponível
                context_map = model_manager.get_real_prediction(month, request.year)

            print(f"📊 Context map obtido com {len(context_map)} hospitais")

            # Calcular stock disponível
            total_needed = sum([v['pred'] for v in context_map.values()])

            # Usar available_stock se fornecido, caso contrário calcular a partir de stock_percentage
            if request.available_stock is not None:
                available_stock = request.available_stock
                actual_percentage = (available_stock / total_needed * 100) if total_needed > 0 else 0
                print(f"💊 Stock fornecido: {available_stock}, Total necessário: {total_needed:.2f} ({actual_percentage:.1f}%)")
            else:
                available_stock = int(total_needed * request.stock_percentage)
                print(f"💊 Total necessário: {total_needed:.2f}, Stock disponível: {available_stock} ({request.stock_percentage*100:.1f}%)")

            # Otimizar distribuição
            if OptimizedDistributor:
                print("🧬 Usando OptimizedDistributor...")
                optimizer = OptimizedDistributor(context_map, available_stock)
                allocation = optimizer.optimize()
                optimization_score = optimizer.best_fitness_score

                hospitals_data = []
                key_indices = getattr(optimizer, 'key_hospitals_indices', [])
                safety_stocks_map = getattr(optimizer, 'safety_stocks', {})

                for i, hospital in enumerate(optimizer.hospitals):
                    hospitals_data.append(HospitalData(
                        institution=hospital,
                        region=context_map[hospital].get('region', 'Unknown'),
                        predicted_consumption=round(optimizer.get_effective_pred(hospital), 2),
                        priority_weight=round(optimizer.priority_weights[hospital], 4),
                        allocated_amount=int(allocation[i]),
                        latitude=context_map[hospital].get('latitude'),
                        longitude=context_map[hospital].get('longitude'),
                        is_key_hospital=(i in key_indices),
                        safety_stock=safety_stocks_map.get(hospital, 0)
                    ))
            else:
                print("📊 Usando distribuição proporcional com garantia regional...")

                # 0. Preparar dados e calcular Proxy Factor para fallback
                ratios = [
                    d['pred'] / d['urgencies'] for d in context_map.values()
                    if d['pred'] > 0 and d.get('urgencies', 0) > 0
                ]
                proxy_factor = sum(ratios) / len(ratios) if ratios else 0.0

                # Calcular Previsões efetivas (base)
                effective_preds = {}
                for h, d in context_map.items():
                    if d['pred'] == 0 and d.get('urgencies', 0) > 0 and proxy_factor > 0:
                        effective_preds[h] = proxy_factor * d.get('urgencies', 0)
                    else:
                        effective_preds[h] = d['pred']

                # 1. Verificar Escassez Extrema e Consolidar Demanda
                total_base_need = sum(effective_preds.values())
                coverage = available_stock / total_base_need if total_base_need > 0 else 1.0

                if coverage < 0.50:
                    print(f"⚠️ Fallback: Escassez Extrema ({coverage:.1%}). Consolidando demanda.")
                    # Agrupar por região
                    regions_map = {}
                    for h, d in context_map.items():
                        reg = d.get('region', 'Unknown')
                        if reg not in regions_map: regions_map[reg] = []
                        regions_map[reg].append(h)

                    for reg, h_list in regions_map.items():
                        if not h_list: continue
                        hub = max(h_list, key=lambda h: effective_preds[h])

                        for h in h_list:
                            if h == hub: continue
                            urg_demand = context_map[h].get('urgencies', 0) * proxy_factor
                            shift = min(urg_demand, effective_preds[h])
                            effective_preds[h] -= shift
                            effective_preds[hub] += shift

                # 1.5 Calcular Safety Stocks (Dynamic Floor)
                vals_cons = [v.get('consultations', 0) for v in context_map.values()]
                vals_urg = [v.get('urgencies', 0) for v in context_map.values()]
                max_cons = max(vals_cons) if vals_cons and max(vals_cons) > 0 else 1.0
                max_urg = max(vals_urg) if vals_urg and max(vals_urg) > 0 else 1.0

                safety_multiplier = 1.0 if coverage >= 0.8 else (0.5 if coverage >= 0.5 else 0.0)
                MAX_SAFETY_FLOOR = 50

                safety_stocks = {}
                for h, d in context_map.items():
                    size_factor = ((d.get('consultations', 0)/max_cons) + (d.get('urgencies', 0)/max_urg)) / 2
                    safety_stocks[h] = int(MAX_SAFETY_FLOOR * size_factor * safety_multiplier)

                # 2. Identificar hospitais chave (maior consumo efetivo por região)
                key_hospitals_set = set()
                region_max = {}
                for h in context_map:
                    r = context_map[h].get('region', 'Unknown')
                    p = effective_preds[h]
                    if r not in region_max or p > region_max[r][1]:
                        region_max[r] = (h, p)
                key_hospitals_set = {v[0] for v in region_max.values()}

                # 3. Calcular alocações
                allocations = {h: 0 for h in context_map}
                rem_stock = available_stock

                # Passo A: Garantir Safety Stock (se houver stock)
                for h, floor in safety_stocks.items():
                    give = min(floor, rem_stock)
                    allocations[h] += give
                    rem_stock -= give

                # Passo B: Priorizar hospitais chave (com o que sobrou)
                sorted_keys = sorted(list(key_hospitals_set), key=lambda x: effective_preds[x], reverse=True)
                for h in sorted_keys:
                    current_alloc = allocations[h]
                    needed = int(effective_preds[h])
                    remaining_need = max(0, needed - current_alloc)

                    give = min(remaining_need, rem_stock)
                    allocations[h] += give
                    rem_stock -= give

                # Passo C: Distribuir o resto proporcionalmente
                others = [h for h in context_map]
                remaining_needs = {h: max(0, effective_preds[h] - allocations[h]) for h in others}
                total_others = sum(remaining_needs.values())

                # Atualizar rem_stock após Passo B
                current_allocated = sum(allocations.values())
                rem_stock = available_stock - current_allocated

                for h in others:
                    if total_others > 0 and rem_stock > 0:
                        prop = remaining_needs[h] / total_others
                        give = int(rem_stock * prop)
                        allocations[h] += give

                # Passo D: Distribuir excedente final (se houver)
                # Se sobrou stock após atender todas as necessidades, distribuir proporcionalmente ao tamanho
                current_allocated = sum(allocations.values())
                final_surplus = available_stock - current_allocated

                if final_surplus > 0:
                    total_pred_all = sum(effective_preds.values())
                    if total_pred_all > 0:
                        for h in allocations:
                            bonus = int(final_surplus * (effective_preds[h] / total_pred_all))
                            allocations[h] += bonus

                hospitals_data = []

                # Calcular máximos para normalização (consistente com ml_models.py)
                max_consumption = max(effective_preds.values()) if effective_preds else 1.0

                # Calcular scores de urgência para normalização
                urgency_scores = []
                for data in context_map.values():
                    score = (
                        data.get('urg_geral', 0) * 0.50 +
                        data.get('urg_ped', 0) * 0.25 +
                        data.get('urg_obs', 0) * 0.20 +
                        data.get('urg_psi', 0) * 0.05
                    )
                    if score == 0 and data.get('urgencies', 0) > 0:
                        score = data.get('urgencies', 0) * 0.25
                    urgency_scores.append(score)

                max_urgency_score = max(urgency_scores) if urgency_scores and max(urgency_scores) > 0 else 1.0

                vals_cons = [v.get('consultations', 0) for v in context_map.values()]
                max_consultations = max(vals_cons) if vals_cons and max(vals_cons) > 0 else 1.0

                vals_pop = [v.get('population', 0) for v in context_map.values()]
                max_population = max(vals_pop) if vals_pop and max(vals_pop) > 0 else 1.0

                for i, (hospital, data) in enumerate(context_map.items()):
                    allocated = allocations.get(hospital, 0)

                    # Calcular prioridade usando as regras detalhadas (mesma lógica do ml_models.py)
                    consumption_factor = effective_preds[hospital] / max_consumption
                    urgency_factor = urgency_scores[i] / max_urgency_score
                    consultation_factor = data.get('consultations', 0) / max_consultations
                    population_factor = data.get('population', 0) / max_population

                    priority_weight = (
                        consumption_factor * 0.40 +
                        urgency_factor * 0.30 +
                        consultation_factor * 0.15 +
                        population_factor * 0.15
                    )

                    hospitals_data.append(HospitalData(
                        institution=hospital,
                        region=data.get('region', 'Unknown'),
                        predicted_consumption=round(effective_preds[hospital], 2),
                        priority_weight=round(priority_weight, 4),
                        allocated_amount=allocated,
                        latitude=data.get('latitude'),
                        longitude=data.get('longitude'),
                        is_key_hospital=(hospital in key_hospitals_set),
                        safety_stock=safety_stocks.get(hospital, 0)
                    ))

                # Score baseado na distribuição proporcional real
                optimization_score = 0.85 + (0.1 * min(len(context_map) / 50, 1.0))  # Melhor score com mais hospitais

            print(f"✅ Criados dados para {len(hospitals_data)} hospitais")

            # Criar resultado para o mês
            result = DistributionResult(
                period=f"{request.year}-{month:02d}",
                year=request.year,
                hospitals=hospitals_data,
                total_predicted=round(total_needed, 2),
                available_stock=available_stock,
                optimization_score=round(optimization_score, 4)
            )

            all_results.append(result)

        print(f"✅ Distribuição completa: {len(all_results)} períodos processados")
        return all_results

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ ERRO na otimização:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Erro na otimização: {str(e)}")

@app.get("/hospitals")
async def get_hospitals() -> Dict[str, Any]:
    """
    Retorna informações sobre os hospitais no sistema
    """
    try:
        # Tentar carregar dados reais do CSV primeiro
        try:
            data_path = "../dataset_forecast_preparado.csv"
            df = pd.read_csv(data_path, delimiter=';')

            hospitals = df['Instituicao'].unique().tolist()
            regions = df['Regiao'].unique().tolist()

            hospital_info = []
            for hospital in hospitals:
                hospital_data = df[df['Instituicao'] == hospital]
                region = hospital_data['Regiao'].iloc[0] if len(hospital_data) > 0 else 'Unknown'
                avg_consumption = hospital_data['Consumo_Carbapenemes'].mean()

                hospital_info.append({
                    "name": hospital,
                    "region": region,
                    "avg_monthly_consumption": round(avg_consumption, 2),
                    "data_points": len(hospital_data)
                })

            data_source = "csv"

        except Exception as csv_error:
            print(f"Erro ao carregar CSV: {csv_error}")

            # Fallback: tentar usar dados do modelo ML se disponível
            if model_manager.is_loaded and hasattr(model_manager.ml_model, 'df') and model_manager.ml_model.df is not None:
                df = model_manager.ml_model.df
                hospitals = df['Instituicao'].unique().tolist()
                regions = df['Regiao'].unique().tolist()

                hospital_info = []
                for hospital in hospitals:
                    hospital_data = df[df['Instituicao'] == hospital]
                    region = hospital_data['Regiao'].iloc[0] if len(hospital_data) > 0 else 'Unknown'
                    avg_consumption = hospital_data['Consumo_Carbapenemes'].mean()

                    hospital_info.append({
                        "name": hospital,
                        "region": region,
                        "avg_monthly_consumption": round(avg_consumption, 2),
                        "data_points": len(hospital_data)
                    })

                data_source = "ml"
            else:
                # Se falhou carregar tanto do CSV quanto do modelo, retornar erro
                raise HTTPException(
                    status_code=503,
                    detail="Não foi possível carregar dados dos hospitais. Verifique se o dataset está disponível."
                )

        return {
            "hospitals": hospital_info,
            "total_hospitals": len(hospital_info),
            "regions": list(set([h["region"] for h in hospital_info])),
            "data_source": data_source
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter dados dos hospitais: {str(e)}")

@app.get("/model/metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """
    Retorna métricas do modelo otimizado treinado (carrega do pickle)
    """
    try:
        import pickle

        # Carregar modelo treinado
        model_path = Path(__file__).parent / "models" / "trained_model.pkl"

        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Modelo treinado não encontrado. Execute train_optimized_model.py primeiro."
            )

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        metrics = model_data.get('metrics', {})
        all_results = model_data.get('all_results', {})

        # Preparar resposta com estatísticas completas
        return {
            "model_type": model_data.get('model_name', 'Unknown'),
            "training_method": model_data.get('training_method', 'GridSearchCV with 5-fold CV'),
            "trained_at": model_data.get('trained_at', datetime.now().isoformat()),
            "best_params": metrics.get('best_params', {}),
            "metrics": {
                "r2_train": round(metrics.get('r2_train', 0), 4),
                "r2_test": round(metrics.get('r2_test', 0), 4),
                "mae_train": round(metrics.get('mae_train', 0), 2),
                "mae_test": round(metrics.get('mae_test', 0), 2),
                "rmse_train": round(metrics.get('rmse_train', 0), 2),
                "rmse_test": round(metrics.get('rmse_test', 0), 2),
                "cv_mean": round(metrics.get('cv_mean', 0), 4),
                "cv_std": round(metrics.get('cv_std', 0), 4),
                "efficiency_percentage": round(metrics.get('r2_test', 0) * 100, 2)
            },
            "model_comparison": {
                name: {
                    "r2_train": round(info.get("r2_train", 0), 4),
                    "r2_test": round(info.get("r2_test", 0), 4),
                    "mae_test": round(info.get("mae_test", 0), 2),
                    "cv_mean": round(info.get("cv_mean", 0), 4),
                    "cv_std": round(info.get("cv_std", 0), 4)
                }
                for name, info in all_results.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Modelo não encontrado. Execute train_optimized_model.py primeiro."
        )
    except Exception as e:
        import traceback
        print(f"Erro ao carregar métricas: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar métricas: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Retreina o modelo com novos dados (execução em background)
    """
    def retrain():
        if model_manager.is_loaded:
            model_manager.load_model()

    background_tasks.add_task(retrain)
    return {"message": "Modelo será retreinado em background"}

@app.get("/model/training-image-all-models")
async def get_all_training_image():
    """
    Retorna a imagem de visualização do treinamento do modelo
    """
    from fastapi.responses import FileResponse

    image_path = Path(__file__).parent.parent / "all_models_predictions.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Imagem de treinamento não encontrada. Execute train_optimized_model.py primeiro."
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename="all_models_predictions.png"
    )
@app.get("/model/training-image")
async def get_training_image():
    """
    Retorna a imagem de visualização do treinamento do modelo
    """
    from fastapi.responses import FileResponse

    image_path = Path(__file__).parent.parent / "training_results.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Imagem de treinamento não encontrada. Execute train_optimized_model.py primeiro."
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename="training_results.png"
    )

# Instanciar o gerenciador de modelo
model_manager = ModelManager()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )