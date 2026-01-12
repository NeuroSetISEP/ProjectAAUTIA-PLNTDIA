"""
Script de treino otimizado para modelo de previsão de carbapenemes
Usa GridSearchCV, validação cruzada e análise detalhada de performance
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

class OptimizedMLTrainer:
    """
    Classe para treino otimizado com busca de hiperparâmetros
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.le_regiao = LabelEncoder()
        self.le_instituicao = LabelEncoder()
        self.feature_names = None
        self.training_results = {}

    def load_data(self):
        """Carrega e valida os dados"""
        print("=" * 60)
        print("📊 CARREGANDO DADOS")
        print("=" * 60)

        self.df = pd.read_csv(self.data_path, sep=';')
        self.df['Consumo_Carbapenemes'] = self.df['Consumo_Carbapenemes'].fillna(0)

        print(f"✅ Dataset carregado: {len(self.df)} registros")
        print(f"   📅 Período: {self.df['Periodo'].min()} até {self.df['Periodo'].max()}")
        print(f"   🏥 Hospitais únicos: {self.df['Instituicao'].nunique()}")
        print(f"   🌍 Regiões: {self.df['Regiao'].nunique()}")
        print(f"   💊 Registros com consumo > 0: {(self.df['Consumo_Carbapenemes'] > 0).sum()}")
        print(f"   📈 Consumo total: {self.df['Consumo_Carbapenemes'].sum():.0f}")

        # Estatísticas de consumo
        consumo_stats = self.df[self.df['Consumo_Carbapenemes'] > 0]['Consumo_Carbapenemes'].describe()
        print(f"\n📊 Estatísticas de Consumo (registros > 0):")
        print(f"   Média: {consumo_stats['mean']:.2f}")
        print(f"   Mediana: {consumo_stats['50%']:.2f}")
        print(f"   Desvio padrão: {consumo_stats['std']:.2f}")
        print(f"   Min: {consumo_stats['min']:.2f} | Max: {consumo_stats['max']:.2f}")

    def engineer_features(self):
        """Engenharia de features avançada"""
        print("\n" + "=" * 60)
        print("🔧 ENGENHARIA DE FEATURES")
        print("=" * 60)

        df_model = self.df.copy()

        # Encoding categórico
        df_model['Regiao_Encoded'] = self.le_regiao.fit_transform(df_model['Regiao'])
        df_model['Instituicao_Encoded'] = self.le_instituicao.fit_transform(df_model['Instituicao'])

        # Features temporais
        df_model['Mes_Sin'] = np.sin(2 * np.pi * df_model['Mes'] / 12)
        df_model['Mes_Cos'] = np.cos(2 * np.pi * df_model['Mes'] / 12)

        # Features do dataset
        self.feature_names = [
            'Ano', 'Mes', 'Mes_Sin', 'Mes_Cos',
            'Regiao_Encoded', 'Instituicao_Encoded',
            'valor_base_sazonal', 'media_3m', 'media_6m',
            'tendencia_mom', 'tendencia_yoy', 'indice_sazonal',
            'forecast_hibrido', 'variacao_prevista_pct'
        ]

        X = df_model[self.feature_names].fillna(0)
        y = df_model['Consumo_Carbapenemes']

        print(f"✅ Features criadas: {len(self.feature_names)}")
        print(f"   Dimensão X: {X.shape}")
        print(f"   Dimensão y: {y.shape}")

        # Análise de correlação com target
        print(f"\n📈 Top 10 Features por correlação com Consumo:")
        correlations = df_model[self.feature_names + ['Consumo_Carbapenemes']].corr()['Consumo_Carbapenemes'].abs().sort_values(ascending=False)
        for i, (feat, corr) in enumerate(correlations[1:11].items(), 1):
            print(f"   {i:2d}. {feat:30s}: {corr:.4f}")

        return X, y

    def train_with_grid_search(self, X, y):
        """Treina múltiplos modelos com busca de hiperparâmetros"""
        print("\n" + "=" * 60)
        print("🧠 TREINAMENTO COM GRID SEARCH")
        print("=" * 60)

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print(f"📊 Divisão dos dados:")
        print(f"   Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Teste: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Modelos e hiperparâmetros para testar
        models_params = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 5.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
        }

        best_score = -np.inf

        for name, config in models_params.items():
            print(f"\n🔍 Treinando: {name}")
            print(f"   Testando {len(config['params'])} hiperparâmetros...")

            # Grid Search com validação cruzada
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train_scaled, y_train)

            # Melhor modelo desta família
            best_estimator = grid_search.best_estimator_

            # Predições
            y_pred_train = best_estimator.predict(X_train_scaled)
            y_pred_test = best_estimator.predict(X_test_scaled)

            # Métricas
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Validação cruzada
            cv_scores = cross_val_score(best_estimator, X_train_scaled, y_train, cv=5, scoring='r2')

            # Armazenar resultados
            self.training_results[name] = {
                'model': best_estimator,
                'best_params': grid_search.best_params_,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'mae_train': mae_train,
                'mae_test': mae_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }

            print(f"   ✅ R² Train: {r2_train:.4f} | R² Test: {r2_test:.4f}")
            print(f"   📊 MAE Test: {mae_test:.2f} | RMSE Test: {rmse_test:.2f}")
            print(f"   🔄 CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   ⚙️  Melhores params: {grid_search.best_params_}")

            # Selecionar melhor modelo global (evitando overfitting)
            score = r2_test - abs(r2_train - r2_test) * 0.2 + cv_scores.mean() * 0.1
            if score > best_score:
                best_score = score
                self.best_model = best_estimator
                self.best_model_name = name

        return X_test_scaled, y_test

    def analyze_results(self, X_test_scaled, y_test):
        """Análise detalhada dos resultados"""
        print("\n" + "=" * 60)
        print("📊 ANÁLISE DE RESULTADOS")
        print("=" * 60)

        # Comparação de modelos
        print(f"\n🏆 Ranking de Modelos por R² Test:")
        sorted_models = sorted(
            self.training_results.items(),
            key=lambda x: x[1]['r2_test'],
            reverse=True
        )

        for i, (name, results) in enumerate(sorted_models, 1):
            marker = "👑" if name == self.best_model_name else "  "
            print(f"{marker} {i}. {name:20s} | R²: {results['r2_test']:.4f} | MAE: {results['mae_test']:8.2f} | RMSE: {results['rmse_test']:8.2f}")

        # Análise do melhor modelo
        print(f"\n🎯 MELHOR MODELO: {self.best_model_name}")
        best_results = self.training_results[self.best_model_name]

        print(f"\n📈 Métricas de Performance:")
        print(f"   R² Train:  {best_results['r2_train']:.4f}")
        print(f"   R² Test:   {best_results['r2_test']:.4f}")
        print(f"   R² CV:     {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}")
        print(f"   MAE Test:  {best_results['mae_test']:.2f} unidades")
        print(f"   RMSE Test: {best_results['rmse_test']:.2f} unidades")

        # Análise de overfitting
        overfit = abs(best_results['r2_train'] - best_results['r2_test'])
        if overfit < 0.05:
            print(f"   ✅ Overfitting: {overfit:.4f} (EXCELENTE)")
        elif overfit < 0.1:
            print(f"   ✅ Overfitting: {overfit:.4f} (BOM)")
        elif overfit < 0.2:
            print(f"   ⚠️  Overfitting: {overfit:.4f} (MODERADO)")
        else:
            print(f"   ❌ Overfitting: {overfit:.4f} (ALTO)")

        # Análise de erros
        y_pred = best_results['y_pred_test']
        errors = y_test - y_pred

        print(f"\n📊 Análise de Erros:")
        print(f"   Erro Médio: {errors.mean():.2f}")
        print(f"   Erro Mediano: {np.median(errors):.2f}")
        print(f"   Erro Std Dev: {errors.std():.2f}")
        print(f"   MAPE: {(np.abs(errors[y_test > 0]) / y_test[y_test > 0]).mean() * 100:.2f}%")

        # Percentis de erro
        percentiles = [90, 95, 99]
        print(f"\n   Percentis de Erro Absoluto:")
        for p in percentiles:
            val = np.percentile(np.abs(errors), p)
            print(f"   P{p}: {val:.2f} unidades")

        # Feature importance (se disponível)
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n🔍 Top 10 Features Mais Importantes:")
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            for i, idx in enumerate(indices, 1):
                print(f"   {i:2d}. {self.feature_names[idx]:30s}: {importances[idx]:.4f}")

    def save_model(self, filepath: str):
        """Salva o modelo otimizado"""
        print("\n" + "=" * 60)
        print("💾 SALVANDO MODELO")
        print("=" * 60)

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'le_regiao': self.le_regiao,
            'le_instituicao': self.le_instituicao,
            'feature_names': self.feature_names,
            'metrics': self.training_results[self.best_model_name],
            'all_results': self.training_results,
            'trained_at': datetime.now().isoformat(),
            'training_method': 'GridSearchCV with 5-fold CV'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Modelo salvo em: {filepath}")
        print(f"   Modelo: {self.best_model_name}")
        print(f"   R² Test: {self.training_results[self.best_model_name]['r2_test']:.4f}")
        print(f"   MAE Test: {self.training_results[self.best_model_name]['mae_test']:.2f}")

    def create_visualizations(self, X_test_scaled, y_test):
        """Cria visualizações dos resultados"""
        print("\n" + "=" * 60)
        print("📊 CRIANDO VISUALIZAÇÕES")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Comparação de R² scores
        ax = axes[0, 0]
        models = list(self.training_results.keys())
        r2_scores = [self.training_results[m]['r2_test'] for m in models]
        colors = ['gold' if m == self.best_model_name else 'steelblue' for m in models]

        bars = ax.barh(models, r2_scores, color=colors)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title('Comparação de R² Test entre Modelos', fontsize=14, fontweight='bold')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.3, label='Bom (0.8)')
        ax.axvline(x=0.9, color='darkgreen', linestyle='--', alpha=0.3, label='Excelente (0.9)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        # 2. Predições vs Real (melhor modelo)
        ax = axes[0, 1]
        y_pred = self.training_results[self.best_model_name]['y_pred_test']
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)

        # Linha de predição perfeita
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Predição Perfeita')

        ax.set_xlabel('Consumo Real', fontsize=12)
        ax.set_ylabel('Consumo Predito', fontsize=12)
        ax.set_title(f'Predições vs Real - {self.best_model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Distribuição de erros
        ax = axes[1, 0]
        errors = y_test - y_pred
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Erro Zero')
        ax.set_xlabel('Erro (Real - Predito)', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_title('Distribuição de Erros', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Comparação de MAE
        ax = axes[1, 1]
        mae_scores = [self.training_results[m]['mae_test'] for m in models]
        colors = ['gold' if m == self.best_model_name else 'coral' for m in models]

        bars = ax.barh(models, mae_scores, color=colors)
        ax.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
        ax.set_title('Comparação de MAE entre Modelos', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Salvar figura
        fig_path = Path(__file__).parent / 'training_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráficos salvos em: {fig_path}")

        plt.close()


def main():
    """Execução principal do treino otimizado"""
    print("\n" + "=" * 60)
    print("🚀 TREINO OTIMIZADO DE MODELO ML - CARBAPENEMES")
    print("=" * 60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Caminhos
    data_path = Path(__file__).parent / 'dataset_forecast_preparado.csv'
    model_path = Path(__file__).parent / 'backend' / 'models' / 'trained_model.pkl'

    if not data_path.exists():
        print(f"❌ Arquivo não encontrado: {data_path}")
        return

    # Criar diretório de modelos se não existir
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Inicializar trainer
    trainer = OptimizedMLTrainer(str(data_path))

    # Pipeline de treino
    trainer.load_data()
    X, y = trainer.engineer_features()
    X_test_scaled, y_test = trainer.train_with_grid_search(X, y)
    trainer.analyze_results(X_test_scaled, y_test)
    trainer.create_visualizations(X_test_scaled, y_test)
    trainer.save_model(str(model_path))

    print("\n" + "=" * 60)
    print("✅ TREINO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📂 Modelo salvo em: {model_path}")
    print(f"📊 Gráficos salvos em: training_results.png")
    print(f"\n🎯 Modelo final: {trainer.best_model_name}")
    print(f"   R² Test: {trainer.training_results[trainer.best_model_name]['r2_test']:.4f}")
    print(f"   MAE Test: {trainer.training_results[trainer.best_model_name]['mae_test']:.2f}")


if __name__ == "__main__":
    main()
