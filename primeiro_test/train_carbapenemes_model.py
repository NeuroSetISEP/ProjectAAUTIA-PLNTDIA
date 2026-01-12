"""
Modelo de Machine Learning para Previsão do Consumo de Carbapenemes
Por Instituição, Região e População

Este script treina diferentes modelos de ML para prever o consumo de carbapenemes
utilizando features como dados temporais, população, urgências e consultas médicas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CarbapenemesPredictionModel:
    """
    Classe para treinar e avaliar modelos de previsão de consumo de carbapenemes
    """

    def __init__(self, data_path):
        """
        Inicializa o modelo com o caminho do dataset

        Args:
            data_path (str): Caminho para o ficheiro CSV com os dados
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None

    def load_and_prepare_data(self):
        """
        Carrega e prepara os dados para treino
        """
        print("📊 Carregando dados...")
        self.df = pd.read_csv(self.data_path, sep=';', decimal='.')

        print(f"   Dataset carregado: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
        print(f"   Período: {self.df['Periodo'].min()} a {self.df['Periodo'].max()}")
        print(f"   Regiões únicas: {self.df['Regiao'].nunique()}")
        print(f"   Instituições únicas: {self.df['Instituicao'].nunique()}")

        # Análise básica do consumo de carbapenemes
        print(f"\n📈 Estatísticas do Consumo de Carbapenemes:")
        print(f"   Média: {self.df['Consumo_Carbapenemes'].mean():.2f}")
        print(f"   Mediana: {self.df['Consumo_Carbapenemes'].median():.2f}")
        print(f"   Desvio padrão: {self.df['Consumo_Carbapenemes'].std():.2f}")
        print(f"   Registos com consumo > 0: {(self.df['Consumo_Carbapenemes'] > 0).sum()}")

    def engineer_features(self):
        """
        Cria features adicionais e prepara os dados para o modelo
        """
        print("\n🔧 Engenharia de features...")

        # Criar cópia para não modificar o original
        df_model = self.df.copy()

        # Remover linhas onde a variável alvo é NaN
        df_model = df_model.dropna(subset=['Consumo_Carbapenemes'])

        # Encoding de variáveis categóricas
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

        # Selecionar features para o modelo
        feature_columns = [
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

        # Verificar e remover features com valores ausentes
        available_features = [col for col in feature_columns if col in df_model.columns]

        X = df_model[available_features].copy()
        y = df_model['Consumo_Carbapenemes'].copy()

        # Preencher valores NaN com a mediana
        X = X.fillna(X.median())

        self.feature_names = available_features

        print(f"   Features criadas: {len(available_features)}")
        print(f"   Amostras totais: {len(X)}")

        return X, y

    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """
        Divide os dados em treino e teste e aplica normalização
        """
        print(f"\n📦 Divisão dos dados (treino: {100*(1-test_size):.0f}%, teste: {100*test_size:.0f}%)...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Normalizar features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"   Treino: {len(self.X_train)} amostras")
        print(f"   Teste: {len(self.X_test)} amostras")

    def train_models(self):
        """
        Treina diferentes modelos e compara o desempenho
        """
        print("\n🤖 Treinando modelos...")

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\n   Treinando {name}...")

            # Treinar modelo
            model.fit(self.X_train_scaled, self.y_train)

            # Fazer previsões
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)

            # Avaliar desempenho
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            test_mae = mean_absolute_error(self.y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train,
                cv=5, scoring='r2', n_jobs=-1
            )

            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test
            }

            print(f"      R² (treino): {train_r2:.4f}")
            print(f"      R² (teste): {test_r2:.4f}")
            print(f"      RMSE (teste): {test_rmse:.2f}")
            print(f"      MAE (teste): {test_mae:.2f}")
            print(f"      CV R² médio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return results

    def optimize_best_model(self, results):
        """
        Otimiza o melhor modelo usando GridSearch
        """
        print("\n🔍 Otimizando o melhor modelo...")

        # Selecionar melhor modelo baseado no R² de teste
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        print(f"   Modelo selecionado: {best_model_name}")

        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [5, 10]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        else:
            print(f"   Otimização não disponível para {best_model_name}")
            self.best_model = results[best_model_name]['model']
            return results[best_model_name]

        # Grid Search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=5, scoring='r2',
            n_jobs=-1, verbose=1
        )

        print("   Executando Grid Search (pode demorar alguns minutos)...")
        grid_search.fit(self.X_train_scaled, self.y_train)

        # Melhor modelo
        self.best_model = grid_search.best_estimator_

        # Avaliação final
        y_pred_test = self.best_model.predict(self.X_test_scaled)

        final_results = {
            'model': self.best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(self.y_train, self.best_model.predict(self.X_train_scaled)),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'predictions': y_pred_test
        }

        print(f"\n   ✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"   ✅ R² (treino): {final_results['train_r2']:.4f}")
        print(f"   ✅ R² (teste): {final_results['test_r2']:.4f}")
        print(f"   ✅ RMSE (teste): {final_results['test_rmse']:.2f}")
        print(f"   ✅ MAE (teste): {final_results['test_mae']:.2f}")

        return final_results

    def plot_results(self, results, final_results):
        """
        Cria visualizações dos resultados
        """
        print("\n📊 Gerando visualizações...")

        # 1. Comparação de modelos
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # R² scores
        model_names = list(results.keys())
        train_r2 = [results[name]['train_r2'] for name in model_names]
        test_r2 = [results[name]['test_r2'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        axes[0, 0].bar(x - width/2, train_r2, width, label='Treino', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='Teste', alpha=0.8)
        axes[0, 0].set_xlabel('Modelo')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Comparação de R² Score por Modelo')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # RMSE
        rmse_values = [results[name]['test_rmse'] for name in model_names]
        axes[0, 1].bar(model_names, rmse_values, color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('Modelo')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Root Mean Squared Error (Teste)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Predições vs Real (melhor modelo)
        axes[1, 0].scatter(self.y_test, final_results['predictions'], alpha=0.5)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()],
                       'r--', lw=2, label='Predição Perfeita')
        axes[1, 0].set_xlabel('Valor Real')
        axes[1, 0].set_ylabel('Valor Previsto')
        axes[1, 0].set_title(f'Predições vs Valores Reais (Melhor Modelo)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Resíduos
        residuals = self.y_test - final_results['predictions']
        axes[1, 1].scatter(final_results['predictions'], residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Valor Previsto')
        axes[1, 1].set_ylabel('Resíduo')
        axes[1, 1].set_title('Análise de Resíduos')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Gráfico de comparação salvo: model_comparison.png")

        # 2. Importância das features (se disponível)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()

    def plot_feature_importance(self):
        """
        Plota a importância das features para modelos baseados em árvores
        """
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20

        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Features Mais Importantes')
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Importância')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("   ✅ Gráfico de importância salvo: feature_importance.png")

    def save_model(self, filename='carbapenemes_model.pkl'):
        """
        Guarda o modelo treinado
        """
        import pickle

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Modelo guardado: {filename}")

    def predict_new_data(self, new_data):
        """
        Faz previsões em novos dados

        Args:
            new_data (pd.DataFrame): DataFrame com as mesmas features de treino

        Returns:
            np.array: Previsões de consumo de carbapenemes
        """
        new_data_scaled = self.scaler.transform(new_data[self.feature_names])
        predictions = self.best_model.predict(new_data_scaled)
        return predictions


def main():
    """
    Função principal para executar o pipeline completo
    """
    print("="*70)
    print("🏥 MODELO DE PREVISÃO DO CONSUMO DE CARBAPENEMES")
    print("="*70)

    # Caminho do dataset
    data_path = 'dataset_medicamentos_por_regiao.csv'

    # Inicializar modelo
    model = CarbapenemesPredictionModel(data_path)

    # 1. Carregar dados
    model.load_and_prepare_data()

    # 2. Engenharia de features
    X, y = model.engineer_features()

    # 3. Dividir e normalizar dados
    model.split_and_scale_data(X, y, test_size=0.2)

    # 4. Treinar múltiplos modelos
    results = model.train_models()

    # 5. Otimizar melhor modelo
    final_results = model.optimize_best_model(results)

    # 6. Visualizar resultados
    model.plot_results(results, final_results)

    # 7. Guardar modelo
    model.save_model('carbapenemes_best_model.pkl')

    print("\n" + "="*70)
    print("✅ TREINO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\n📌 Ficheiros gerados:")
    print("   - carbapenemes_best_model.pkl (modelo treinado)")
    print("   - model_comparison.png (comparação de modelos)")
    print("   - feature_importance.png (importância das features)")
    print("\n💡 Para fazer previsões, carregue o modelo com pickle e use o método predict_new_data()")


if __name__ == "__main__":
    main()
