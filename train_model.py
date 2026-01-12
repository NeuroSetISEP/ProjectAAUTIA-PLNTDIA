#!/usr/bin/env python3
"""
Script para treinar o modelo ML de previsão de carbapenemes
"""
import sys
from pathlib import Path

# Adicionar backend ao path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from ml_models import MLPredictionModel

# Caminhos
DATA_PATH = "dataset_forecast_preparado.csv"
MODEL_OUTPUT = "backend/models/trained_model.pkl"

def main():
    print("=" * 70)
    print("🚀 SNS AI - TREINO DO MODELO ML")
    print("=" * 70)

    # 1. Criar instância do modelo
    model = MLPredictionModel(data_path=DATA_PATH)

    # 2. Carregar dados
    print("\n📂 Carregando dados...")
    if not model.load_data():
        print("❌ Erro ao carregar dados!")
        return

    print(f"✅ Dataset carregado com sucesso")
    print(f"   - Registros: {len(model.df)}")
    print(f"   - Hospitais: {model.df['Instituicao'].nunique()}")
    print(f"   - Regiões: {model.df['Regiao'].nunique()}")

    # 3. Engenharia de features
    print("\n🛠️  Preparando features...")
    X, y = model.engineer_features()
    print(f"✅ Features preparadas:")
    print(f"   - Amostras: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Target range: [{y.min():.1f}, {y.max():.1f}]")

    # 4. Treinar modelos
    print("\n🤖 Treinando modelos candidatos...")
    print("   (RandomForest, GradientBoosting, Ridge)")
    results = model.train_model(X, y)

    # 5. Mostrar resultados
    print("\n📊 RESULTADOS DO TREINO:")
    print("-" * 70)
    for name, metrics in results.items():
        r2_train = metrics['r2_train']
        r2_test = metrics['r2_test']
        overfit = abs(r2_train - r2_test)

        emoji = "🏆" if name == type(model.best_model).__name__ else "  "
        print(f"{emoji} {name:20s} | R² Train: {r2_train:.4f} | R² Test: {r2_test:.4f} | Overfit: {overfit:.4f}")

    print("-" * 70)
    print(f"✅ Melhor modelo: {type(model.best_model).__name__}")

    # 6. Salvar modelo
    print(f"\n💾 Salvando modelo em: {MODEL_OUTPUT}")

    # Criar diretório se não existir
    Path(MODEL_OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_OUTPUT)

    print("\n" + "=" * 70)
    print("✅ TREINO COMPLETO!")
    print("=" * 70)
    print("\n💡 Próximos passos:")
    print("   1. Reinicia o backend: ./start-system.sh")
    print("   2. O novo modelo será carregado automaticamente")
    print("   3. Verifica as métricas em: http://localhost:3000/metrics")

if __name__ == "__main__":
    main()
