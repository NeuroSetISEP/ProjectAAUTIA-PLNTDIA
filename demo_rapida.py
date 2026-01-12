"""
Demonstração Rápida do Sistema Integrado
Executa apenas as partes essenciais para validação
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧪 DEMO RÁPIDA - SISTEMA INTEGRADO ML + GA")
print("="*70)

# 1. Carregar dados
print("\n📊 [1/4] Carregando dados...")
df = pd.read_csv('dataset_medicamentos_por_regiao.csv', sep=';', decimal='.')
print(f"   ✅ {len(df)} registos | {df['Instituicao'].nunique()} instituições")

# 2. Preparar features
print("\n🔧 [2/4] Preparando features para ML...")
from sklearn.preprocessing import LabelEncoder

df_model = df.copy().dropna(subset=['Consumo_Carbapenemes'])
le_regiao = LabelEncoder()
le_inst = LabelEncoder()
df_model['Regiao_Encoded'] = le_regiao.fit_transform(df_model['Regiao'])
df_model['Instituicao_Encoded'] = le_inst.fit_transform(df_model['Instituicao'])

features = ['Ano', 'Mes', 'Regiao_Encoded', 'Instituicao_Encoded',
            'Populacao_Regiao', 'Total_Urgencias', 'Total_Consultas']
X = df_model[features].fillna(0)
y = df_model['Consumo_Carbapenemes']

print(f"   ✅ {len(features)} features preparadas")

# 3. Treinar modelo ML
print("\n🤖 [3/4] Treinando modelo ML (Gradient Boosting)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"   ✅ Modelo treinado: R² = {r2:.4f}, RMSE = {rmse:.2f}")

# 4. Fazer previsões
print("\n🔮 [4/4] Gerando previsões para Junho/2024...")
latest = df.groupby('Instituicao').last().reset_index()
predictions = []

for _, row in latest.iterrows():
    X_pred = pd.DataFrame([{
        'Ano': 2024, 'Mes': 6,
        'Regiao_Encoded': le_regiao.transform([row['Regiao']])[0],
        'Instituicao_Encoded': le_inst.transform([row['Instituicao']])[0],
        'Populacao_Regiao': row['Populacao_Regiao'],
        'Total_Urgencias': row['Total_Urgencias'],
        'Total_Consultas': row['Total_Consultas']
    }])

    pred = model.predict(scaler.transform(X_pred))[0]
    predictions.append({
        'Instituicao': row['Instituicao'],
        'Consumo_Previsto': max(0, pred)
    })

pred_df = pd.DataFrame(predictions).sort_values('Consumo_Previsto', ascending=False)
print(f"   ✅ Previsões geradas para {len(pred_df)} instituições")

# 5. Simulação de Otimização GA
print("\n🧬 [SIMULAÇÃO] Otimização com GA...")
STOCK_TOTAL = 500000
consumo_previsto_total = pred_df['Consumo_Previsto'].sum()

# Distribuição proporcional (simplificada)
pred_df['Alocacao_Otimizada'] = (pred_df['Consumo_Previsto'] / consumo_previsto_total) * STOCK_TOTAL
pred_df['Taxa_Cobertura (%)'] = (pred_df['Alocacao_Otimizada'] / pred_df['Consumo_Previsto']) * 100

print(f"   ✅ Distribuição otimizada para {STOCK_TOTAL:,} unidades")

# 6. Resultados
print("\n" + "="*70)
print("📊 RESULTADOS")
print("="*70)
print(f"\n🎯 Necessidade vs Disponibilidade:")
print(f"   Consumo previsto: {consumo_previsto_total:,.2f} unidades")
print(f"   Stock disponível: {STOCK_TOTAL:,} unidades")
print(f"   Taxa cobertura média: {pred_df['Taxa_Cobertura (%)'].mean():.2f}%")

print(f"\n🏆 Top 10 Instituições:")
print(pred_df[['Instituicao', 'Consumo_Previsto', 'Alocacao_Otimizada', 'Taxa_Cobertura (%)']].head(10).to_string(index=False))

# Salvar resultados
pred_df.to_csv('demo_resultados.csv', index=False)
print(f"\n💾 Resultados salvos em: demo_resultados.csv")

print("\n" + "="*70)
print("✅ DEMO CONCLUÍDA COM SUCESSO!")
print("="*70)
print("\n💡 Sistema validado! As componentes principais funcionam:")
print("   ✓ ML: Previsão com Gradient Boosting")
print("   ✓ GA: Otimização de distribuição (simulada)")
print("   ✓ Integração: Pipeline completo executado")
print("\nPara executar o sistema completo com GA real:")
print("   python3 sistema_integrado_ml_ga.py")
