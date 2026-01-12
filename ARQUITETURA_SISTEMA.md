# 🏥 Sistema Integrado de Previsão e Otimização do Consumo de Carbapenemes

## 📋 Resumo Executivo

Este projeto implementa um **sistema híbrido** que combina **Machine Learning** (previsão) com **Algoritmo Genético** (otimização) para resolver o problema da distribuição eficiente de antibióticos carbapenemes em instituições hospitalares portuguesas.

---

## 🎯 Problema Resolvido

### Desafio Real

- **Stock limitado** de carbapenemes (antibióticos críticos)
- **Necessidade de distribuir eficientemente** entre múltiplas instituições
- **Variação sazonal** e **regional** no consumo
- **Diferentes níveis de urgência** entre hospitais

### Solução Proposta

Sistema em **3 camadas** que integra:

1. **Previsão inteligente** (ML)
2. **Otimização multi-objetivo** (GA)
3. **Interface de decisão** (análise comparativa)

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA INTEGRADO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐         ┌─────────────────────┐        │
│  │   MOTOR ML         │         │   MOTOR GA          │        │
│  │   (Previsão)       │────────▶│   (Otimização)      │        │
│  │                    │         │                     │        │
│  │ • Random Forest    │         │ • Fitness Multi-    │        │
│  │ • Gradient Boost   │         │   objetivo          │        │
│  │ • Feature Eng.     │         │ • População: 30     │        │
│  │ • R² > 0.85        │         │ • Gerações: 300     │        │
│  └────────────────────┘         └─────────────────────┘        │
│           │                              │                      │
│           │                              │                      │
│           └──────────────┬───────────────┘                      │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │  INTERFACE DECISÃO    │                          │
│              │                       │                          │
│              │ • Análise Comparativa │                          │
│              │ • Visualizações       │                          │
│              │ • Cenários "What-if"  │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Componente 1: Motor de Previsão (ML)

### Objetivo

Prever o consumo futuro de carbapenemes para cada instituição num determinado mês.

### Metodologia

- **Algoritmo**: Gradient Boosting Regressor
- **Features**: 24 variáveis (temporais, populacionais, clínicas)
- **Métricas**: R² > 0.85, RMSE < 500

### Features Principais

```python
Temporais:       Ano, Mês, Trimestre, Semestre, Sazonalidade (sin/cos)
Geográficas:     Região, Instituição, População, Nº Municípios
Clínicas:        Urgências (Geral, Pediátricas, Obstétricas)
                 Consultas (Primeiras, Subsequentes)
Antibióticos:    Consumo de outros antibióticos
```

### Outputs

- **Previsão por instituição**: Consumo esperado em unidades
- **Intervalo de confiança**: ±10-15%
- **Agregação regional**: Total por região de saúde

---

## 🧬 Componente 2: Motor de Otimização (GA)

### Objetivo

Distribuir o stock disponível de forma ótima, considerando múltiplos objetivos.

### Função de Fitness Multi-Objetivo

```python
Fitness = w1 × (Minimizar erro de alocação)
        + w2 × (Penalizar situações críticas)
        + w3 × (Recompensar distribuição proporcional)
        - (Penalizar variância excessiva)

Onde:
w1 = 0.70  # Prioridade: satisfazer necessidade prevista
w2 = 0.20  # Prioridade: evitar colapso em instituições críticas
w3 = 0.10  # Prioridade: distribuição equilibrada
```

### Parâmetros do GA

- **População**: 30 soluções
- **Gerações**: 300 iterações
- **Crossover**: Single-point
- **Mutação**: 15% dos genes
- **Seleção**: Steady-state selection (SSS)

### Outputs

- **Alocação otimizada**: Unidades por instituição
- **Taxa de cobertura**: % da necessidade prevista atendida
- **Análise de défice**: Instituições em risco

---

## 📊 Componente 3: Interface de Decisão

### Funcionalidades

#### 1. Análise de Cenário Único

- Input: Mês, Ano, Stock disponível
- Output: Distribuição otimizada + métricas

#### 2. Comparação de Cenários

- Testa diferentes níveis de stock
- Identifica ponto de equilíbrio
- Análise de sensibilidade

#### 3. Visualizações

- Gráficos comparativos (Previsto vs Alocado)
- Mapas de calor regionais
- Taxa de cobertura por instituição
- Análise de défices críticos

---

## 📈 Métricas de Avaliação

### Métricas do ML

| Métrica  | Valor Esperado | Descrição              |
| -------- | -------------- | ---------------------- |
| R² Score | > 0.85         | Qualidade da previsão  |
| RMSE     | < 500          | Erro médio em unidades |
| MAE      | < 300          | Erro absoluto médio    |

### Métricas do GA

| Métrica               | Valor Esperado | Descrição                    |
| --------------------- | -------------- | ---------------------------- |
| Taxa Cobertura Média  | > 80%          | % das necessidades atendidas |
| Instituições Críticas | < 10%          | Com cobertura < 50%          |
| Desvio Padrão         | Minimizado     | Equilíbrio na distribuição   |

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pygad
```

### Execução Básica

```bash
python3 sistema_integrado_ml_ga.py
```

### Inputs Necessários

1. **Mês** (1-12): Para qual mês fazer a previsão
2. **Ano** (ex: 2024): Ano alvo
3. **Stock total**: Unidades disponíveis para distribuir

### Outputs Gerados

1. `otimizacao_final.csv` - Resultados detalhados
2. `sistema_integrado_resultados.png` - 6 visualizações

---

## 💡 Vantagens do Sistema Híbrido

### vs. Apenas ML

| Aspecto            | Apenas ML        | ML + GA      |
| ------------------ | ---------------- | ------------ |
| Previsão           | ✅ Ótima         | ✅ Ótima     |
| Distribuição       | ❌ Não otimizada | ✅ Otimizada |
| Restrição de Stock | ❌ Não considera | ✅ Considera |
| Multi-objetivo     | ❌ Não           | ✅ Sim       |

### vs. Apenas GA

| Aspecto            | Apenas GA           | ML + GA        |
| ------------------ | ------------------- | -------------- |
| Previsão Futura    | ❌ Só usa histórico | ✅ Aprendizado |
| Adaptação          | ❌ Regras fixas     | ✅ Dinâmica    |
| Precisão           | ⚠️ Moderada         | ✅ Alta        |
| Features Complexas | ❌ Limitado         | ✅ Avançado    |

---

## 🎓 Contribuição Científica

### Originalidade

1. **Integração inédita**: ML + GA para distribuição hospitalar
2. **Multi-objetivo**: 3 objetivos simultâneos (necessidade, equidade, eficiência)
3. **Sazonalidade**: Features temporais avançadas (sin/cos encoding)
4. **Validação cruzada**: Sistema testado em dados reais portugueses

### Aplicabilidade

- **Saúde Pública**: Gestão de recursos escassos
- **Logística Hospitalar**: Planeamento de stocks
- **Políticas Públicas**: Decisões baseadas em evidência
- **Extensível**: Adaptável a outros medicamentos/recursos

---

## 📚 Referências Técnicas

### Machine Learning

- Gradient Boosting: Chen & Guestrin (2016) - XGBoost
- Feature Engineering: Kuhn & Johnson (2019) - Feature Engineering and Selection

### Algoritmos Genéticos

- Multi-objective GA: Deb et al. (2002) - NSGA-II
- Fitness Function Design: Coello et al. (2007)

### Aplicação em Saúde

- Healthcare Resource Allocation: Bertsimas et al. (2020)
- Antibiotic Stewardship: WHO Guidelines (2023)

---

## 👥 Autores

**Projeto de Mestrado**
Universidade [Nome]
Curso: [Curso]
Orientador: [Nome do Professor]

---

## 📞 Suporte

Para questões técnicas ou sugestões:

- Email: [teu_email]
- GitHub: [teu_repo]

---

## 📄 Licença

Este projeto é desenvolvido para fins académicos.

---

## ✅ Checklist de Entrega

- [x] Código ML (train_carbapenemes_model.py)
- [x] Código GA (GA_code.py)
- [x] **Sistema Integrado** (sistema_integrado_ml_ga.py) ⭐
- [x] Documentação de arquitetura
- [x] Visualizações
- [ ] Relatório final (a entregar)
- [ ] Apresentação PowerPoint

---

**Última atualização**: Janeiro 2026
