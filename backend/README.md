# SNS AI - Backend

API FastAPI para o sistema de distribuição otimizada de medicamentos.

## Funcionalidades

- 🤖 **Previsão ML**: Predição de consumo de carbapenemes usando modelos avançados
- 🧬 **Otimização Genética**: Distribuição otimizada usando algoritmos genéticos
- 📊 **Analytics**: Estatísticas e insights dos hospitais
- 🔄 **Modo Híbrido**: Funciona com ML real ou dados mock
- 🌐 **API RESTful**: Endpoints completos para integração frontend

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar servidor
python main.py
# ou
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints Principais

### 🔍 Previsão

```
POST /predict
{
  "month": 3,
  "year": 2026,
  "stock_percentage": 0.7
}
```

### ⚡ Otimização

```
POST /distribute
{
  "months": [3, 4, 5],
  "year": 2026,
  "stock_percentage": 0.8,
  "mode": "quarter"
}
```

### 🏥 Hospitais

```
GET /hospitals
```

### 💚 Health Check

```
GET /health
```

## Integração com Frontend

O backend está configurado para trabalhar diretamente com o frontend React:

- CORS habilitado para `localhost:8080`
- Modelos de dados compatíveis
- Respostas otimizadas para UI

## Arquitetura

```
backend/
├── main.py              # API principal
├── models/              # Modelos Pydantic
├── services/            # Lógica de negócio
├── utils/               # Utilitários
└── requirements.txt     # Dependências
```

## Modo de Desenvolvimento

O sistema funciona em dois modos:

- **ML Mode**: Usa os modelos reais do `GA_code.py`
- **Mock Mode**: Dados simulados para desenvolvimento

## Próximos Passos

1. ✅ Backend FastAPI funcionando
2. 🔄 Integração com frontend (próximo)
3. 📈 Cache e performance
4. 🔐 Autenticação
5. 📊 Logs e monitoramento
