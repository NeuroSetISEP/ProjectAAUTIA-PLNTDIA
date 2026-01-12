# SNS AI - Sistema de Distribuição Inteligente de Medicamentos

🤖 **Sistema completo com backend FastAPI e frontend React para otimização de distribuição de carbapenemes usando Machine Learning e Algoritmos Genéticos.**

## ✨ Funcionalidades Principais

### 🧠 **Backend Inteligente (Novo!)**

- ✅ **API FastAPI** com endpoints RESTful
- ✅ **Machine Learning** para predição de consumo
- ✅ **Algoritmo Genético** para otimização de distribuição
- ✅ **Modo Híbrido**: Funciona com dados reais ou mock
- ✅ **Health Checks** e monitoramento

### 🌐 **Frontend Interativo**

- ✅ **React + TypeScript** moderno
- ✅ **Interface intuitiva** para configuração
- ✅ **Dashboard analytics** com visualizações
- ✅ **Integração completa** com backend
- ✅ **Feedback visual** do status da conexão

### 📊 **Análise Avançada**

- ✅ **Predição temporal** de consumo hospitalar
- ✅ **Otimização multiobjetivo** (necessidade vs desperdício)
- ✅ **Priorização inteligente** baseada em urgências e população
- ✅ **Visualização interativa** dos resultados

## 🚀 Inicialização Rápida

### Método 1: Script Automático (Recomendado)

```bash
# Executar o sistema completo
./start-system.sh
```

Isso irá:

- ✅ Configurar backend Python automaticamente
- ✅ Configurar frontend React
- ✅ Iniciar ambos os serviços
- ✅ Mostrar logs em tempo real

### Método 2: Manual

#### Backend

```bash
cd backend
./setup.sh                    # Configuração inicial
source venv/bin/activate       # Ativar ambiente
python main.py                 # Iniciar API
```

#### Frontend

```bash
cd frontend
npm install                    # Instalar dependências
npm run dev                    # Iniciar desenvolvimento
```

## 📋 URLs do Sistema

| Serviço          | URL                          | Descrição               |
| ---------------- | ---------------------------- | ----------------------- |
| **Frontend**     | http://localhost:8080        | Interface principal     |
| **Backend API**  | http://localhost:8000        | API REST                |
| **Documentação** | http://localhost:8000/docs   | Swagger docs interativa |
| **Health Check** | http://localhost:8000/health | Status do sistema       |

## 🏗️ Arquitetura

```
ProjectAAUTIA-PLNTDIA/
├── 🗂️ backend/               # API FastAPI
│   ├── main.py              # Servidor principal
│   ├── ml_models.py         # Modelos ML refatorados
│   ├── models/              # Modelos treinados
│   └── requirements.txt     # Dependências Python
├── 🌐 frontend/              # Interface React
│   ├── src/
│   │   ├── services/api.ts  # Cliente API
│   │   ├── pages/           # Páginas da aplicação
│   │   └── components/      # Componentes UI
├── 📊 Data Files             # Datasets originais
├── 🤖 GA_code.py             # Algoritmo original
└── 📈 *.csv                  # Dados hospitalares
```

## 📡 Endpoints da API

### 🔍 Predição

```http
POST /predict
{
  "month": 3,
  "year": 2026,
  "stock_percentage": 0.7
}
```

### ⚡ Otimização

```http
POST /distribute
{
  "months": [3, 4, 5],
  "year": 2026,
  "stock_percentage": 0.8,
  "mode": "quarter"
}
```

### 🏥 Hospitais

```http
GET /hospitals
```

## 🛠️ Melhorias Implementadas

### 🔄 **Interatividade**

- **Antes**: Scripts Python isolados
- **Agora**: Sistema web completo com tempo real

### 🎯 **Usabilidade**

- **Antes**: Linha de comando técnica
- **Agora**: Interface gráfica intuitiva

### 📈 **Escalabilidade**

- **Antes**: Processamento local
- **Agora**: Arquitetura client-server

### 🔧 **Configurabilidade**

- **Antes**: Parâmetros hardcoded
- **Agora**: Interface para ajustar todas as variáveis

### 📊 **Visualização**

- **Antes**: Prints simples no terminal
- **Agora**: Dashboard com gráficos e tabelas

## 🔮 Próximos Passos Sugeridos

### 🚀 **Produção**

1. **Docker**: Containerização para deploy
2. **Database**: PostgreSQL para persistência
3. **Cache**: Redis para performance
4. **Auth**: Autenticação e autorização

### 📈 **Analytics**

5. **Logs**: Sistema de auditoria completo
6. **Metrics**: Monitoramento em tempo real
7. **Alerts**: Notificações automáticas
8. **Reports**: Relatórios PDF automáticos

### 🤖 **IA Avançada**

9. **Deep Learning**: Modelos mais sofisticados
10. **Real-time**: Predições em tempo real
11. **AutoML**: Retreinamento automático
12. **Ensemble**: Combinação de múltiplos modelos

## 📞 Suporte

- 📧 **Logs**: Verifique `logs/backend.log` e `logs/frontend.log`
- 🔍 **Debug**: Use http://localhost:8000/docs para testar API
- 🛠️ **Issues**: Backend usa modo mock se dados ML não estiverem disponíveis

---

🎉 **O sistema agora é completamente interativo e pronto para uso profissional!**
