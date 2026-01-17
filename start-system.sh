#!/bin/bash

# Script de inicialização completa do projeto SNS AI
# Executa backend e frontend simultaneamente

echo "🚀 Iniciando Sistema SNS AI Completo..."
echo "========================================="

# Verificar se estamos no diretório correto
if [ ! -f "GA_code.py" ]; then
    echo "❌ Execute este script a partir do diretório raiz do projeto"
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p logs

# Treinar modelo otimizado antes de iniciar serviços

# Verificar se o modelo já está treinado
MODEL_PATH="backend/models/trained_model.pkl"
if [ -f "$MODEL_PATH" ]; then
    echo "🧠 Modelo otimizado já existe em $MODEL_PATH. Pulando treinamento."
else
    echo "🧠 Treinando modelo otimizado de previsão (train_optimized_model.py)..."
    if python3 train_optimized_model.py; then
        echo "✅ Modelo treinado com sucesso."
    else
        echo "❌ Erro ao treinar modelo otimizado. Verifique o script train_optimized_model.py."
        exit 1
    fi
fi

# Função para cleanup ao sair
cleanup() {
    echo ""
    echo "🛑 Parando serviços..."
    # Matar processos em background
    jobs -p | xargs -r kill
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Setup do Backend
echo "📡 Configurando Backend..."
cd backend

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual para backend..."
    python3 -m venv venv
fi

# Ativar ambiente virtual e instalar dependências
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

echo "📚 Atualizando pip e setuptools..."
pip install --upgrade pip setuptools wheel

echo "📚 Instalando dependências do backend..."
if ! pip install --upgrade -r requirements.txt; then
    echo "❌ Erro ao instalar dependências do backend"
    exit 1
fi

echo "✅ Backend configurado"

# Verificar se o Python consegue importar as dependências principais
echo "🔍 Verificando importações Python..."
if ! python -c "import fastapi, uvicorn, pandas, numpy, sklearn"; then
    echo "❌ Erro nas importações Python"
    exit 1
fi

# Iniciar backend em background com ambiente virtual
echo "🔧 Iniciando Backend API (http://localhost:8000)..."
source venv/bin/activate && python main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!

cd ..

# 2. Setup do Frontend
echo "💻 Configurando Frontend..."
cd frontend

# Verificar e ativar Node.js LTS via NVM
if command -v nvm &> /dev/null; then
    echo "🔧 Usando Node.js LTS via NVM..."
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm use --lts > /dev/null 2>&1
    echo "📊 Node.js ativo: $(node --version)"
else
    echo "⚠️  NVM não encontrado, usando Node.js padrão: $(node --version)"
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo "❌ Node.js $NODE_VERSION não suportado. Necessário Node.js 18+"
        echo "💡 Instale NVM e execute: nvm use --lts"
        exit 1
    fi
fi

# Verificar se npm está funcionando
if ! command -v npm &> /dev/null; then
    echo "❌ npm não encontrado"
    exit 1
fi

# Verificar se node_modules existe e instalar dependências
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
    echo "📦 Instalando dependências do frontend..."
    if ! npm install; then
        echo "❌ Erro ao instalar dependências do frontend"
        exit 1
    fi
else
    echo "📦 Dependências do frontend já instaladas"
fi

# Verificar se vite está disponível
if ! npx vite --version > /dev/null 2>&1; then
    echo "❌ Vite não está disponível. Tentando com npm run dev..."
    if ! npm run dev --dry-run > /dev/null 2>&1; then
        echo "❌ npm run dev também falhou. Reinstalando dependências..."
        rm -rf node_modules package-lock.json
        if ! npm install; then
            echo "❌ Erro crítico na instalação das dependências"
            exit 1
        fi
    fi
fi

echo "✅ Frontend configurado"

# Aguardar backend estar pronto
echo "⏳ Aguardando backend inicializar..."
sleep 5

# Verificar se backend está respondendo
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend respondendo em http://localhost:8000"
        break
    elif [ $i -eq 10 ]; then
        echo "❌ Backend não está respondendo após 50 segundos"
        echo "📋 Verifique os logs em logs/backend.log:"
        tail -20 ../logs/backend.log 2>/dev/null || echo "Logs não encontrados"
        exit 1
    else
        echo "⏳ Tentativa $i/10 - aguardando backend..."
        sleep 5
    fi
done

# Iniciar frontend
echo "🚀 Iniciando Frontend (http://localhost:3000)..."

# Tentar com npx vite primeiro, se falhar usar npm run dev
if npx vite dev --port 3000 --host > ../logs/frontend.log 2>&1 &
then
    echo "🔧 Frontend iniciado com: npx vite dev --port 3000"
    FRONTEND_PID=$!
else
    echo "⚠️  npx vite falhou, tentando npm run dev..."
    npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "🔧 Frontend iniciado com: npm run dev"
fi

cd ..

# Criar diretório de logs se não existir
mkdir -p logs

echo ""
echo "🎉 Sistema SNS AI iniciado com sucesso!"
echo "========================================="
echo ""
echo "🌐 Serviços disponíveis:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📋 Funcionalidades:"
echo "   ✅ Previsão ML de consumo de medicamentos"
echo "   ✅ Otimização genética para distribuição"
echo "   ✅ Interface web interativa"
echo "   ✅ Dashboard com analytics"
echo ""
echo "📖 Logs:"
echo "   Backend:  tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo "⌨️  Pressione Ctrl+C para parar todos os serviços"
echo ""

 # Aguardar os processos em background
echo "📊 Processos ativos:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""

# Aguardar qualquer processo ou interrupção
wait $BACKEND_PID $FRONTEND_PID