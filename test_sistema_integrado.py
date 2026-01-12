"""
Script de Teste Rápido do Sistema Integrado
Executa o sistema com valores pré-definidos para validação
"""

import sys
import os

# Simular inputs do utilizador
class MockInput:
    def __init__(self, inputs):
        self.inputs = iter(inputs)

    def __call__(self, prompt=''):
        print(prompt, end='')
        value = next(self.inputs)
        print(value)
        return value

# Substituir input() temporariamente
original_input = input
sys.modules['builtins'].input = MockInput([
    '6',      # Mês: Junho
    '2024',   # Ano: 2024
    '500000', # Stock: 500,000 unidades
    'n'       # Não comparar cenários (para teste rápido)
])

# Importar e executar o sistema
try:
    print("="*70)
    print("🧪 TESTE DO SISTEMA INTEGRADO")
    print("="*70)
    print("\n📋 Valores de teste:")
    print("   - Mês: 6 (Junho)")
    print("   - Ano: 2024")
    print("   - Stock: 500,000 unidades")
    print("   - Comparação de cenários: Não\n")

    from sistema_integrado_ml_ga import main
    main()

    print("\n" + "="*70)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*70)

except Exception as e:
    print(f"\n❌ ERRO NO TESTE: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restaurar input original
    sys.modules['builtins'].input = original_input
