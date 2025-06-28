import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def criar_diagrama_pipeline():
    """Cria um diagrama visual do pipeline de dados"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Cores para diferentes tipos de etapas
    cores = {
        'dados': '#E3F2FD',      # Azul claro
        'processo': '#FFF3E0',    # Laranja claro
        'modelo': '#E8F5E8',      # Verde claro
        'output': '#F3E5F5'       # Roxo claro
    }
    
    # Função para criar caixas
    def criar_caixa(x, y, width, height, texto, cor, ax):
        caixa = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.1",
                              facecolor=cor,
                              edgecolor='black',
                              linewidth=1.5)
        ax.add_patch(caixa)
        ax.text(x + width/2, y + height/2, texto,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                wrap=True)
    
    # Função para criar setas
    def criar_seta(x1, y1, x2, y2, ax):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # TÍTULO
    ax.text(5, 13.5, 'PIPELINE DE MACHINE LEARNING', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # ETAPA 1: DADOS BRUTOS
    criar_caixa(0.5, 11.5, 2, 1.5, 
               'DADOS BRUTOS\n(PostgreSQL)\n500 registros\nVendas 2024', 
               cores['dados'], ax)
    
    # ETAPA 2: PRÉ-PROCESSAMENTO
    criar_seta(2.5, 12.25, 3.5, 12.25, ax)
    criar_caixa(3.5, 11.5, 2.5, 1.5, 
               'PRÉ-PROCESSAMENTO\n• Conversão de datas\n• Feature Engineering\n• Limpeza dados', 
               cores['processo'], ax)
    
    # FEATURE ENGINEERING (DETALHADO)
    criar_seta(5.75, 11.5, 5.75, 10.5, ax)
    criar_caixa(4.5, 9, 2.5, 1.5, 
               'FEATURE ENGINEERING\n• Ano, mês, trimestre\n• Dia da semana\n• Faixas de quantidade\n• Target alta_venda', 
               cores['processo'], ax)
    
    # DADOS PROCESSADOS
    criar_seta(5.75, 9, 5.75, 8, ax)
    criar_caixa(4.5, 6.5, 2.5, 1.5, 
               'DADOS PROCESSADOS\n• Features numéricas: 6\n• Features categóricas: 3\n• 2 targets definidos', 
               cores['dados'], ax)
    
    # DIVISÃO DOS DADOS
    criar_seta(5.75, 6.5, 5.75, 5.5, ax)
    criar_caixa(4.5, 4, 2.5, 1.5, 
               'DIVISÃO TREINO/TESTE\n• 80% Treino (400)\n• 20% Teste (100)\n• Stratificação', 
               cores['processo'], ax)
    
    # BRANCH PARA REGRESSÃO
    criar_seta(4.5, 4.75, 2, 4.75, ax)
    criar_caixa(0.5, 4, 2, 1.5, 
               'REGRESSÃO\nTarget: valor_total\nPredict: R$ vendas', 
               cores['modelo'], ax)
    
    # BRANCH PARA CLASSIFICAÇÃO
    criar_seta(7, 4.75, 8.5, 4.75, ax)
    criar_caixa(8.5, 4, 2, 1.5, 
               'CLASSIFICAÇÃO\nTarget: alta_venda\nPredict: 0 ou 1', 
               cores['modelo'], ax)
    
    # MODELOS DE REGRESSÃO
    criar_seta(1.5, 4, 1.5, 3, ax)
    criar_caixa(0.5, 1.5, 2, 1.5, 
               'MODELOS REGRESSÃO\n• Linear Regression\n• Random Forest\nMétricas: MSE, R²', 
               cores['modelo'], ax)
    
    # MODELOS DE CLASSIFICAÇÃO
    criar_seta(9.5, 4, 9.5, 3, ax)
    criar_caixa(8.5, 1.5, 2, 1.5, 
               'MODELOS CLASSIFICAÇÃO\n• Logistic Regression\n• Random Forest\nMétricas: Acc, F1', 
               cores['modelo'], ax)
    
    # MODELOS SALVOS
    criar_seta(1.5, 1.5, 1.5, 0.5, ax)
    criar_seta(9.5, 1.5, 9.5, 0.5, ax)
    criar_caixa(4, 0, 3, 1, 
               'MODELOS SALVOS (.pkl)\n4 modelos prontos para produção', 
               cores['output'], ax)
    
    # LEGENDA
    legenda_elementos = [
        mpatches.Patch(color=cores['dados'], label='Dados'),
        mpatches.Patch(color=cores['processo'], label='Processamento'),
        mpatches.Patch(color=cores['modelo'], label='Modelagem'),
        mpatches.Patch(color=cores['output'], label='Output')
    ]
    ax.legend(handles=legenda_elementos, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('pipeline_ml_vendas.png', dpi=300, bbox_inches='tight')
    plt.show()

def mostrar_fluxo_dados():
    """Mostra o fluxo detalhado dos dados"""
    print("📊 FLUXO DETALHADO DOS DADOS NO PIPELINE")
    print("=" * 60)
    
    fluxo = [
        {
            "etapa": "1. DADOS BRUTOS",
            "entrada": "PostgreSQL (tabela vendas)",
            "processamento": "Extração SQL",
            "saida": "DataFrame com 9 colunas",
            "formato": "500 registros × 9 features"
        },
        {
            "etapa": "2. PRÉ-PROCESSAMENTO", 
            "entrada": "DataFrame bruto",
            "processamento": "Conversão datas + Feature Engineering",
            "saida": "DataFrame enriquecido",
            "formato": "500 registros × 15 features"
        },
        {
            "etapa": "3. PREPARAÇÃO FEATURES",
            "entrada": "DataFrame processado", 
            "processamento": "Seleção features + targets",
            "saida": "X (features) + y (targets)",
            "formato": "X: 9 features, y: 2 targets"
        },
        {
            "etapa": "4. DIVISÃO TREINO/TESTE",
            "entrada": "X e y completos",
            "processamento": "train_test_split (80/20)",
            "saida": "4 conjuntos de dados",
            "formato": "Train: 400, Test: 100"
        },
        {
            "etapa": "5. TREINAMENTO",
            "entrada": "Dados de treino",
            "processamento": "Fit de 4 modelos + Pipeline",
            "saida": "Modelos treinados",
            "formato": "4 modelos com preprocessor"
        },
        {
            "etapa": "6. AVALIAÇÃO",
            "entrada": "Modelos + dados teste",
            "processamento": "Predição + cálculo métricas", 
            "saida": "Métricas de performance",
            "formato": "MSE, R², Accuracy, F1"
        },
        {
            "etapa": "7. PERSISTÊNCIA",
            "entrada": "Modelos treinados",
            "processamento": "Serialização joblib",
            "saida": "Arquivos .pkl",
            "formato": "4 arquivos modelo_*.pkl"
        }
    ]
    
    for i, etapa in enumerate(fluxo, 1):
        print(f"\n🔸 {etapa['etapa']}")
        print(f"   📥 Entrada:      {etapa['entrada']}")
        print(f"   ⚙️  Processamento: {etapa['processamento']}")
        print(f"   📤 Saída:        {etapa['saida']}")
        print(f"   📊 Formato:      {etapa['formato']}")
        
        if i < len(fluxo):
            print("        ⬇️")

def mostrar_metricas_esperadas():
    """Mostra as métricas que serão calculadas"""
    print("\n📈 MÉTRICAS DE AVALIAÇÃO DOS MODELOS")
    print("=" * 50)
    
    print("\n🔹 MODELOS DE REGRESSÃO (predizer valor_total):")
    print("   • MSE (Mean Squared Error) - quanto menor, melhor")
    print("   • RMSE (Root MSE) - erro em unidades originais (R$)")  
    print("   • R² (Coeficiente determinação) - quanto mais próximo de 1, melhor")
    
    print("\n🔸 MODELOS DE CLASSIFICAÇÃO (predizer alta_venda):")
    print("   • Accuracy - % acertos total")
    print("   • Precision - % acertos entre predições positivas") 
    print("   • Recall - % captura dos casos positivos reais")
    print("   • F1-Score - média harmônica precision/recall")

if __name__ == "__main__":
    print("🎨 VISUALIZAÇÃO DO PIPELINE DE MACHINE LEARNING")
    print("=" * 60)
    
    # Mostrar fluxo textual
    mostrar_fluxo_dados()
    
    # Mostrar métricas
    mostrar_metricas_esperadas()
    
    # Criar diagrama visual
    print("\n🖼️ Gerando diagrama visual do pipeline...")
    try:
        criar_diagrama_pipeline()
        print("✅ Diagrama salvo como 'pipeline_ml_vendas.png'")
    except Exception as e:
        print(f"⚠️ Erro ao criar diagrama: {e}")