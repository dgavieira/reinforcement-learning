import matplotlib.pyplot as plt
import numpy as np

def plot_returns(policy1_returns, policy2_returns, out_file="returns_plot.png"):
    """Plota a comparação entre as duas políticas do CartPole.
    
    Args:
        policy1_returns: Retornos da Política 1
        policy2_returns: Retornos da Política 2  
        out_file: Nome do arquivo de saída
    """
    plt.figure(figsize=(10, 6))
    
    episodes = range(1, len(policy1_returns) + 1)
    
    plt.plot(episodes, policy1_returns, 'o-', label="Política 1 (Determinística)", 
             color='blue', linewidth=2, markersize=6)
    plt.plot(episodes, policy2_returns, 's-', label="Política 2 (REINFORCE)", 
             color='red', linewidth=2, markersize=6)
    
    # Adiciona linhas horizontais para as médias
    plt.axhline(y=np.mean(policy1_returns), color='blue', linestyle='--', alpha=0.7,
                label=f'Média P1: {np.mean(policy1_returns):.1f}')
    plt.axhline(y=np.mean(policy2_returns), color='red', linestyle='--', alpha=0.7,
                label=f'Média P2: {np.mean(policy2_returns):.1f}')
    
    plt.xlabel("Episódio", fontsize=12)
    plt.ylabel("Número de Tentativas Bem-sucedidas", fontsize=12)
    plt.title("Comparação das Políticas no Problema do CartPole", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, len(policy1_returns) + 0.5)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nGráfico salvo em: {out_file}")
