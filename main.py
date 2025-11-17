from cartpole.evaluate import evaluate_policy_1, evaluate_trained_policy
from cartpole.policy2_reinforce import train_policy_network
from cartpole.plot import plot_returns
from qlearning.run_qlearning import run_all_qlearning, print_results
import numpy as np

def main():
    print("\n" + "="*60)
    print("TRABALHO 5: APRENDIZADO POR REFORÇO")
    print("="*60)

    # ---------------------
    # Parte 1 — CartPole
    # ---------------------
    print("\n" + "="*40)
    print("PARTE 1: PROBLEMA DO CARTPOLE")
    print("="*40)
    
    print("\nComparando duas políticas para manter a haste na vertical:")
    print("- Política 1: Acelera para esquerda/direita baseado na inclinação do poste")
    print("- Política 2: Rede neural treinada com algoritmo REINFORCE")
    print("\nCada política será avaliada em 10 episódios de até 500 tentativas.")
    
    # Avalia Política 1
    print("\n" + "-"*30)
    policy1_returns = evaluate_policy_1(episodes=10, max_steps=500)
    
    print("\n📊 RESULTADOS - POLÍTICA 1 (Determinística):")
    print(f"   Média: {np.mean(policy1_returns):.2f} tentativas")
    print(f"   Desvio padrão: {np.std(policy1_returns):.2f}")
    print(f"   Valores: {policy1_returns}")
    
    # Treina e avalia Política 2
    print("\n" + "-"*30)
    policy2, training_returns = train_policy_network(episodes=10, max_steps=500, use_gpu=True)
    
    # Avalia a política treinada separadamente
    policy2_returns = evaluate_trained_policy(policy2, episodes=10, max_steps=500)
    
    print("\n📊 RESULTADOS - POLÍTICA 2 (REINFORCE):")
    print(f"   Média: {np.mean(policy2_returns):.2f} tentativas")
    print(f"   Desvio padrão: {np.std(policy2_returns):.2f}")
    print(f"   Valores: {policy2_returns}")
    
    # Comparação
    print("\n📈 COMPARAÇÃO:")
    if np.mean(policy1_returns) > np.mean(policy2_returns):
        print("   → Política 1 obteve melhor desempenho médio")
    else:
        print("   → Política 2 obteve melhor desempenho médio")
    
    print(f"   → Diferença na média: {abs(np.mean(policy1_returns) - np.mean(policy2_returns)):.2f} tentativas")

    # Gera gráfico
    plot_returns(policy1_returns, policy2_returns)

    # ---------------------
    # Parte 2 — Q-Learning
    # ---------------------
    print("\n" + "="*40)
    print("PARTE 2: PROBLEMA DO ROBÔ (Q-LEARNING)")
    print("="*40)
    
    print("\nProblema: Robô de reciclagem que coleta latas com bateria recarregável")
    print("Estados: high (bateria alta), low (bateria baixa)")  
    print("Ações: search (procurar latas), wait (esperar), recharge (recarregar)")
    print("\nEstimando ação ótima para cada estado usando Q-Learning.")
    print("Simulando três configurações de parâmetros:")
    print("a) α=0.2 (learning rate), β=0.2 (discount factor)")
    print("b) α=0.4 (learning rate), β=0.1 (discount factor)") 
    print("c) α=0.1 (learning rate), β=0.4 (discount factor)")

    results = run_all_qlearning()
    print_results(results)
    
    print("\n" + "="*60)
    print("TRABALHO CONCLUÍDO COM SUCESSO!")
    print("="*60)

if __name__ == "__main__":
    main()
