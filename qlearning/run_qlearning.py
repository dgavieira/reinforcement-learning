from qlearning.qlearning import q_learning, extract_policy
import numpy as np

def run_all_qlearning():
    """Executa Q-Learning para os três cenários especificados no trabalho.
    
    Returns:
        dict: Resultados para cada configuração (α, β)
    """
    # Configurações conforme especificado no trabalho
    configs = [
        (0.2, 0.2),  # a) α=0.2; β=0.2
        (0.4, 0.1),  # b) α=0.4; β=0.1  
        (0.1, 0.4)   # c) α=0.1; β=0.4
    ]

    results = {}
    
    print("Executando Q-Learning para os três cenários...")

    for i, (alpha, beta) in enumerate(configs, 1):
        print(f"\nCenário {chr(96+i)}: α={alpha}, β={beta}")
        
        # Treina o Q-Learning
        Q = q_learning(alpha, beta)
        
        # Extrai a política ótima
        policy = extract_policy(Q)
        
        # Armazena os resultados
        results[(alpha, beta)] = (Q, policy)
        
        print(f"Treinamento concluído para α={alpha}, β={beta}")

    return results


def print_results(results):
    """Imprime os resultados de forma organizada.
    
    Args:
        results (dict): Resultados do Q-Learning
    """
    state_names = ['high', 'low']
    action_names = ['search', 'wait', 'recharge']
    
    for (alpha, beta), (Q, policy) in results.items():
        print(f"\n{'='*60}")
        print(f"Configuração: α={alpha} (learning rate), β={beta} (discount factor)")
        print(f"{'='*60}")
        
        print("\nMatriz Q(s,a):")
        print(f"{'Estado':<8} {'search':<10} {'wait':<10} {'recharge':<10}")
        print("-" * 45)
        
        for i in range(len(state_names)):
            search_val = Q[i,0] if i < Q.shape[0] else 0.0
            wait_val = Q[i,1] if i < Q.shape[0] else 0.0 
            recharge_val = Q[i,2] if i < Q.shape[0] and Q.shape[1] > 2 else 0.0
            
            # Indica ações inválidas
            if i == 0:  # Estado high
                recharge_str = "N/A" if recharge_val == 0.0 else f"{recharge_val:.3f}"
            else:
                recharge_str = f"{recharge_val:.3f}"
                
            print(f"{state_names[i]:<8} {search_val:<10.3f} {wait_val:<10.3f} {recharge_str:<10}")
        
        print("\nPolítica Ótima:")
        print("-" * 25)
        for i in range(len(state_names)):
            if i < len(policy):
                action = action_names[policy[i]]
                if i == 0 and policy[i] == 2:  # recharge em high (inválido)
                    action += " (INVÁLIDO)"
                print(f"{state_names[i]:<8} -> {action}")
        
        print("\nInterpretação:")
        print("- search: Procurar ativamente por latas (maior recompensa, mas gasta bateria)")  
        print("- wait: Esperar que tragam latas (menor recompensa, economiza energia)")
        print("- recharge: Ir para base recarregar (só disponível em low, recompensa zero)")
        print("- N/A: Ação não disponível neste estado")
