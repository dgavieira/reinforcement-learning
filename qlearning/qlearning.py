import numpy as np
from qlearning.mdp import num_states, num_actions, step_robot_deterministic

def q_learning(learning_rate, discount_factor, episodes=5000, steps_per_episode=50):
    """Implementa o algoritmo Q-Learning para o problema do robô de reciclagem.
    
    NOTA: Os parâmetros α e β do trabalho correspondem a:
    - learning_rate (α): Taxa de aprendizado do algoritmo Q-Learning
    - discount_factor (β): Fator de desconto para recompensas futuras
    
    Args:
        learning_rate (float): Taxa de aprendizado (α no trabalho)
        discount_factor (float): Fator de desconto (β no trabalho) 
        episodes (int): Número de episódios de treinamento
        steps_per_episode (int): Passos por episódio
    
    Returns:
        np.ndarray: Matriz Q(s,a) treinada
    """
    # Inicializa a tabela Q com zeros
    Q = np.zeros((num_states, num_actions))

    for episode in range(episodes):
        # Começa em um estado aleatório (high=0 ou low=1)
        s = np.random.randint(num_states)

        for step in range(steps_per_episode):
            # Escolhe uma ação válida aleatoriamente (exploração)
            if s == 0:  # Estado high
                a = np.random.choice([0, 1])  # search ou wait (recharge não disponível)
            else:  # Estado low
                a = np.random.choice([0, 1, 2])  # search, wait ou recharge
            
            # Executa a ação e observa o resultado
            s2, r = step_robot_deterministic(s, a)

            # Atualização Q-Learning: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            Q[s, a] += learning_rate * (r + discount_factor * np.max(Q[s2]) - Q[s, a])
            
            # Transição para o próximo estado
            s = s2

    return Q


def extract_policy(Q):
    """Extrai a política ótima da matriz Q.
    
    Args:
        Q (np.ndarray): Matriz Q treinada
    
    Returns:
        np.ndarray: Política ótima (ação para cada estado)
    """
    policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        if s == 0:  # Estado high
            # Considera apenas ações válidas: search=0, wait=1
            valid_actions = [0, 1]
            valid_q_values = [Q[s, a] for a in valid_actions]
            best_action_idx = np.argmax(valid_q_values)
            policy[s] = valid_actions[best_action_idx]
        else:  # Estado low
            # Todas as ações são válidas: search=0, wait=1, recharge=2
            policy[s] = np.argmax(Q[s])
    
    return policy
