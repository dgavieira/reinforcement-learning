import numpy as np

"""
MDP do Robô de Reciclagem

Descrição do problema:
- Robô móvel coleta latas vazias em ambiente de escritório
- Funciona com bateria recarregável com dois níveis: alto e baixo
- Ações disponíveis: search (procurar latas), wait (esperar), recharge (recarregar)
- Recompensas: +Rsearch/+Rwait por latas coletadas, -3 se bateria esgotar
- α, β são parâmetros de probabilidade de transição (α > β)

Estados:
- 0: high (nível alto de bateria)
- 1: low (nível baixo de bateria)

Ações:
- 0: search (procurar ativamente por latas)  
- 1: wait (esperar que alguém traga latas)
- 2: recharge (ir para base recarregar - só disponível em low)

Dinâmica de transições:
- high + search: permanece high com prob α, vai para low com prob (1-α)
- high + wait: sempre permanece high (sem risco de esgotar bateria)
- low + search: permanece low com prob β, esgota bateria com prob (1-β) -> precisa ser resgatado
- low + wait: sempre vai para high (economia de energia)
- low + recharge: sempre vai para high

Recompensas esperadas:
- search: Rsearch (maior que Rwait pois é mais eficiente)
- wait: Rwait (menor que Rsearch)
- esgotamento da bateria: -3 (penalidade por precisar ser resgatado)
"""

# Estados do MDP
num_states = 2  # high=0, low=1
# Ações: search=0, wait=1, recharge=2 (só disponível em low)
num_actions = 3

# Parâmetros do problema (serão definidos externamente via α e β)
# α: probabilidade de permanecer em high durante search (α > β)
# β: probabilidade de permanecer em low durante search (β < α)
# Rsearch: recompensa esperada ao procurar latas (maior valor)
# Rwait: recompensa esperada ao esperar por latas (menor valor)

# Valores padrão para simulação (podem ser alterados)
DEFAULT_ALPHA = 0.7
DEFAULT_BETA = 0.3  
DEFAULT_R_SEARCH = 2.0
DEFAULT_R_WAIT = 1.0
BATTERY_DEPLETED_REWARD = -3.0

def get_transition_probs(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """Retorna as probabilidades de transição para o MDP.
    
    Args:
        alpha (float): Probabilidade de permanecer em high durante search
        beta (float): Probabilidade de permanecer em low durante search
        
    Returns:
        dict: Probabilidades de transição P[s][a][s'] 
    """
    # P[estado_atual][ação][próximo_estado] = probabilidade
    P = {
        0: {  # Estado high
            0: {0: alpha, 1: 1-alpha},        # search: high com prob α, low com prob (1-α)
            1: {0: 1.0, 1: 0.0},              # wait: sempre permanece high
            2: {0: 1.0, 1: 0.0}               # recharge não disponível (ação inválida)
        },
        1: {  # Estado low  
            0: {0: 0.0, 1: beta},             # search: low com prob β, bateria esgotada com prob (1-β)
            1: {0: 1.0, 1: 0.0},              # wait: sempre vai para high
            2: {0: 1.0, 1: 0.0}               # recharge: sempre vai para high
        }
    }
    return P

def get_rewards(r_search=DEFAULT_R_SEARCH, r_wait=DEFAULT_R_WAIT):
    """Retorna as recompensas esperadas para o MDP.
    
    Args:
        r_search (float): Recompensa esperada ao procurar latas
        r_wait (float): Recompensa esperada ao esperar por latas
        
    Returns:
        dict: Recompensas esperadas R[s][a]
    """
    R = {
        0: {  # Estado high
            0: r_search,    # search: recompensa por latas encontradas
            1: r_wait,      # wait: recompensa menor por latas trazidas
            2: 0.0          # recharge não disponível
        },
        1: {  # Estado low
            0: r_search * 0.5 + BATTERY_DEPLETED_REWARD * 0.5,  # search: média entre sucesso e falha
            1: r_wait,      # wait: recompensa por latas trazidas  
            2: 0.0          # recharge: sem latas coletadas
        }
    }
    return R

def step_robot(s, a, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, 
               r_search=DEFAULT_R_SEARCH, r_wait=DEFAULT_R_WAIT):
    """Executa uma ação no ambiente do robô (simulação estocástica).
    
    Args:
        s (int): Estado atual (0=high, 1=low)
        a (int): Ação (0=search, 1=wait, 2=recharge)
        alpha (float): Probabilidade de permanecer em high durante search
        beta (float): Probabilidade de permanecer em low durante search
        r_search (float): Recompensa por procurar latas
        r_wait (float): Recompensa por esperar latas
    
    Returns:
        tuple: (próximo_estado, recompensa_obtida)
    """
    if s == 0:  # Estado high
        if a == 0:  # search
            if np.random.random() < alpha:
                return 0, r_search  # Permanece high, coleta latas
            else:
                return 1, r_search  # Vai para low, mas ainda coleta latas
        elif a == 1:  # wait
            return 0, r_wait  # Permanece high, coleta menos latas
        else:  # recharge (ação inválida)
            return 0, 0.0
            
    else:  # Estado low (s == 1)
        if a == 0:  # search
            if np.random.random() < beta:
                return 1, r_search  # Permanece low, coleta latas
            else:
                return 0, BATTERY_DEPLETED_REWARD  # Bateria esgotada, precisa ser resgatado -> volta para high após resgate
        elif a == 1:  # wait
            return 0, r_wait  # Vai para high, economiza energia
        else:  # recharge
            return 0, 0.0  # Vai para high, sem latas coletadas

# Função simplificada para Q-learning (determinística baseada nas probabilidades)
def step_robot_deterministic(s, a):
    """Versão determinística para Q-learning baseada nas transições mais prováveis.
    
    Args:
        s (int): Estado atual
        a (int): Ação
        
    Returns:
        tuple: (próximo_estado, recompensa_esperada)
    """
    # Usa valores padrão para as probabilidades e recompensas
    P = get_transition_probs()
    R = get_rewards()
    
    # Ação inválida
    if s == 0 and a == 2:  # recharge em high
        return s, 0.0
        
    # Escolhe próximo estado baseado na transição mais provável
    if s == 0:  # high
        if a == 0:  # search
            # Transição mais provável (depende de α)
            next_state = 0 if DEFAULT_ALPHA >= 0.5 else 1
            return next_state, R[s][a]
        else:  # wait
            return 0, R[s][a]
    else:  # low
        if a == 0:  # search  
            # Considerando o risco de esgotar bateria
            expected_reward = DEFAULT_BETA * DEFAULT_R_SEARCH + (1 - DEFAULT_BETA) * BATTERY_DEPLETED_REWARD
            next_state = 1 if DEFAULT_BETA >= 0.5 else 0  # Se β >= 0.5, mais provável ficar em low
            return next_state, expected_reward
        elif a == 1:  # wait
            return 0, R[s][a]
        else:  # recharge
            return 0, R[s][a]
