# Política 1: acelera para a esquerda quando o poste está inclinado para a esquerda 
# e acelera para a direita quando o poste está inclinado para a direita
# Ação 0 = esquerda, Ação 1 = direita

def policy_1(obs):
    """Política determinística baseada na inclinação do poste.
    
    Args:
        obs: observação do ambiente [posição_carrinho, velocidade_carrinho, ângulo_poste, velocidade_angular_poste]
    
    Returns:
        int: ação (0=esquerda, 1=direita)
    """
    _, _, theta, _ = obs
    # Se theta > 0 (inclinado para direita), acelera para direita (ação 1)
    # Se theta < 0 (inclinado para esquerda), acelera para esquerda (ação 0)
    return 1 if theta > 0 else 0
