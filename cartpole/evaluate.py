import gymnasium as gym
import numpy as np
from cartpole.policy1 import policy_1

def evaluate_policy_1(episodes=10, max_steps=500):
    """Avalia a Política 1 (determinística) por um número de episódios.
    
    Args:
        episodes (int): Número de episódios para avaliação (padrão: 10)
        max_steps (int): Máximo de tentativas por episódio (padrão: 500)
    
    Returns:
        np.array: Array com o número de tentativas bem-sucedidas por episódio
    """
    env = gym.make("CartPole-v1")
    results = []
    
    print("Avaliando Política 1 (determinística)...")

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = policy_1(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        results.append(total_reward)
        
        if (episode + 1) % 5 == 0:
            print(f"Episódio {episode + 1}/{episodes}: {total_reward} tentativas bem-sucedidas")

    env.close()
    return np.array(results)


def evaluate_trained_policy(policy, episodes=10, max_steps=500):
    """Avalia uma política já treinada.
    
    Args:
        policy: Política neural treinada
        episodes (int): Número de episódios para avaliação
        max_steps (int): Máximo de tentativas por episódio
    
    Returns:
        np.array: Array com o número de tentativas bem-sucedidas por episódio
    """
    import torch
    
    env = gym.make("CartPole-v1")
    results = []
    
    # Detecta o dispositivo da política
    device = next(policy.parameters()).device
    
    print(f"Avaliando Política 2 (REINFORCE) treinada em {policy.get_device_info()}...")

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            with torch.no_grad():
                # Move observação para o mesmo dispositivo da política
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                probs = policy(obs_tensor)
                action = torch.argmax(probs).item()  # Ação determinística na avaliação
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        results.append(total_reward)
        
        if (episode + 1) % 5 == 0:
            print(f"Episódio {episode + 1}/{episodes}: {total_reward} tentativas bem-sucedidas")

    env.close()
    return np.array(results)
