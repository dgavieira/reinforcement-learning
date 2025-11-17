import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    """Rede neural para Política 2 - Algoritmo REINFORCE de Ronald Williams.
    
    Recebe uma observação como entrada e produz a ação a ser executada como saída.
    Suporta execução em GPU quando disponível.
    """
    def __init__(self, input_size=4, hidden_size=24, output_size=2, device=None):
        super().__init__()
        
        # Detecta automaticamente o dispositivo se não especificado
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Move a rede para o dispositivo apropriado
        self.to(self.device)

    def forward(self, x):
        # Garante que a entrada esteja no mesmo dispositivo que a rede
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return self.net(x)
    
    def get_device_info(self):
        """Retorna informações sobre o dispositivo em uso."""
        if self.device.type == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(self.device.index)} (CUDA {torch.version.cuda})"
        else:
            return "CPU"


def compute_returns(rewards, gamma=0.99, device=None):
    """Calcula os retornos descontados (Monte Carlo) para o algoritmo REINFORCE.
    
    Args:
        rewards (list): Lista de recompensas do episódio
        gamma (float): Fator de desconto
        device (torch.device): Dispositivo para alocar o tensor
        
    Returns:
        torch.Tensor: Retornos descontados
    """
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Cria o tensor no dispositivo apropriado
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.tensor(returns, dtype=torch.float32, device=device)


def train_policy_network(episodes=10, max_steps=500, lr=0.001, gamma=0.99, use_gpu=True):
    """Treina a rede neural usando o algoritmo REINFORCE de Ronald Williams.
    
    Args:
        episodes (int): Número de episódios para treinamento (padrão: 10)
        max_steps (int): Máximo de tentativas por episódio (padrão: 500)
        lr (float): Taxa de aprendizado
        gamma (float): Fator de desconto
        use_gpu (bool): Se deve usar GPU quando disponível
    
    Returns:
        tuple: (rede_treinada, lista_de_retornos)
    """
    # Configura o dispositivo
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print("⚠️  GPU solicitada mas não disponível, usando CPU")
        else:
            print("💻 Usando CPU")
    
    env = gym.make("CartPole-v1")
    policy = PolicyNetwork(device=device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    
    print(f"Treinando Política 2 (REINFORCE) em {policy.get_device_info()}...")

    # Metrics para monitoramento de performance
    import time
    start_time = time.time()

    for episode in range(episodes):
        obs, _ = env.reset()
        rewards = []
        log_probs = []

        for step in range(max_steps):
            # Move observação para o dispositivo correto
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            # Calcula probabilidades (mantém gradientes para REINFORCE)
            probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Armazena log_prob (precisa manter gradiente para backprop)
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

            if terminated or truncated:
                break

        # Calcula os retornos (algoritmo REINFORCE)
        returns = compute_returns(rewards, gamma, device)

        # Atualiza a política
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        
        # Clip gradientes para estabilidade
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        
        optimizer.step()

        episode_return = sum(rewards)
        episode_returns.append(episode_return)
        
        if (episode + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episódio {episode + 1}/{episodes}: Retorno = {episode_return:.1f} | Tempo: {elapsed_time:.1f}s")
            
            # Limpeza de cache da GPU se disponível
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    env.close()
    
    # Estatísticas finais
    total_time = time.time() - start_time
    print(f"✅ Treinamento concluído em {total_time:.2f}s")
    if device.type == 'cuda':
        print(f"   Memória GPU utilizada: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)
    
    return policy, episode_returns
