# Trabalho 5: Aprendizado por Reforço

Este projeto implementa os dois problemas solicitados no Trabalho 5 da disciplina de Reconhecimento de Padrões.

## Estrutura do Projeto

```
reinforcement-learning/
├── main.py                    # Arquivo principal - executa ambas as partes
├── cartpole/                  # Parte 1: Problema do CartPole
│   ├── policy1.py            # Política 1 (determinística)
│   ├── policy2_reinforce.py  # Política 2 (rede neural + REINFORCE)
│   ├── evaluate.py           # Avaliação das políticas
│   └── plot.py               # Geração de gráficos
└── qlearning/                # Parte 2: Problema do robô
    ├── mdp.py                # Definição do MDP do robô de reciclagem
    ├── qlearning.py          # Algoritmo Q-Learning
    └── run_qlearning.py      # Execução dos três cenários
```

## Parte 1: Problema do CartPole

### Objetivo
Comparar duas políticas diferentes para manter uma haste vertical em um carrinho:

### Política 1 (Determinística)
- **Estratégia**: Acelera para a esquerda quando o poste está inclinado para a esquerda e acelera para a direita quando o poste está inclinado para a direita
- **Implementação**: Decisão baseada no sinal do ângulo θ da haste
- **Código**: `cartpole/policy1.py`

### Política 2 (Rede Neural + REINFORCE)
- **Estratégia**: Rede neural treinada com o algoritmo REINFORCE de Ronald Williams
- **Arquitetura**: 
  - Entrada: 4 observações do ambiente (posição carrinho, velocidade carrinho, ângulo poste, velocidade angular poste)
  - Camadas ocultas: 24 neurônios com ReLU
  - Saída: 2 ações (esquerda/direita) com Softmax
- **Algoritmo**: REINFORCE (Policy Gradient)
- **Suporte GPU**: ✅ Detecção automática e execução otimizada em CUDA
- **Código**: `cartpole/policy2_reinforce.py`

### Configuração
- **Episódios**: 10 episódios para cada política
- **Máximo de tentativas**: 500 por episódio
- **Métricas**: Média e desvio padrão do número de tentativas bem-sucedidas

## Parte 2: Problema do Robô de Reciclagem

### Descrição do Problema
Robô móvel que coleta latas vazias em ambiente de escritório:
- **Bateria**: Dois níveis (alto/baixo) 
- **Objetivo**: Maximizar coleta de latas evitando esgotamento da bateria
- **Penalidade**: -3 pontos quando a bateria esgota e precisa ser resgatado

### Estados do MDP
- **high**: Nível alto de bateria
- **low**: Nível baixo de bateria

### Ações Disponíveis
- **search**: Procurar ativamente por latas (maior recompensa, gasta bateria)
- **wait**: Esperar que alguém traga latas (menor recompensa, economiza energia)  
- **recharge**: Ir para base recarregar (só disponível no estado low)

### Dinâmica de Transições
- **high + search**: permanece high com prob α, vai para low com prob (1-α)
- **high + wait**: sempre permanece high
- **low + search**: permanece low com prob β, esgota bateria com prob (1-β)
- **low + wait**: sempre vai para high
- **low + recharge**: sempre vai para high

### Cenários Simulados
O algoritmo Q-Learning é executado com três configurações:

1. **Cenário a**: α=0.2 (learning rate), β=0.2 (discount factor)
2. **Cenário b**: α=0.4 (learning rate), β=0.1 (discount factor)  
3. **Cenário c**: α=0.1 (learning rate), β=0.4 (discount factor)

### Saída Esperada
Para cada cenário:
- **Matriz Q(s,a)**: Valores Q para cada par estado-ação
- **Política Ótima**: Ação recomendada para cada estado

## Como Executar

### Pré-requisitos
```bash
pip install gymnasium torch numpy matplotlib
```

**Para suporte a GPU (opcional):**
- NVIDIA GPU com CUDA compatível
- Drivers NVIDIA atualizados
- PyTorch com suporte CUDA instalado

**Verificar GPU:**
```bash
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

### Execução
```bash
python main.py              # Execução padrão (GPU automática)
python test_gpu.py          # Teste de performance GPU vs CPU
```

**Controle de GPU:**
```python
# No código, para forçar CPU:
train_policy_network(use_gpu=False)

# Para forçar GPU (se disponível):
train_policy_network(use_gpu=True)
```

### Saída
O programa irá:
1. Avaliar a Política 1 (determinística) no CartPole
2. Treinar e avaliar a Política 2 (REINFORCE) no CartPole  
3. Gerar gráfico comparativo das políticas
4. Executar Q-Learning para os três cenários do robô
5. Exibir matrizes Q e políticas ótimas para cada cenário

## Resultados Esperados

### Parte 1 (CartPole)
- Comparação do desempenho entre política determinística e aprendida
- Gráfico mostrando tentativas bem-sucedidas por episódio
- Estatísticas: média e desvio padrão para cada política

### Parte 2 (Robô)
- Três matrizes Q diferentes baseadas nos parâmetros α e β
- Políticas ótimas que maximizam a coleta de latas
- Análise de como os parâmetros afetam a estratégia do robô

## Implementação Técnica

### Algoritmo REINFORCE
- Implementação fiel ao algoritmo de Ronald Williams
- Cálculo de retornos descontados (Monte Carlo)
- Atualização de gradiente de política

### Algoritmo Q-Learning  
- Atualização temporal: Q(s,a) ← Q(s,a) + α[r + γmax(Q(s',a')) - Q(s,a)]
- Exploração aleatória durante treinamento
- Extração de política ε-greedy (ação com maior valor Q)

### Validação
O código implementa fielmente:
- ✅ Política 1 baseada na inclinação do poste
- ✅ Política 2 com rede neural e REINFORCE
- ✅ Episódios limitados a 500 tentativas
- ✅ Avaliação em 10 episódios com média e desvio padrão
- ✅ MDP do robô com estados e ações corretos
- ✅ Três cenários de Q-Learning conforme especificado
- ✅ Matriz Q(s,a) e política ótima para cada cenário
- ✅ Suporte a GPU com detecção automática e otimizações
- ✅ Monitoramento de performance e uso de memória GPU