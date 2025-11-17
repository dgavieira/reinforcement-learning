# Trabalho 5: Aprendizado por Reforço

**Autor:** Diego Giovanni de Alcântara Vieira  
**Programa:** Pós-Graduação em Engenharia Elétrica - UFAM  
**Email:** diego.vieira@ufam.edu.br

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Resumo Executivo

Este projeto implementa e compara duas abordagens fundamentais de **Aprendizado por Reforço** em problemas clássicos:

### 🎯 **Parte 1 - CartPole**
Comparação entre política determinística e rede neural treinada com **REINFORCE** (algoritmo de Ronald Williams) com suporte completo a **GPU (CUDA)**. 

**Resultados:** Política determinística superior (**40,4±7,9** vs **9,5±0,9** tentativas)

### 🤖 **Parte 2 - Robô de Reciclagem**  
Aplicação de **Q-Learning** em MDP discreto com três configurações de hiperparâmetros, revelando política ótima consistente: buscar latas em bateria alta, aguardar em bateria baixa.

---

## 🏗️ Estrutura do Projeto

```
reinforcement-learning/
├── main.py                    # 🚀 Execução principal
├── main.tex                   # 📄 Relatório científico LaTeX
├── cartpole/                  # 🎮 Parte 1: Problema CartPole
│   ├── __init__.py
│   ├── policy1.py            # Política determinística
│   ├── policy2_reinforce.py  # Política REINFORCE + GPU
│   ├── evaluate.py           # Avaliação de políticas
│   └── plot.py               # Visualização de resultados
├── qlearning/                # 🧠 Parte 2: Q-Learning
│   ├── __init__.py
│   ├── mdp.py               # Definição do MDP do robô
│   ├── qlearning.py         # Algoritmo Q-Learning
│   └── run_qlearning.py     # Execução dos cenários
├── test_gpu.py              # ⚡ Teste de performance GPU vs CPU
└── returns_plot.png         # 📊 Gráfico comparativo gerado
```

---

## 🚀 Início Rápido

### Pré-requisitos

```bash
# Dependências básicas
pip install gymnasium torch numpy matplotlib

# Para GPU (opcional)
# NVIDIA GPU + CUDA drivers + PyTorch CUDA
```

**Verificar GPU:**
```bash
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

### Execução

```bash
# Execução completa (GPU automático)
python main.py

# Teste de performance GPU vs CPU
python test_gpu.py
```

### Controle de GPU

```python
# Forçar CPU
train_policy_network(use_gpu=False)

# Forçar GPU (se disponível)  
train_policy_network(use_gpu=True)
```

---

## 📊 Resultados Experimentais

### Parte 1: CartPole

| **Política** | **Média** | **Desvio** | **Dispositivo** |
|--------------|-----------|------------|-----------------|
| Determinística | 40,4 | 7,9 | CPU |
| REINFORCE | 9,5 | 0,9 | GPU (RTX 4050) |

**Performance GPU:**
- **Dispositivo:** NVIDIA GeForce RTX 4050 Laptop GPU
- **Tempo:** 0,34s de treinamento  
- **Memória:** 0,02 GB VRAM utilizada
- **CUDA:** 13.0

### Parte 2: Q-Learning

#### Cenário A: α=0.2, β=0.2

| **Estado** | **search** | **wait** | **recharge** |
|------------|------------|----------|--------------|
| high | 2,500 | 1,500 | N/A |
| low | -1,000 | 1,500 | 0,500 |

**Política Ótima:** high→search, low→wait

#### Cenário B: α=0.4, β=0.1

| **Estado** | **search** | **wait** | **recharge** |
|------------|------------|----------|--------------|
| high | 2,222 | 1,222 | N/A |
| low | -1,278 | 1,222 | 0,222 |

**Política Ótima:** high→search, low→wait

#### Cenário C: α=0.1, β=0.4

| **Estado** | **search** | **wait** | **recharge** |
|------------|------------|----------|--------------|
| high | 3,333 | 2,333 | N/A |
| low | -0,167 | 2,333 | 1,333 |

**Política Ótima:** high→search, low→wait

---

## 🔬 Metodologia Científica

### Parte 1: CartPole

#### Política 1 (Determinística)
Estratégia baseada na inclinação da haste:
```python
ação = 1 se θ > 0 (direita)
ação = 0 se θ ≤ 0 (esquerda)
```

#### Política 2 (REINFORCE)
**Arquitetura da Rede Neural:**
- **Entrada:** 4 observações (posição, velocidade, ângulo, velocidade angular)
- **Camadas Ocultas:** 2 × 24 neurônios + ReLU
- **Saída:** 2 ações + Softmax (distribuição de probabilidades)

**Algoritmo REINFORCE:**
```
∇θ J(θ) = E[∑(t=0 to T) ∇θ log πθ(at|st) · Gt]
Gt = ∑(k=t to T) γ^(k-t) · rk  (γ = 0.99)
```

**Implementação GPU:**
- ✅ Detecção automática CUDA
- ✅ Transferência eficiente de tensores  
- ✅ Monitoramento de memória
- ✅ Gradient clipping para estabilidade

### Parte 2: MDP do Robô

**Estados:** `{high, low}` (níveis de bateria)  
**Ações:** `{search, wait, recharge}` (recharge só em low)

**Dinâmica de Recompensas:**
- `search`: R=+2.0 (coleta ativa, risco bateria)
- `wait`: R=+1.0 (coleta passiva, economia energia)
- Esgotamento: R=-3.0 (penalidade resgate)

**Q-Learning:**
```
Q(s,a) ← Q(s,a) + α[r + β·max Q(s',a') - Q(s,a)]
```

**Cenários:** 
- a) α=0.2, β=0.2 (balanceado)
- b) α=0.4, β=0.1 (rápido, presente) 
- c) α=0.1, β=0.4 (lento, futuro)

---

## 🎯 Análise e Discussão

### Por que Política Determinística Venceu?

1. **Conhecimento do Domínio:** Aproveita física do sistema diretamente
2. **Insuficiência de Dados:** 10 episódios inadequados para rede neural
3. **Complexidade vs Necessidade:** Problema simples não justifica deep learning

### Quando REINFORCE Seria Superior?

- Problemas com dinâmicas complexas/não-lineares
- Estados parcialmente observáveis  
- Ambientes com ruído significativo
- Treinamento extenso (>100 episódios)

### Robustez do Q-Learning

- **Política Ótima Consistente:** Independente dos hiperparâmetros testados
- **α (learning rate):** Afeta velocidade, não altera solução final
- **β (discount factor):** Influencia magnitude Q, preserva ordenação
- **MDP Bem-Estruturado:** Solução ótima clara e estável

---

## 🚀 Implementação GPU

### Funcionalidades

✅ **Detecção Automática:** GPU/CPU transparente  
✅ **Otimizações:** Transferências eficientes, cache management  
✅ **Monitoramento:** Uso de memória e performance  
✅ **Compatibilidade:** Funciona com/sem GPU  

### Sistema Testado

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6.1 GB)
- **CUDA:** 13.0  
- **Framework:** PyTorch 2.9.1+cu130

### Casos Ideais para GPU

- Redes neurais grandes (>1M parâmetros)
- Treinamento longo (>100 episódios)  
- Múltiplos ambientes paralelos
- Experimentos de hiperparâmetros

---

## 📈 Resultados e Conclusões

### Principais Descobertas

#### CartPole
- **Política determinística superou rede neural** (diferença: 30,9 tentativas)
- **Conhecimento de domínio** > aprendizado end-to-end em problemas simples
- **REINFORCE precisa mais dados** para convergir adequadamente
- **GPU implementada com sucesso** para escalabilidade futura

#### Q-Learning  
- **Convergência robusta** independente de hiperparâmetros
- **Política ótima consistente** em todos os cenários
- **MDP bem-estruturado** com solução clara
- **Estratégia emergente intuitiva:** explorar em high, conservar em low

### Contribuições Técnicas

1. **Implementação Completa:** Código modular e documentado
2. **Suporte GPU:** Sistema automático de detecção/otimização  
3. **Análise Comparativa:** Avaliação rigorosa de diferentes abordagens
4. **Reprodutibilidade:** Configurações completamente especificadas

### Limitações Identificadas

- **REINFORCE:** Poucos episódios, falta de baseline para variância
- **Comparação:** Ausência de outros algoritmos (A2C, PPO)
- **Escala:** Problema pequeno não evidencia vantagens GPU

---

## 🔮 Trabalhos Futuros

### Extensões Promissoras

1. **Algoritmos Avançados:** Actor-Critic, PPO, SAC
2. **Otimização Automática:** Grid search, Bayesian optimization
3. **MDPs Complexos:** Estados contínuos, múltiplos agentes
4. **Escalabilidade GPU:** Redes maiores, ambientes paralelos
5. **Aplicações Reais:** Robótica, jogos, controle industrial

### Melhorias Técnicas

- Implementar baseline para redução de variância
- Adicionar técnicas de regularização avançadas
- Explorar arquiteturas de rede mais sofisticadas
- Desenvolver benchmarks mais desafiadores

---

## 📚 Referências Científicas

1. **Williams, R. J.** (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.

2. **Sutton, R. S. & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.

3. **Farama Foundation** (2022). "Gymnasium: A standard API for reinforcement learning." https://gymnasium.farama.org

4. **Paszke, A.** et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." *Advances in Neural Information Processing Systems*.

---

## 🛠️ Requisitos Técnicos

### Sistema Mínimo
- Python 3.12+
- 4GB RAM  
- Processador multi-core

### Sistema Recomendado  
- Python 3.12+
- 8GB+ RAM
- GPU NVIDIA com CUDA 11.0+
- 4GB+ VRAM

### Dependências Python
```txt
gymnasium>=1.2.2
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. **Fork** o projeto
2. Crie uma **branch** para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)  
5. Abra um **Pull Request**

---

## 📞 Contato

**Diego Giovanni de Alcântara Vieira**
- 📧 Email: diego.vieira@ufam.edu.br
- 🎓 Programa de Pós-Graduação em Engenharia Elétrica - UFAM
- 📍 Manaus, Amazonas, Brasil

---

## ⭐ Agradecimentos

- **Universidade Federal do Amazonas (UFAM)**
- **Programa de Pós-Graduação em Engenharia Elétrica**
- **Comunidade PyTorch e Gymnasium**
- **Desenvolvedores de Aprendizado por Reforço**

---

**🎯 Este projeto demonstra implementações rigorosas de algoritmos fundamentais de Aprendizado por Reforço com infraestrutura computacional moderna, estabelecendo base sólida para pesquisas futuras em domínios de maior complexidade.**