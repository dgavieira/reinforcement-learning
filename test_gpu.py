#!/usr/bin/env python3
"""
Teste da implementação com GPU para a Política 2 (REINFORCE)
"""

from cartpole.evaluate import evaluate_policy_1, evaluate_trained_policy
from cartpole.policy2_reinforce import train_policy_network
from cartpole.plot import plot_returns
import numpy as np
import torch

def test_gpu_performance():
    print("\n" + "="*60)
    print("TESTE DE PERFORMANCE GPU vs CPU - POLÍTICA REINFORCE")
    print("="*60)
    
    # Teste com CPU
    print("\n🔥 TESTE 1: Treinamento em CPU")
    print("-" * 40)
    policy_cpu, returns_cpu = train_policy_network(
        episodes=20, 
        max_steps=500, 
        lr=0.01,  # Learning rate maior para convergência mais rápida
        use_gpu=False
    )
    cpu_results = evaluate_trained_policy(policy_cpu, episodes=10, max_steps=500)
    
    print(f"\n📊 Resultados CPU:")
    print(f"   Média: {np.mean(cpu_results):.2f} tentativas")
    print(f"   Desvio padrão: {np.std(cpu_results):.2f}")
    
    # Teste com GPU (se disponível)
    if torch.cuda.is_available():
        print("\n🚀 TESTE 2: Treinamento em GPU")
        print("-" * 40)
        policy_gpu, returns_gpu = train_policy_network(
            episodes=20, 
            max_steps=500, 
            lr=0.01,  # Learning rate maior para convergência mais rápida
            use_gpu=True
        )
        gpu_results = evaluate_trained_policy(policy_gpu, episodes=10, max_steps=500)
        
        print(f"\n📊 Resultados GPU:")
        print(f"   Média: {np.mean(gpu_results):.2f} tentativas")
        print(f"   Desvio padrão: {np.std(gpu_results):.2f}")
        
        # Comparação
        print(f"\n🏆 COMPARAÇÃO:")
        print(f"   CPU: {np.mean(cpu_results):.1f} ± {np.std(cpu_results):.1f}")
        print(f"   GPU: {np.mean(gpu_results):.1f} ± {np.std(gpu_results):.1f}")
        
        # Gera gráfico comparativo
        try:
            plot_returns(cpu_results, gpu_results, out_file="gpu_vs_cpu_comparison.png")
        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")
    else:
        print("\n⚠️  GPU não disponível para teste comparativo")
    
    print("\n" + "="*60)
    print("TESTE CONCLUÍDO!")
    print("="*60)

if __name__ == "__main__":
    test_gpu_performance()