# Previsão de Inadimplência Bancária via Regressão Logística

> Projeto da disciplina **Otimização Não-Linear** — UFPB
> Prof. Felipe A. G. Moreno

---

## Sobre o Projeto

Este trabalho formula o treinamento de um modelo de **regressão logística** como um
**problema de otimização não linear irrestrito**, aplicado à previsão de inadimplência
bancária. O objetivo é prever se um cliente irá ou não se tornar inadimplente com base
em características financeiras e comportamentais.

Todos os algoritmos foram implementados do zero, sem uso de bibliotecas de otimização
prontas, com foco nos métodos estudados na disciplina.

### Problema de Otimização

Dado um conjunto de treinamento com $n$ exemplos e $p$ features, busca-se o vetor de
parâmetros $\theta \in \mathbb{R}^{p+1}$ que maximize a função de log-verossimilhança:

$$\ell(\theta) = \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

onde $\hat{y}_i = \sigma(x_i^\top \theta) = \dfrac{1}{1 + e^{-x_i^\top \theta}}$.

---

## Estrutura do Repositório

```
ONL/
├── dados/
│   └── ...                        # Dataset de inadimplência
├── scripts/
│   └── regressao_logistica.py     # Implementação completa dos algoritmos
├── notebooks/
│   └── apresentacao.ipynb         # Notebook de apresentação
├── Docs/
│   ├── Projeto_ONL_2026.pdf       # Enunciado do projeto
│   ├── Avanço do Projeto - ONL.pdf
│   ├── Metodos_usando_intervalos.ipynb
│   └── Resultados_Preliminares.ipynb
├── PLANO.md                       # Plano de desenvolvimento detalhado
├── CLAUDE.md                      # Regras de implementação e contribuição
└── README.md
```

---

## Algoritmos Implementados

### Pré-processamento
- Tratamento de valores ausentes
- Normalização por z-score
- Divisão treino/teste
- Adição de bias

### Modelo de Regressão Logística
- Função sigmoide
- Log-verossimilhança e função de perda
- Gradiente: $\nabla L(\theta) = X^\top(\hat{y} - y)$
- Hessiana: $H(\theta) = X^\top W X$, com $W = \text{diag}(\hat{y}_i(1-\hat{y}_i))$

### Métodos de Busca em Linha
| Método | Descrição |
|--------|-----------|
| Intervalo Inicial | Busca automática de um intervalo contendo o mínimo |
| Partição Igual | Trisseção iterativa do intervalo |
| Seção Áurea | Redução pelo fator $\varphi = (\sqrt{5}-1)/2$ por iteração |
| Ajuste Quadrático | Interpolação de parábola pelos três pontos |
| Biseção na Derivada | Biseção aplicada sobre $g'(\alpha) = 0$ |

### Algoritmos de Otimização
| Método | Convergência | Custo por Iteração |
|--------|-------------|-------------------|
| Gradiente Descendente | Linear | $O(np)$ |
| Método de Newton | Quadrática | $O(np^2 + p^3)$ |

### Métricas de Avaliação
- Acurácia, Precisão, Recall, F1-Score
- Curva ROC e AUC
- Matriz de Confusão

---

## Como Executar

### Requisitos

```bash
pip install -r requirements.txt
```

Dependências: `numpy`, `pandas`, `matplotlib`

### Executando o Script

```python
from scripts.regressao_logistica import *

# Carregar e pré-processar dados
X, y = carregar_dados("dados/credit_default.csv")
X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)

# Treinar com gradiente descendente
theta, historico_perda, _, _ = gradiente_descendente(
    funcao_perda, gradiente_perda,
    theta_inicial=np.zeros(X_treino.shape[1]),
    X=X_treino, y=y_treino,
    metodo_busca='secao_aurea'
)

# Avaliar
y_pred = prever_classe(theta, X_teste)
print(f"Acurácia: {calcular_acuracia(y_teste, y_pred):.4f}")
```

### Notebook de Apresentação

```bash
jupyter notebook notebooks/apresentacao.ipynb
```

---

## Integrantes

- Gabriel Negreiros Saraiva
- Júlia Moraes da Silva
- Luiz Eduardo de Almeida Siqueira Silva
- Paulo Victor Cordeiro Rufino de Araújo
- Pedro Lucas Simões Cabral

---

## Referências

- SUN, S. et al. *A Survey of Optimization Methods From a Machine Learning Perspective*. IEEE Transactions on Cybernetics, 2019.
- GAMBELLA, C. et al. *Optimization Problems for Machine Learning: A Survey*. European Journal of Operational Research, 2021.
- CAETANO, T. M. *Algoritmos de Aprendizado de Máquina no Estudo da Inadimplência em uma Instituição Financeira*. TCC — UFU, 2024.
- SOUZA, F. B. *Gestão do Risco de Crédito com Score Dinâmico para previsão de inadimplência de PME*. Dissertação — PUC-SP, 2025.

---

*Universidade Federal da Paraíba — Centro de Informática — 2026*
