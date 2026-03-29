# Regressão Logística via Otimização Não Linear

**Disciplina:** Otimização Não Linear
**Instituição:** UFPB
**Professor:** Felipe A. G. Moreno

**Integrantes:**
- Gabriel Negreiros Saraiva
- Júlia Moraes da Silva
- Luiz Eduardo de Almeida Siqueira Silva
- Paulo Victor Cordeiro Rufino de Araújo
- Pedro Lucas Simões Cabral

---

## Visão Geral

Este projeto formula o treinamento de uma **Regressão Logística** como um problema de otimização não linear e compara empiricamente dois métodos de otimização:

- **Gradiente Descendente** com busca em linha exata
- **Método de Newton** com busca em linha exata

Ambos os métodos são avaliados no problema de **previsão de inadimplência de crédito bancário**.

---

## Estrutura do Repositório

```
ONL/
├── regressao_log.py                        # Implementação principal da classe RegressaoLogistica
├── main.ipynb                              # Notebook principal com análise completa
├── Relatorio_Final_ONL_v2.docx             # Relatório final (versão atualizada)
├── dados/
│   ├── credito.csv                         # Dataset UFPB (10.128 amostras, 15 features)
│   └── credit_default.csv                  # Dataset UCI Credit Card Default (30.000 amostras)
└── Docs/
    ├── Projeto_ONL_2026.pdf                # Especificação oficial do projeto
    ├── Avanço do Projeto - ONL.pdf         # Relatório de progresso
    ├── Metodos_usando_intervalos.ipynb     # Notebook educacional em Julia sobre buscas em linha
    └── Resultados_Preliminares.ipynb       # Resultados e cálculos preliminares
```

---

## Descrição dos Arquivos

### `regressao_log.py` — Implementação Principal

Contém a classe `RegressaoLogistica` com todos os algoritmos implementados do zero.

**Parâmetros do construtor:**

| Parâmetro | Opções | Padrão | Descrição |
|-----------|--------|--------|-----------|
| `metodo_otimizacao` | `'gradiente_descendente'`, `'newton'` | — | Algoritmo de otimização |
| `metodo_busca` | `'secao_aurea'`, `'particao_igual'`, `'ajuste_quadratico'` | — | Método de busca em linha |
| `tmax` | int | `1000` | Número máximo de iterações |
| `tolerancia` | float | `1e-6` | Critério de parada: `‖∇L(θ)‖ < tolerancia` |
| `lambda_` | float | `0.01` | Coeficiente de regularização L2 |

**Métodos públicos:**

- `fit(X, y)` — treina o modelo; retorna `self` (method chaining)
- `predict(X)` — prediz classes binárias (0 ou 1)
- `predict_prob(X)` — retorna P(y=1|x) = σ(xᵀθ) para cada amostra
- `getW()` — retorna o vetor de parâmetros θ treinado
- `getRegressionY(x1, shift=0)` — retorna x₂ da fronteira de decisão para visualização 2D

**Atributos após `.fit()`:**

- `w` — parâmetros ótimos θ*
- `historico_perda_` — valor da perda por iteração
- `historico_norma_grad_` — norma do gradiente por iteração
- `historico_alpha_` — passo α ótimo por iteração
- `historico_tempo_` — tempo acumulado (segundos) por iteração
- `n_iteracoes_` — total de iterações até convergência

---

### `main.ipynb` — Análise Completa

Notebook principal estruturado nas seguintes seções:

| Seção | Conteúdo |
|-------|----------|
| 2 | Carregamento e pré-processamento dos dados |
| 3 | Formulação matemática do problema |
| 3.1 | Gradiente, Hessiana e **prova de convexidade** (H ⪰ 0) |
| 4 | Demonstração da função sigmoide |
| 5 | Demonstração dos métodos de busca em linha 1D |
| 6 | Treinamento com Gradiente Descendente |
| 7 | Treinamento com Método de Newton |
| 8 | **Comparação dos métodos** (iterações, perda, tempo) |
| 8.1 | Comparação dos métodos de busca em linha |
| 8.2 | Análise do número de condição κ(H) |
| 9 | Avaliação no conjunto de teste (acurácia, F1, AUC-ROC) |
| 10 | Conclusões |

---

---

### `Docs/Metodos_usando_intervalos.ipynb` — Notebook Educacional (Julia)

Notebook escrito em **Julia** com implementações didáticas dos métodos de busca em intervalo:
- Busca de intervalo inicial (`intervalo_inicial`)
- Partição igual (trissecção)
- Seção áurea
- Ajuste quadrático
- Bissecção

Serve como base matemática antes da implementação em Python.

---

## Formulação Matemática

### Modelo

$$\hat{y} = \sigma(X\theta), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Função de Perda (Cross-Entropy + Regularização L2)

$$L(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \sigma(x_i^\top\theta) + (1-y_i)\log(1-\sigma(x_i^\top\theta))\right] + \frac{\lambda}{2}\|\theta_{1:}\|^2$$

> O viés `θ₀` não é regularizado (convenção padrão).

### Gradiente

$$\nabla L(\theta) = \frac{1}{n}X^\top(\sigma(X\theta) - y) + \lambda[0,\, \theta_1, \ldots, \theta_p]^\top$$

### Hessiana

$$H(\theta) = \frac{1}{n}X^\top W X + \lambda \operatorname{diag}(0, 1, \ldots, 1)$$

onde $W = \operatorname{diag}(\sigma_i(1 - \sigma_i))$.

**Prova de convexidade:** Para qualquer $v \in \mathbb{R}^{p+1}$:

$$v^\top H v = \frac{1}{n}\sum_i \underbrace{\sigma_i(1-\sigma_i)}_{> 0}\, (x_i^\top v)^2 \geq 0 \implies H \succeq 0$$

Logo $L(\theta)$ é globalmente convexa — qualquer mínimo local é global.

---

## Algoritmos Implementados

### Gradiente Descendente com Busca em Linha

```
para k = 1, 2, ...:
    se ‖∇L(θ_k)‖ < tolerancia: parar
    d ← −∇L(θ_k)
    α* ← argmin_α L(θ_k + α·d)   ← busca em linha exata
    θ_{k+1} ← θ_k + α*·d
```

- **Convergência:** Linear — O(κ(H)) iterações
- **Custo por iteração:** O(np)

### Método de Newton com Busca em Linha

```
para k = 1, 2, ...:
    se ‖∇L(θ_k)‖ < tolerancia: parar
    H ← Hessiana em θ_k
    Resolver (H + εI)·d = −∇L(θ_k)   ← regularização de Tikhonov
    α* ← argmin_α L(θ_k + α·d)
    θ_{k+1} ← θ_k + α*·d
```

- **Convergência:** Quadrática — O(log(1/ε)) iterações, independente de κ(H)
- **Custo por iteração:** O(np² + p³)

### Métodos de Busca em Linha

| Método | Redutor por iteração | Avaliações/iter | Característica |
|--------|---------------------|-----------------|----------------|
| **Seção Áurea** | φ ≈ 0.618 | 1 | Mais eficiente (ótimo para unimodais) |
| **Partição Igual** | 2/3 | 2 | Mais simples, menos eficiente |
| **Ajuste Quadrático** | variável | 1 | Rápido para funções suaves |

Todos os métodos recebem um intervalo inicial via busca por duplicação a partir de α = 0.

---

## Resultados Principais

### Gradiente Descendente vs Método de Newton

| Métrica | Gradiente Descendente | Método de Newton |
|---------|----------------------|-----------------|
| Iterações até convergência | **35** | **3** |
| Perda final L(θ*) | 0,3082 | 0,3082 |
| ‖∇L(θ*)‖ final | 1,00 × 10⁻⁴ | 6,54 × 10⁻⁶ |
| Tempo de execução | ~0,54 s | ~0,07 s |
| Convergência teórica | Linear | Quadrática |

> Newton convergiu em **11,7× menos iterações** e **7× menos tempo** que o GD, usando a curvatura da Hessiana para escalar a direção de descida.

### Comparação dos Métodos de Busca em Linha (GD)

| Método | Iterações | Perda Final | Tempo |
|--------|-----------|-------------|-------|
| Seção Áurea | 35 | 0,3082 | 0,54 s |
| Partição Igual | 35 | 0,3082 | 0,74 s |
| Ajuste Quadrático | 35 | 0,3082 | 0,19 s |

### Desempenho no Conjunto de Teste (Newton, dados originais)

| Métrica | Valor |
|---------|-------|
| Acurácia | **86,53%** |
| Precisão | 68,84% |
| Recall | 29,23% |
| F1-Score | 41,04% |
| AUC-ROC | **87,34%** |

---

## Datasets

### `credito.csv` (UFPB)
- 10.128 amostras × 15 features
- Features: idade, sexo, escolaridade, salário anual, limite de crédito, transações, etc.
- Target: `default` (0 = adimplente, 1 = inadimplente)

### `credit_default.csv` (UCI)
- 30.000 amostras × 23 features
- Features: limite de crédito, histórico de pagamentos, valores de faturas, etc.
- Target: `inadimplente` (0/1)
- Usado no notebook de apresentação

**Pré-processamento:** one-hot encoding das variáveis categóricas, normalização z-score (`StandardScaler`) e divisão treino/teste 80/20 (seed=42, stratify=y).

---

## Como Usar

```python
from regressao_log import RegressaoLogistica
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dividir e normalizar dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Treinar com Gradiente Descendente + Seção Áurea
modelo = RegressaoLogistica(
    metodo_otimizacao='gradiente_descendente',
    metodo_busca='secao_aurea',
    tmax=500,
    tolerancia=1e-6,
    lambda_=0.01
)
modelo.fit(X_train_scaled, y_train)

# Predição e avaliação
y_pred = modelo.predict(X_test_scaled)
print(f"Acurácia: {(y_pred == y_test).mean():.4f}")
print(f"Iterações: {modelo.n_iteracoes_}")
print(f"Perda final: {modelo.historico_perda_[-1]:.6f}")

# Treinar com Newton (muito mais rápido)
modelo_newton = RegressaoLogistica(
    metodo_otimizacao='newton',
    metodo_busca='secao_aurea',
    tmax=100,
    tolerancia=1e-6
)
modelo_newton.fit(X_train_scaled, y_train)
```

---

## Dependências

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
```

Para o notebook Julia (`Docs/Metodos_usando_intervalos.ipynb`), é necessário ter Julia instalado.
