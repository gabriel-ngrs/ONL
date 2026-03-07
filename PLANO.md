# Plano de Desenvolvimento — Projeto ONL 2026

**Tema:** Previsão de Inadimplência Bancária via Regressão Logística
**Disciplina:** Otimização Não-Linear — UFPB
**Prazo do Relatório:** 26 de março de 2026
**Prazo da Apresentação:** 30 de março ou 1º de abril de 2026

---

## Visão Geral

O projeto consiste em implementar do zero um classificador de regressão logística para
prever inadimplência bancária, formulando o treinamento como um problema de otimização
não linear irrestrito. Serão implementados os algoritmos de otimização estudados na
disciplina (busca em linha, gradiente descendente, método de Newton) e o desempenho do
modelo será avaliado em um dataset real.

**Entregáveis:**
- `scripts/regressao_logistica.py` — todas as funções implementadas
- `notebooks/apresentacao.ipynb` — notebook com chamadas e visualizações
- Relatório final (redigido após avaliação do código)

---

## Etapa 1 — Configuração do Ambiente e Dados

**Objetivo:** Preparar o ambiente Python e o dataset para uso nas etapas seguintes.

### 1.1 Estrutura de Pastas
- Criar pastas `dados/`, `scripts/`, `notebooks/`
- Criar `requirements.txt` com as dependências (`numpy`, `matplotlib`, `pandas`)

### 1.2 Escolha e Carregamento do Dataset
- Usar o dataset **"Give Me Some Credit"** (Kaggle) ou o **UCI Credit Card Default Dataset**
- O dataset deve conter features financeiras/comportamentais de clientes e rótulo binário
  de inadimplência (0 = adimplente, 1 = inadimplente)
- Implementar função de carregamento via `pandas` (apenas para leitura do CSV)

### 1.3 Pré-processamento (implementado do zero com numpy)
- Tratar valores ausentes (remoção ou imputação pela mediana)
- Normalização das features: **padronização z-score**
  `x_norm = (x - média) / desvio_padrão`
- Divisão treino/teste (ex.: 80/20) com índices embaralhados manualmente
- Adicionar coluna de bias (coluna de 1s) à matriz de features

**Funções a implementar em `scripts/regressao_logistica.py`:**
```
carregar_dados(caminho)
tratar_ausentes(X, estrategia='mediana')
normalizar_z_score(X) -> (X_norm, media, desvio)
dividir_treino_teste(X, y, proporcao_treino=0.8, semente=42)
adicionar_bias(X)
```

---

## Etapa 2 — Modelo de Regressão Logística

**Objetivo:** Implementar as funções matemáticas do modelo.

### Formulação Matemática

Dado um conjunto de treinamento com `n` exemplos e `p` features:
- `X ∈ R^(n×(p+1))` — matriz de features com coluna de bias
- `y ∈ {0,1}^n` — vetor de rótulos
- `θ ∈ R^(p+1)` — vetor de parâmetros a otimizar

**Probabilidade estimada:**
`σ(z) = 1 / (1 + e^(-z))`, onde `z = Xθ`
`ŷ_i = σ(xᵢᵀθ)`

**Função de log-verossimilhança (a maximizar):**
`ℓ(θ) = Σᵢ [ yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ) ]`

**Função de perda (a minimizar — negativo da log-verossimilhança):**
`L(θ) = -ℓ(θ)`

**Gradiente:**
`∇L(θ) = Xᵀ(ŷ - y)`

**Hessiana:**
`H(θ) = XᵀWX`, onde `W = diag(ŷᵢ(1-ŷᵢ))`

### Funções a implementar:
```
funcao_sigmoide(z)
log_verossimilhanca(theta, X, y)
funcao_perda(theta, X, y)          # negativo da log-verossimilhança
gradiente_perda(theta, X, y)       # ∇L(θ) = Xᵀ(ŷ - y)
hessiana_perda(theta, X, y)        # H(θ) = XᵀWX
prever_probabilidade(theta, X)
prever_classe(theta, X, limiar=0.5)
```

---

## Etapa 3 — Métodos de Busca em Linha (Line Search)

**Objetivo:** Implementar os métodos de busca em linha baseados nos notebooks do professor
para uso nos algoritmos de otimização da Etapa 4.

Esses métodos recebem uma **função unidimensional** `g(α) = L(θ + α·d)` e encontram
o tamanho de passo `α` que minimiza (ou reduz suficientemente) `g`.

### 3.1 Determinação do Intervalo Inicial
Dado um ponto de partida, encontrar automaticamente um intervalo `[a, b]` que contém
um mínimo de `g`.

**Função:**
```
intervalo_inicial(g, alpha_inicial, passo=1e-2, fator=2.0) -> (a, b)
```

### 3.2 Busca por Partição Igual (Trisseção)
Divide o intervalo `[a, b]` em três partes iguais e descarta a parte sem o mínimo,
repetindo até convergir.

**Função:**
```
busca_particao_igual(g, a, b, tolerancia=1e-6) -> (num_iteracoes, alpha_otimo)
```

### 3.3 Busca da Seção Áurea
Usa a razão áurea `φ = (√5-1)/2` para reduzir o intervalo de forma mais eficiente
que a trisseção (fator de redução `φ` por iteração).

**Função:**
```
busca_secao_aurea(g, a, b, num_iteracoes) -> (a, b)
```

### 3.4 Busca por Ajuste Quadrático
Interpola uma parábola pelos três pontos `(a, g(a))`, `(b, g(b))`, `(c, g(c))` e
usa o vértice da parábola como próxima estimativa do mínimo.

**Função:**
```
busca_ajuste_quadratico(g, a, b, c, num_iteracoes) -> (a, b, c)
```

### 3.5 Biseção na Derivada
Aplica o método da biseção sobre a derivada `g'(α)` para encontrar onde `g'(α) = 0`,
ou seja, o ponto de mínimo.

**Funções:**
```
intervalo_bissecao_derivada(dg, a, b, fator=2.0) -> (a, b)
bissecao_derivada(dg, a, b, tolerancia=1e-6) -> (a, b)
```

---

## Etapa 4 — Algoritmos de Otimização

**Objetivo:** Implementar os métodos de otimização iterativos que usam as funções de
busca em linha da Etapa 3 para minimizar `L(θ)`.

### 4.1 Gradiente Descendente com Busca em Linha

Dado `θ_atual`, a direção de descida é `-∇L(θ)`.
A cada iteração:
1. Calcular direção: `d = -∇L(θ_k)`
2. Definir função unidimensional: `g(α) = L(θ_k + α·d)`
3. Encontrar `α*` via busca em linha (seção áurea ou ajuste quadrático)
4. Atualizar: `θ_{k+1} = θ_k + α*·d`
5. Verificar convergência: `‖∇L(θ_{k+1})‖ < tolerância`

**Função:**
```
gradiente_descendente(
    funcao_perda, gradiente_perda,
    theta_inicial, X, y,
    metodo_busca='secao_aurea',
    tolerancia=1e-6, max_iteracoes=1000
) -> (theta_otimo, historico_perda, historico_norma_grad, num_iteracoes)
```

### 4.2 Método de Newton

Usa informação de segunda ordem (Hessiana) para passos mais eficientes.
A cada iteração:
1. Calcular gradiente: `g = ∇L(θ_k)`
2. Calcular Hessiana: `H = H(θ_k)`
3. Resolver sistema: `H·d = -g` → direção de Newton `d`
4. Calcular `α*` via busca em linha aplicada a `g(α) = L(θ_k + α·d)`
5. Atualizar: `θ_{k+1} = θ_k + α*·d`
6. Verificar convergência: `‖∇L(θ_{k+1})‖ < tolerância`

**Função:**
```
metodo_newton(
    funcao_perda, gradiente_perda, hessiana_perda,
    theta_inicial, X, y,
    metodo_busca='secao_aurea',
    tolerancia=1e-6, max_iteracoes=200
) -> (theta_otimo, historico_perda, historico_norma_grad, num_iteracoes)
```

### Observação sobre os métodos de busca em linha
Ambos os métodos de otimização aceitam um parâmetro `metodo_busca` que seleciona
qual método da Etapa 3 será usado internamente para o passo `α*`.

---

## Etapa 5 — Métricas de Avaliação

**Objetivo:** Implementar todas as métricas de avaliação do classificador do zero.

**Funções:**
```
calcular_matriz_confusao(y_real, y_predito) -> (VP, FP, FN, VN)
calcular_acuracia(y_real, y_predito) -> float
calcular_precisao(y_real, y_predito) -> float
calcular_recall(y_real, y_predito) -> float
calcular_f1(y_real, y_predito) -> float
calcular_roc_auc(y_real, probabilidades) -> (taxas_fp, taxas_vp, auc)
```

---

## Etapa 6 — Experimentos e Análise

**Objetivo:** Comparar os métodos de otimização e analisar o modelo treinado.

### 6.1 Comparação dos Métodos de Otimização
- Treinar o modelo com Gradiente Descendente e com Método de Newton
- Para cada método, registrar:
  - Número de iterações até convergência
  - Tempo de execução
  - Valor da perda final
  - Norma do gradiente final
- Plotar curvas de convergência: `L(θ)` e `‖∇L(θ)‖` por iteração

### 6.2 Comparação dos Métodos de Busca em Linha
- Comparar seção áurea vs. ajuste quadrático como sub-rotina de busca
- Analisar número de avaliações de função necessárias

### 6.3 Análise dos Coeficientes
- Identificar as features mais influentes (maiores |θᵢ|)
- Verificar sinal dos coeficientes (associação positiva/negativa com inadimplência)

### 6.4 Avaliação no Conjunto de Teste
- Calcular todas as métricas da Etapa 5
- Plotar curva ROC
- Plotar matriz de confusão

**Funções auxiliares de visualização:**
```
plotar_convergencia(historico_perda, historico_norma_grad, titulo)
plotar_curva_roc(taxas_fp, taxas_vp, auc)
plotar_matriz_confusao(matriz)
plotar_coeficientes(theta, nomes_features)
```

---

## Etapa 7 — Notebook de Apresentação

**Objetivo:** Montar o `notebooks/apresentacao.ipynb` com estrutura didática para
a apresentação em sala.

### Estrutura do Notebook

1. **Introdução** — Contextualização do problema e objetivo
2. **Carregamento e Pré-processamento** — Mostrar o dataset, aplicar pré-processamento
3. **Formulação Matemática** — Células markdown com as equações do modelo
4. **Funções do Modelo** — Demonstrar `sigmoide`, `perda`, `gradiente`, `hessiana`
   com exemplos numéricos simples
5. **Métodos de Busca em Linha** — Demonstrar cada método em uma função simples 1D
   antes de aplicar ao problema real
6. **Treinamento: Gradiente Descendente** — Executar e mostrar convergência
7. **Treinamento: Método de Newton** — Executar e mostrar convergência
8. **Comparação dos Métodos** — Tabela e gráficos comparativos
9. **Avaliação no Conjunto de Teste** — Métricas e gráficos
10. **Conclusões** — Análise dos resultados

---

## Etapa 8 — Relatório Final

**Objetivo:** Redigir o relatório após validação humana do script e notebook.

> Esta etapa só deve ser iniciada após revisão e aprovação do código e dos resultados.

### Estrutura do Relatório

1. **Introdução** — Motivação, contexto da inadimplência bancária, objetivo do trabalho
2. **Formulação do Problema de Otimização** — Modelo de regressão logística,
   função de perda, propriedades (convexidade, diferenciabilidade, existência do mínimo)
3. **Metodologia**
   - Pré-processamento dos dados
   - Métodos de busca em linha implementados
   - Algoritmos de otimização (Gradiente Descendente e Newton)
4. **Implementação** — Descrição do script e do notebook, com trechos de código relevantes
5. **Resultados** — Convergência dos métodos, métricas de classificação, análise
   dos coeficientes
6. **Conclusões** — Comparação entre métodos, limitações, trabalhos futuros
7. **Referências Bibliográficas**

---

## Ordem de Desenvolvimento Recomendada

```
Etapa 1  ->  Etapa 2  ->  Etapa 3  ->  Etapa 4
                                            |
                                        Etapa 5
                                            |
                                        Etapa 6
                                            |
                                        Etapa 7
                                            |
                              [revisão humana do código]
                                            |
                                        Etapa 8
```

## Checklist de Conclusão

- [x] **Etapa 1** — Ambiente configurado, dataset carregado e pré-processado
- [x] **Etapa 2** — Sigmoide, perda, gradiente e hessiana implementados e testados
- [x] **Etapa 3** — Todos os métodos de busca em linha implementados
- [x] **Etapa 4** — Gradiente descendente e Newton implementados e convergindo
- [x] **Etapa 5** — Todas as métricas implementadas
- [x] **Etapa 6** — Experimentos executados, gráficos gerados
- [ ] **Etapa 7** — Notebook organizado e pronto para apresentação
- [ ] **Etapa 8** — Relatório redigido e revisado
