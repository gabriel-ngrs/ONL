# Relatório de Bugs — Projeto ONL

**Data da análise:** 2026-03-26
**Branch analisada:** `main` (commit `cbb5384`)
**Arquivos analisados:** `regressao_log.py`, `main.ipynb`, `README.md`

---

## Índice

| # | Severidade | Arquivo / Local | Descrição resumida |
|---|-----------|-----------------|-------------------|
| 1 | CRÍTICO | `regressao_log.py:350` | Código morto sem efeito algum |
| 2 | ALTO | `main.ipynb` — Célula 3 | Caminho relativo de arquivo frágil |
| 3 | MÉDIO | `main.ipynb` — Célula 4 | Dados normalizados criados mas nunca usados |
| 4 | MÉDIO | `main.ipynb` — Células 46, 48, 56, 58 | `lambda_` hard-coded em cálculos manuais |
| 5 | MÉDIO | `main.ipynb` — Células 46–48 | Mesmo modelo, dados de treino diferentes |
| 6 | MÉDIO | `main.ipynb` — Células 51, 53 | Instabilidade numérica no cálculo da taxa de convergência |
| 7 | MENOR | `regressao_log.py:236` | Eigenvalor negativo pode corromper número de condição |
| 8 | MENOR | `regressao_log.py:141` | `getRegressionY()` quebra silenciosamente com ≠ 2 features |
| 9 | MENOR | `main.ipynb` — Células 36–42 | Comparação de modelos com dados de distribuições diferentes |
| 10 | DOC | `README.md:34` | Referência a `notebooks/apresentacao.ipynb` que não existe |
| 11 | DOC | `README.md:71` | `historico_alpha_` e `historico_tempo_` ausentes nos atributos documentados |
| 12 | DOC | `README.md:228` | Diz que normalização z-score é aplicada, mas dados normalizados nunca são usados |

---

## Bugs Detalhados

---

### BUG 1 — Código morto sem efeito (CRÍTICO)

**Arquivo:** `regressao_log.py`
**Linha:** 350
**Método:** `_busca_ajuste_quadratico()`

**Código problemático:**
```python
gx = g(x)
(c, gc, b, gb) if gx > gb else (a, ga, b, gb).__class__   # ← LINHA 350
if gx > gb: c, gc = x, gx
else:        a, ga, b, gb = b, gb, x, gx
```

**Problema:** A linha 350 avalia uma expressão condicional mas não atribui o resultado a nenhuma variável. O primeiro ramo retorna uma tupla `(c, gc, b, gb)`, o segundo retorna a classe `tuple` — nenhum dos dois tem qualquer efeito. As atualizações reais acontecem nas linhas 351–352, fazendo a linha 350 ser completamente inútil.

Parece ser um resquício de uma tentativa de atribuição múltipla não completada, possivelmente uma linha como `c, gc, b, gb = (c, gc, b, gb) if ... else (a, ga, b, gb)` que foi escrita pela metade.

**Impacto:** O método funciona, mas a linha desperdiça CPU e induz a leituras erradas do código.

---

### BUG 2 — Caminho relativo frágil para leitura do CSV (ALTO)

**Arquivo:** `main.ipynb`
**Célula:** 3

**Código problemático:**
```python
df = pd.read_csv('../ONL/dados/credito.csv', na_values=['na', '','NaN','NA','NAN'])
```

**Problema:** O caminho `../ONL/dados/credito.csv` assume que o notebook é executado a partir de um diretório pai específico (`/home/.../projetos/`). Se o kernel for iniciado de outro diretório de trabalho, ou se o repositório for clonado em outro local, ocorrerá `FileNotFoundError`.

**Impacto:** O notebook falha imediatamente para qualquer colaborador cujo diretório de trabalho seja diferente.

---

### BUG 3 — Dados normalizados criados mas nunca usados (MÉDIO)

**Arquivo:** `main.ipynb`
**Célula:** 4

**Código problemático:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Problema:** `X_train_scaled` e `X_test_scaled` são criados mas nenhuma célula subsequente os utiliza. Todo o treinamento e avaliação do modelo usa `X_train` e `X_test` originais (sem normalização). Verificado por busca em todo o notebook.

**Impacto duplo:**
1. Código morto — computação desperdiçada.
2. O README afirma que o pré-processamento inclui "normalização z-score", mas isso **não acontece de fato** no treinamento. Os modelos operam em features não normalizadas, o que pode afetar convergência e comparação entre métodos.

---

### BUG 4 — `lambda_` hard-coded em cálculos manuais (MÉDIO)

**Arquivo:** `main.ipynb`
**Células:** 46, 48, 56, 58

**Código problemático:**
```python
H    = RegressaoLogistica._hessiana(w, X_b_full, 0.01)
grad = RegressaoLogistica._gradiente(w, X_b_full, y_np_full, 0.01)
g    = lambda a: RegressaoLogistica._funcao_perda(w + a * direcao, X_b_full, y_np_full, 0.01)
```

**Problema:** O valor `0.01` de `lambda_` está fixo no código. Se o modelo for instanciado com um `lambda_` diferente, as análises manuais dessas células não corresponderão ao modelo real — calculando Hessiana, gradiente e perda com parâmetro de regularização errado.

**Impacto:** Análises de número de condição e convergência se tornam incorretas caso `lambda_` mude.

---

### BUG 5 — Mesmo modelo analisado com dados diferentes (MÉDIO)

**Arquivo:** `main.ipynb`
**Células:** 46 e 48

**Problema:** Ambas as células analisam o mesmo `modelo_newton`, mas usam bases de dados distintas:

| Célula | Dados usados |
|--------|-------------|
| 46 | `X_train`, `y_train` — dados originais (desbalanceados) |
| 48 | `X_res`, `y_res` — dados balanceados via SMOTE |

O modelo foi treinado em apenas uma dessas distribuições. Analisá-lo com dados diferentes produz Hessiana, gradiente e número de condição que não refletem o estado real do treinamento.

**Impacto:** A análise do número de condição κ(H) nas células 46 e 48 não é diretamente comparável nem consistente.

---

### BUG 6 — Instabilidade numérica no cálculo da taxa de convergência (MÉDIO)

**Arquivo:** `main.ipynb`
**Células:** 51, 53

**Código problemático:**
```python
razao_gd = [grad_gd[k+1] / grad_gd[k] for k in range(len(grad_gd)-1)
            if grad_gd[k] > 1e-15]
```

**Problema:** A condição `if grad_gd[k] > 1e-15` filtra valores muito pequenos no denominador, mas ainda permite divisões por valores muito próximos de zero (ex: `1e-14`), gerando razões enormes e sem sentido matemático. Perto da convergência, onde `grad → 0`, a razão `grad[k+1]/grad[k]` deixa de representar a taxa de convergência real.

**Impacto:** Gráficos e tabelas de taxa de convergência podem exibir valores incorretos ou explodidos próximo ao final da otimização.

---

### BUG 7 — Eigenvalor negativo corrompe número de condição (MENOR)

**Arquivo:** `main.ipynb`
**Células:** 46, 48

**Código problemático:**
```python
avs = np.linalg.eigvalsh(H)
kappas.append(avs.max() / max(avs.min(), 1e-12))
```

**Problema:** Se `avs.min()` for negativo (possível por erros numéricos de ponto flutuante), `max(avs.min(), 1e-12)` devolve o próprio negativo, e a divisão produz um κ(H) negativo — matematicamente inválido, pois o número de condição é sempre ≥ 1.

**Impacto:** Raro em prática, mas pode gerar valores incorretos nos gráficos de número de condição sem nenhum aviso.

---

### BUG 8 — `getRegressionY()` quebra silenciosamente para dados com ≠ 2 features (MENOR)

**Arquivo:** `regressao_log.py`
**Linhas:** 141–151
**Método:** `getRegressionY()`

**Código problemático:**
```python
def getRegressionY(self, regressionX, shift=0):
    self._checar_treinado()
    return (-self.w[0] + shift - self.w[1] * regressionX) / self.w[2]
```

**Problema:** O método acessa diretamente `w[0]`, `w[1]` e `w[2]`, assumindo exatamente 2 features de entrada. Com qualquer outro número de features (o dataset `credito.csv` tem 15), chamá-lo causaria `IndexError` ou retornaria resultado silenciosamente errado se `w` tiver mais elementos.

**Impacto:** O método não é chamado no `main.ipynb` atual, mas está exposto na API pública sem proteção.

---

### BUG 9 — Comparação de métodos com dados de distribuições distintas (MENOR)

**Arquivo:** `main.ipynb`
**Células:** 36–42

**Problema:** Os modelos comparados nas análises de convergência foram treinados em conjuntos com distribuições diferentes:

| Modelo | Dados de treino |
|--------|----------------|
| Gradiente Descendente (original) | `X_train` — desbalanceado |
| Newton + SMOTE | `X_res` — balanceado via SMOTE |
| Newton (puro) | `X_train` — desbalanceado |

Comparar iterações, perda e tempo entre modelos treinados em dados diferentes não é uma comparação justa dos algoritmos — parte da diferença observada pode ser atribuída ao balanceamento, não ao método de otimização.

**Impacto:** Conclusões sobre a superioridade de um método podem estar parcialmente viesadas pela diferença nos dados.

---

## Bugs de Documentação

---

### BUG 10 — README referencia pasta `notebooks/` que não existe (DOC)

**Arquivo:** `README.md`
**Linha:** 34

**Trecho no README:**
```
├── notebooks/
│   └── apresentacao.ipynb   # Notebook de apresentação formal do projeto
```

**Problema:** A pasta `notebooks/` e o arquivo `apresentacao.ipynb` não existem no repositório. O README descreve uma estrutura desatualizada.

---

### BUG 11 — Atributos `historico_alpha_` e `historico_tempo_` não documentados no README (DOC)

**Arquivo:** `README.md`
**Linhas:** 71–76

**Trecho no README:**
```markdown
**Atributos após `.fit()`:**
- `w` — parâmetros ótimos θ*
- `historico_perda_` — valor da perda por iteração
- `historico_norma_grad_` — norma do gradiente por iteração
- `n_iteracoes_` — total de iterações até convergência
```

**Problema:** Os atributos `historico_alpha_` (histórico do passo α por iteração) e `historico_tempo_` (tempo acumulado por iteração) foram adicionados na PR #5 (commit `6f7533b`) mas **não aparecem na documentação do README**. Qualquer pessoa lendo o README não saberá da existência desses atributos.

---

### BUG 12 — README afirma normalização z-score mas ela não é aplicada ao treino (DOC)

**Arquivo:** `README.md`
**Linha:** 228

**Trecho no README:**
```
Pré-processamento: normalização z-score + divisão treino/teste 80/20 (seed=42).
```

**Problema:** Ligado diretamente ao BUG 3. O código cria o `StandardScaler` e gera `X_train_scaled`/`X_test_scaled`, mas **nunca os usa** no treinamento. A documentação descreve um comportamento que não acontece.

---

## Resumo por Severidade

| Severidade | Quantidade | IDs |
|-----------|-----------|-----|
| CRÍTICO | 1 | #1 |
| ALTO | 1 | #2 |
| MÉDIO | 4 | #3, #4, #5, #6 |
| MENOR | 3 | #7, #8, #9 |
| DOCUMENTAÇÃO | 3 | #10, #11, #12 |
| **Total** | **12** | |
