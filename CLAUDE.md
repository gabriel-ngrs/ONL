# CLAUDE.md — Regras Gerais do Projeto ONL

## Contexto do Projeto

Projeto da disciplina **Otimização Não-Linear** (Prof. Felipe A. G. Moreno — UFPB).
Tema: **Previsão de Inadimplência Bancária** via Regressão Logística formulada como
problema de otimização não linear irrestrito (maximização de log-verossimilhança).

Integrantes: Gabriel Negreiros Saraiva, Júlia Moraes da Silva, Luiz Eduardo de Almeida
Siqueira Silva, Paulo Victor Cordeiro Rufino de Araújo, Pedro Lucas Simões Cabral.

**Prazos:**
- Relatório: 26 de março de 2026
- Apresentação: 30 de março ou 1º de abril de 2026

---

## Estrutura do Projeto

```
ONL/
├── CLAUDE.md
├── PLANO.md
├── Docs/                        # Documentos do professor e avanço do projeto
├── dados/                       # Dataset bruto e processado
├── scripts/
│   └── regressao_logistica.py   # Script principal com todas as funções
└── notebooks/
    └── apresentacao.ipynb       # Notebook de apresentação (chama o script)
```

---

## Regras de Implementação

### Filosofia do Código

- Os algoritmos devem ser implementados **do zero**, sem usar bibliotecas que resolvam o
  problema inteiro de uma vez (ex.: `sklearn.LogisticRegression`, `scipy.optimize.minimize`).
- Bibliotecas de matemática básica são permitidas: `numpy` (álgebra linear, exponencial,
  logaritmo), `math`, `matplotlib` (visualizações).
- Cada função deve resolver **uma única responsabilidade**.

### Estilo do Código Python

- **Variáveis em português**, com nomes completos. Evitar abreviações, exceto quando o
  nome for muito longo (ex.: `prob_estimada` é melhor que `p` ou `prob`).
- **Docstrings obrigatórias** em todas as funções, descrevendo: o que a função faz,
  os parâmetros (`Args:`) e o retorno (`Returns:`).
- **Comentários em português** explicando os passos matemáticos relevantes.
- Separar o script por seções com blocos de comentário:
  ```python
  # =============================================================================
  # SEÇÃO: Nome da Seção
  # =============================================================================
  ```
- Usar type hints nas assinaturas das funções.

### Exemplo de Padrão Esperado

```python
def funcao_sigmoide(z: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula a função sigmoide (função logística).

    A sigmoide mapeia qualquer valor real para o intervalo (0, 1),
    sendo usada para estimar probabilidades na regressão logística.

    Args:
        z: Valor ou vetor de valores reais (combinação linear dos parâmetros).

    Returns:
        Probabilidade estimada no intervalo (0, 1).
    """
    # sigma(z) = 1 / (1 + e^(-z))
    return 1.0 / (1.0 + np.exp(-z))
```

---

## Regras de Commits

- Usar **Conventional Commits** com mensagens em **português**.
- Nunca mencionar ferramentas de IA nas mensagens de commit.
- Formato: `tipo(escopo opcional): descrição curta no imperativo`

### Tipos Aceitos

| Tipo       | Quando usar                                              |
|------------|----------------------------------------------------------|
| `feat`     | Nova função, módulo ou funcionalidade implementada       |
| `fix`      | Correção de bug ou erro matemático                       |
| `docs`     | Criação ou atualização de documentação, docstrings       |
| `refactor` | Reorganização de código sem mudança de comportamento     |
| `test`     | Adição ou correção de testes                             |
| `chore`    | Tarefas de configuração, dependências, estrutura         |
| `style`    | Formatação, padronização de código (sem mudança lógica)  |

### Exemplos

```
feat: adiciona função sigmoide e log-verossimilhança
feat(otimizacao): implementa método do gradiente descendente com busca em linha
fix: corrige cálculo do gradiente da log-verossimilhança
docs: adiciona docstrings nas funções de busca em linha
refactor: reorganiza script em seções com blocos de comentário
chore: adiciona requirements.txt com dependências do projeto
```

---

## Regras Gerais para o Claude

- Não criar arquivos além dos previstos na estrutura acima, a não ser que seja pedido.
- Não usar bibliotecas de otimização prontas (`scipy.optimize`, `sklearn`, etc.).
- Ao implementar funções, sempre conferir se já existe uma versão no script antes de criar uma nova.
- Ao sugerir mudanças no notebook, lembrar que ele é para apresentação: código limpo,
  saídas visíveis, progressão didática.
- Commits devem ser feitos a cada mudança significativa (nova função, correção importante,
  nova seção concluída).

---

## Skills Disponíveis

Skills são capacidades especializadas invocadas com o comando `/nome-da-skill`. Abaixo
estão as aprovadas para este projeto, com instrução clara de quando e onde usar cada uma.

### `python-pro` — Referência passiva de implementação

**O que faz:** Garante código Python idiomático, type-safe e bem estruturado.
Cobre type hints, docstrings no estilo Google, padrões Pythonic e PEP 8.

**Quando invocar:** Antes de implementar qualquer seção nova do script principal
(`regressao_logistica.py`) — especialmente os algoritmos de otimização (Etapas 3, 4 e 5).
Também útil ao revisar funções já escritas que pareçam verbosas ou pouco idiomáticas.

**Onde se aplica:** `scripts/regressao_logistica.py`

**Exemplo de gatilho:**
> "Vamos implementar o gradiente descendente com busca em linha" → invocar `/python-pro`
> antes de começar.

---

### `doc-coauthoring` — Escrita do relatório e do notebook

**O que faz:** Conduz um fluxo estruturado de co-autoria em 3 estágios: coleta de
contexto, refinamento seção por seção, e teste com leitor externo (sub-agente sem
contexto). Evita que o relatório faça sentido apenas para quem já conhece o projeto.

**Quando invocar:** Ao iniciar a escrita do relatório técnico ou ao estruturar o
notebook de apresentação. Também útil para redigir seções matemáticas (derivação do
gradiente, prova de convexidade via Hessiana) que precisam ser acessíveis à banca.

**Onde se aplica:** Relatório final (`.pdf`/`.md`) e `notebooks/apresentacao.ipynb`

**Prazo crítico:** relatório em **26/03/2026**, apresentação em **30/03–01/04/2026**.

**Exemplo de gatilho:**
> "Vamos começar a escrever o relatório" → invocar `/doc-coauthoring` imediatamente.

---

### `the-fool` — Revisão crítica antes da apresentação

**O que faz:** Aplica raciocínio crítico estruturado em 5 modos (Socrático, Dialético,
Pré-mortem, Red team, Auditoria de evidências) para encontrar pontos cegos, premissas
não declaradas e argumentos fracos.

**Quando invocar:** Na semana anterior à apresentação, após a implementação estar
completa. Útil para antecipar perguntas difíceis do professor sobre as escolhas
matemáticas e de implementação.

**Perguntas típicas que a skill ajuda a preparar:**
- Por que busca em linha de Armijo e não passo fixo?
- A Hessiana de 24×24 realmente compensa computacionalmente com 30k amostras?
- Como justificar o critério de parada escolhido?
- Newton converge mais rápido, mas a que custo de memória?

**Onde se aplica:** Revisão geral do projeto antes da apresentação.

**Exemplo de gatilho:**
> "Quero stress-testar nossas escolhas antes da apresentação" → invocar `/the-fool`.

---

### `debugging-wizard` — Diagnóstico de falhas nos algoritmos

**O que faz:** Investiga erros, analisa comportamentos inesperados e encontra causas
raiz. Especialmente útil para falhas silenciosas em algoritmos numéricos (NaN, divergência,
oscilação, convergência prematura).

**Quando invocar:** Se qualquer método de otimização apresentar comportamento estranho:
perda que não decresce, gradiente que explode, Newton que diverge, ou resultados
inconsistentes entre rodadas.

**Onde se aplica:** `scripts/regressao_logistica.py` durante as Etapas 3, 4 e 5.

**Exemplo de gatilho:**
> "O gradiente descendente está divergindo" ou "a perda virou NaN" → invocar
> `/debugging-wizard`.
