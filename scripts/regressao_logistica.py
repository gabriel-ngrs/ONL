"""
regressao_logistica.py
======================
Implementação do modelo de Regressão Logística para previsão de inadimplência
bancária, formulado como problema de otimização não linear irrestrito
(maximização da log-verossimilhança).

Disciplina: Otimização Não-Linear — UFPB
Prof. Felipe A. G. Moreno
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# SEÇÃO: Pré-processamento dos Dados
# =============================================================================


def carregar_dados(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset de inadimplência a partir de um arquivo CSV.

    Utiliza pandas apenas para leitura do arquivo. A última coluna é
    tratada como vetor de rótulos (y) e as demais como features (X).

    Args:
        caminho: Caminho para o arquivo CSV.

    Returns:
        Tupla (X, y) onde X é a matriz de features (n × p) e y é o vetor
        de rótulos binários (n,), ambos como arrays NumPy float64.
    """
    dados = pd.read_csv(caminho)
    X = dados.iloc[:, :-1].values.astype(np.float64)
    y = dados.iloc[:, -1].values.astype(np.float64)
    return X, y


def tratar_ausentes(X: np.ndarray, estrategia: str = 'mediana') -> np.ndarray:
    """
    Substitui valores ausentes (NaN) nas features pela mediana ou média da coluna.

    Args:
        X: Matriz de features (n × p), podendo conter valores NaN.
        estrategia: 'mediana' ou 'media' — define o valor de imputação por coluna.

    Returns:
        Cópia de X com os valores ausentes substituídos.
    """
    X_tratado = X.copy()
    for j in range(X_tratado.shape[1]):
        coluna = X_tratado[:, j]
        mascara_ausentes = np.isnan(coluna)
        if mascara_ausentes.any():
            if estrategia == 'mediana':
                valor_imputacao = np.nanmedian(coluna)
            else:
                valor_imputacao = np.nanmean(coluna)
            X_tratado[mascara_ausentes, j] = valor_imputacao
    return X_tratado


def normalizar_z_score(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza as features pela padronização z-score.

    Fórmula: x_norm = (x - μ) / σ

    Colunas com desvio padrão zero (constantes) não são alteradas para
    evitar divisão por zero — situação típica da coluna de bias.

    Args:
        X: Matriz de features (n × p) sem coluna de bias.

    Returns:
        Tupla (X_norm, media, desvio_padrao) onde X_norm é a matriz
        normalizada e media/desvio_padrao são vetores (p,) necessários
        para normalizar o conjunto de teste com os mesmos parâmetros.
    """
    media = np.mean(X, axis=0)
    desvio_padrao = np.std(X, axis=0)

    # Substitui desvio zero por 1 para evitar divisão por zero
    desvio_seguro = np.where(desvio_padrao == 0, 1.0, desvio_padrao)

    X_norm = (X - media) / desvio_seguro
    return X_norm, media, desvio_padrao


def dividir_treino_teste(
    X: np.ndarray,
    y: np.ndarray,
    proporcao_treino: float = 0.8,
    semente: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide features e rótulos em conjuntos de treino e teste com embaralhamento.

    O embaralhamento é implementado manualmente via NumPy, sem uso de
    bibliotecas de machine learning.

    Args:
        X: Matriz de features (n × p).
        y: Vetor de rótulos (n,).
        proporcao_treino: Fração dos dados para treino (padrão: 0.8).
        semente: Semente aleatória para reprodutibilidade.

    Returns:
        Tupla (X_treino, X_teste, y_treino, y_teste).
    """
    numero_amostras = X.shape[0]
    gerador = np.random.default_rng(semente)
    indices_embaralhados = gerador.permutation(numero_amostras)

    corte = int(numero_amostras * proporcao_treino)
    indices_treino = indices_embaralhados[:corte]
    indices_teste = indices_embaralhados[corte:]

    return (
        X[indices_treino], X[indices_teste],
        y[indices_treino], y[indices_teste],
    )


def adicionar_bias(X: np.ndarray) -> np.ndarray:
    """
    Adiciona uma coluna de uns à esquerda da matriz de features.

    A coluna de bias permite que o modelo aprenda um intercepto θ₀,
    tornando o vetor de parâmetros θ ∈ R^(p+1).

    Args:
        X: Matriz de features (n × p).

    Returns:
        Matriz aumentada (n × (p+1)) com coluna de 1s na primeira posição.
    """
    coluna_bias = np.ones((X.shape[0], 1))
    return np.hstack([coluna_bias, X])


# =============================================================================
# SEÇÃO: Modelo de Regressão Logística
# =============================================================================


def funcao_sigmoide(z: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula a função sigmoide (função logística).

    A sigmoide mapeia qualquer valor real para o intervalo (0, 1),
    sendo usada para estimar probabilidades na regressão logística.
    O argumento é limitado ao intervalo [-500, 500] para evitar
    overflow numérico em np.exp.

    Args:
        z: Valor ou vetor de valores reais (combinação linear dos parâmetros).

    Returns:
        Probabilidade estimada no intervalo (0, 1).
    """
    # Limitação numérica para evitar overflow no cálculo de e^(-z)
    z_limitado = np.clip(z, -500.0, 500.0)
    # σ(z) = 1 / (1 + e^(-z))
    return 1.0 / (1.0 + np.exp(-z_limitado))


def log_verossimilhanca(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """
    Calcula a função de log-verossimilhança da regressão logística.

    ℓ(θ) = Σᵢ [ yᵢ·log(ŷᵢ) + (1 - yᵢ)·log(1 - ŷᵢ) ]

    onde ŷᵢ = σ(xᵢᵀθ) é a probabilidade estimada pelo modelo.
    As probabilidades são limitadas a (ε, 1-ε) para evitar log(0).

    Args:
        theta: Vetor de parâmetros (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,).

    Returns:
        Valor escalar da log-verossimilhança (maior = melhor ajuste).
    """
    epsilon = 1e-15
    prob_estimada = funcao_sigmoide(X @ theta)

    # Limita probabilidades para estabilidade numérica nos logaritmos
    prob_estimada = np.clip(prob_estimada, epsilon, 1.0 - epsilon)

    # ℓ(θ) = Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
    return float(
        np.sum(y * np.log(prob_estimada) + (1.0 - y) * np.log(1.0 - prob_estimada))
    )


def funcao_perda(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """
    Calcula a função de perda (negativo da log-verossimilhança).

    L(θ) = -ℓ(θ)

    Minimizar L(θ) é equivalente a maximizar a log-verossimilhança.
    Esta formulação permite o uso direto de algoritmos de minimização.

    Args:
        theta: Vetor de parâmetros (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,).

    Returns:
        Valor escalar da função de perda (menor = melhor ajuste).
    """
    # L(θ) = -ℓ(θ)
    return -log_verossimilhanca(theta, X, y)


def gradiente_perda(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calcula o gradiente da função de perda em relação a θ.

    ∇L(θ) = Xᵀ(ŷ - y)

    onde ŷᵢ = σ(xᵢᵀθ). O gradiente aponta a direção de maior crescimento
    da perda; a direção de descida é -∇L(θ).

    Args:
        theta: Vetor de parâmetros (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,).

    Returns:
        Vetor gradiente (p+1,).
    """
    prob_estimada = funcao_sigmoide(X @ theta)
    # Erro de predição: diferença entre probabilidade estimada e rótulo real
    erro = prob_estimada - y
    # ∇L(θ) = Xᵀ(ŷ - y)
    return X.T @ erro


def hessiana_perda(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Calcula a matriz Hessiana da função de perda em relação a θ.

    H(θ) = XᵀWX

    onde W = diag(ŷᵢ(1 - ŷᵢ)) é a matriz diagonal de pesos, com
    ŷᵢ = σ(xᵢᵀθ). A Hessiana é positiva semi-definida, o que confirma
    a convexidade global da função de perda.

    Args:
        theta: Vetor de parâmetros (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,) — incluído por consistência da API.

    Returns:
        Matriz Hessiana (p+1) × (p+1).
    """
    prob_estimada = funcao_sigmoide(X @ theta)
    # Pesos: wᵢ = ŷᵢ(1 - ŷᵢ) — diagonal da matriz W
    pesos = prob_estimada * (1.0 - prob_estimada)
    # H = XᵀWX — equivalente a X.T @ diag(pesos) @ X, porém eficiente
    return X.T @ (pesos[:, np.newaxis] * X)


def prever_probabilidade(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Estima a probabilidade de inadimplência para cada exemplo.

    Calcula ŷᵢ = σ(xᵢᵀθ) para cada linha de X.

    Args:
        theta: Vetor de parâmetros treinados (p+1,).
        X: Matriz de features com bias (n × (p+1)).

    Returns:
        Vetor de probabilidades estimadas (n,) no intervalo (0, 1).
    """
    return funcao_sigmoide(X @ theta)


def prever_classe(
    theta: np.ndarray,
    X: np.ndarray,
    limiar: float = 0.5,
) -> np.ndarray:
    """
    Classifica cada exemplo como adimplente (0) ou inadimplente (1).

    Aplica um limiar sobre a probabilidade estimada:
        ŷᵢ = 1 se σ(xᵢᵀθ) ≥ limiar, e 0 caso contrário.

    Args:
        theta: Vetor de parâmetros treinados (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        limiar: Limiar de decisão (padrão: 0.5).

    Returns:
        Vetor de predições binárias (n,) com valores 0.0 ou 1.0.
    """
    prob_estimada = prever_probabilidade(theta, X)
    return (prob_estimada >= limiar).astype(np.float64)


# =============================================================================
# SEÇÃO: Métodos de Busca em Linha (Line Search)
# =============================================================================


def intervalo_inicial(
    g: Callable[[float], float],
    alpha_inicial: float = 0.0,
    passo: float = 1e-2,
    fator: float = 2.0,
    max_iteracoes: int = 200,
) -> tuple[float, float]:
    """
    Determina automaticamente um intervalo [a, b] que contém um mínimo de g.

    Parte de alpha_inicial e avança na direção de decrescimento de g,
    dobrando o passo a cada iteração até que g comece a crescer.

    Args:
        g: Função unidimensional g(α) a ser minimizada.
        alpha_inicial: Ponto de partida da busca (padrão: 0.0).
        passo: Tamanho do passo inicial (padrão: 1e-2).
        fator: Fator de ampliação do passo a cada iteração (padrão: 2.0).
        max_iteracoes: Número máximo de expansões do passo (padrão: 200).

    Returns:
        Tupla (a, b) com a ≤ b, contendo um mínimo de g no interior.
    """
    a = alpha_inicial
    ga = g(a)
    b = a + passo
    gb = g(b)

    # Se g(a) < g(b), inverte a direção: busca para a esquerda
    if ga < gb:
        a, b = b, a
        ga, gb = gb, ga
        passo = -passo

    for _ in range(max_iteracoes):
        c = b + passo
        gc = g(c)
        if gb < gc:
            # g começa a crescer: o mínimo está entre a e c
            return (a, c) if a < c else (c, a)
        a, ga, b, gb = b, gb, c, gc
        passo *= fator

    return (a, b) if a < b else (b, a)


def busca_particao_igual(
    g: Callable[[float], float],
    a: float,
    b: float,
    tolerancia: float = 1e-6,
) -> tuple[int, float]:
    """
    Minimiza g em [a, b] por partição em três partes iguais (trisseção).

    A cada iteração, o intervalo é dividido em três partes iguais pelos
    pontos u = a + (b-a)/3 e v = a + 2(b-a)/3. A terça parte sem o mínimo
    é descartada, reduzindo o intervalo por um fator de 2/3 por iteração.

    Args:
        g: Função unidimensional g(α) a ser minimizada.
        a: Extremo esquerdo do intervalo.
        b: Extremo direito do intervalo.
        tolerancia: Largura mínima do intervalo (critério de parada).

    Returns:
        Tupla (num_iteracoes, alpha_otimo) com o número de iterações e
        o ponto central do intervalo final.
    """
    num_iteracoes = 0
    while b - a >= tolerancia:
        u = a + (b - a) / 3.0
        v = a + 2.0 * (b - a) / 3.0
        if g(u) < g(v):
            b = v
        else:
            a = u
        num_iteracoes += 1
    return num_iteracoes, (a + b) / 2.0


def busca_secao_aurea(
    g: Callable[[float], float],
    a: float,
    b: float,
    num_iteracoes: int = 50,
) -> tuple[float, float]:
    """
    Minimiza g em [a, b] pelo método da seção áurea.

    Usa a razão áurea φ = (√5 - 1)/2 ≈ 0.618 para posicionar dois pontos
    interiores e descartar a porção do intervalo sem o mínimo a cada passo.
    A redução por iteração é φ ≈ 0.618, mais eficiente que a trisseção (2/3).

    Args:
        g: Função unidimensional g(α) a ser minimizada.
        a: Extremo esquerdo do intervalo inicial.
        b: Extremo direito do intervalo inicial.
        num_iteracoes: Número fixo de iterações (padrão: 50).

    Returns:
        Tupla (a, b) com o intervalo reduzido contendo o mínimo.
    """
    # Razão áurea conjugada: φ = (√5 - 1)/2 ≈ 0.618
    rho = (np.sqrt(5.0) - 1.0) / 2.0

    # Dois pontos interiores iniciais
    c = b - rho * (b - a)  # ponto esquerdo interior
    d = a + rho * (b - a)  # ponto direito interior
    gc, gd = g(c), g(d)

    for _ in range(num_iteracoes - 2):
        if gc < gd:
            # Mínimo em [a, d]: descarta parte direita
            b, d, gd = d, c, gc
            c = b - rho * (b - a)
            gc = g(c)
        else:
            # Mínimo em [c, b]: descarta parte esquerda
            a, c, gc = c, d, gd
            d = a + rho * (b - a)
            gd = g(d)

    return a, b


def busca_ajuste_quadratico(
    g: Callable[[float], float],
    a: float,
    b: float,
    c: float,
    num_iteracoes: int = 50,
) -> tuple[float, float, float]:
    """
    Minimiza g interpolando uma parábola pelos pontos (a, g(a)), (b, g(b)), (c, g(c)).

    A cada iteração, o vértice x* da parábola interpoladora é calculado.
    O pior dos três pontos é substituído por x*, mantendo o triplete ordenado
    em torno do melhor ponto b (menor valor de g).

    Fórmula do vértice:
        x* = 0.5 · [g(a)(b²-c²) + g(b)(c²-a²) + g(c)(a²-b²)] /
                    [g(a)(b-c)  + g(b)(c-a)  + g(c)(a-b)]

    Args:
        g: Função unidimensional g(α) a ser minimizada.
        a: Primeiro ponto (extremo esquerdo).
        b: Segundo ponto (extremo direito).
        c: Terceiro ponto (ponto médio inicial).
        num_iteracoes: Número fixo de iterações (padrão: 50).

    Returns:
        Tupla (a, b, c) onde b é o melhor ponto encontrado (menor g(b)).
    """
    ga, gb, gc = g(a), g(b), g(c)

    for _ in range(num_iteracoes - 3):
        denominador = ga * (b - c) + gb * (c - a) + gc * (a - b)
        if abs(denominador) < 1e-14:
            break
        # Vértice x* da parábola interpoladora pelos três pontos
        x = 0.5 * (
            ga * (b**2 - c**2) + gb * (c**2 - a**2) + gc * (a**2 - b**2)
        ) / denominador
        gx = g(x)

        if x > b:
            if gx > gb:
                c, gc = x, gx      # x pior que b e à direita: tighten direita
            else:
                a, ga, b, gb = b, gb, x, gx  # x melhor que b e à direita: b sobe
        elif x < b:
            if gx > gb:
                a, ga = x, gx      # x pior que b e à esquerda: tighten esquerda
            else:
                c, gc, b, gb = b, gb, x, gx  # x melhor que b e à esquerda: b desce

    return a, b, c


def intervalo_bissecao_derivada(
    dg: Callable[[float], float],
    a: float,
    b: float,
    fator: float = 2.0,
    max_iteracoes: int = 200,
) -> tuple[float, float]:
    """
    Expande o intervalo [a, b] até que dg(a) e dg(b) tenham sinais opostos.

    Garante a existência de uma raiz de dg (mínimo de g) no interior do
    intervalo, pré-condição para a aplicação da biseção na derivada.

    Args:
        dg: Derivada unidimensional g'(α).
        a: Extremo inicial esquerdo.
        b: Extremo inicial direito.
        fator: Fator de expansão do intervalo a cada iteração (padrão: 2.0).
        max_iteracoes: Número máximo de expansões (padrão: 200).

    Returns:
        Tupla (a, b) com a ≤ b, tal que dg(a)·dg(b) ≤ 0.
    """
    if a > b:
        a, b = b, a
    centro = (a + b) / 2.0
    metade = (b - a) / 2.0

    for _ in range(max_iteracoes):
        if dg(a) * dg(b) <= 0.0:
            break
        metade *= fator
        a = centro - metade
        b = centro + metade

    return a, b


def bissecao_derivada(
    dg: Callable[[float], float],
    a: float,
    b: float,
    tolerancia: float = 1e-6,
) -> tuple[float, float]:
    """
    Encontra a raiz de dg em [a, b] pelo método da biseção.

    Localiza o mínimo de g determinando onde g'(α) = 0. Requer que
    dg(a) e dg(b) tenham sinais opostos (garantido por intervalo_bissecao_derivada).

    Args:
        dg: Derivada unidimensional g'(α).
        a: Extremo esquerdo (com dg(a) e dg(b) de sinais opostos).
        b: Extremo direito.
        tolerancia: Largura mínima do intervalo (critério de parada).

    Returns:
        Tupla (a, b) com o intervalo reduzido contendo a raiz de dg.
    """
    if a > b:
        a, b = b, a

    dga = dg(a)
    if dga == 0.0:
        return a, a
    if dg(b) == 0.0:
        return b, b

    while b - a > tolerancia:
        x = (a + b) / 2.0
        dx = dg(x)
        if dx == 0.0:
            return x, x
        if np.sign(dx) == np.sign(dga):
            a = x
            dga = dx
        else:
            b = x

    return a, b


# =============================================================================
# SEÇÃO: Algoritmos de Otimização
# =============================================================================


def _aplicar_busca_linha(
    g: Callable[[float], float],
    a: float,
    b: float,
    metodo_busca: str,
    num_iteracoes_busca: int = 50,
) -> float:
    """
    Aplica o método de busca em linha selecionado e retorna o passo ótimo α*.

    Função auxiliar interna usada por gradiente_descendente e metodo_newton
    para desacoplar a escolha do método de busca da lógica de otimização.

    Args:
        g: Função unidimensional g(α) = L(θ + α·d).
        a: Extremo esquerdo do intervalo que contém o mínimo.
        b: Extremo direito do intervalo que contém o mínimo.
        metodo_busca: Um de 'secao_aurea', 'ajuste_quadratico', 'particao_igual'.
        num_iteracoes_busca: Número de iterações/avaliações do método (padrão: 50).

    Returns:
        Passo ótimo α* que minimiza g no intervalo [a, b].
    """
    if metodo_busca == 'secao_aurea':
        a_final, b_final = busca_secao_aurea(g, a, b, num_iteracoes_busca)
        return (a_final + b_final) / 2.0

    elif metodo_busca == 'ajuste_quadratico':
        _, alpha_otimo, _ = busca_ajuste_quadratico(
            g, a, b, (a + b) / 2.0, num_iteracoes_busca
        )
        return alpha_otimo

    elif metodo_busca == 'particao_igual':
        _, alpha_otimo = busca_particao_igual(g, a, b)
        return alpha_otimo

    else:
        raise ValueError(f"Método de busca desconhecido: '{metodo_busca}'.")


def gradiente_descendente(
    funcao_perda: Callable,
    gradiente_perda: Callable,
    theta_inicial: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metodo_busca: str = 'secao_aurea',
    tolerancia: float = 1e-6,
    max_iteracoes: int = 1000,
) -> tuple[np.ndarray, list[float], list[float], int]:
    """
    Minimiza L(θ) pelo método do gradiente descendente com busca em linha.

    A cada iteração k:
        1. Calcula direção: d = -∇L(θₖ)
        2. Define g(α) = L(θₖ + α·d)
        3. Encontra α* via busca em linha no intervalo que contém o mínimo de g
        4. Atualiza: θₖ₊₁ = θₖ + α*·d
        5. Para se ‖∇L(θₖ₊₁)‖ < tolerância

    Args:
        funcao_perda: Callable(theta, X, y) → float — função L(θ).
        gradiente_perda: Callable(theta, X, y) → ndarray — gradiente ∇L(θ).
        theta_inicial: Vetor de parâmetros inicial (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,).
        metodo_busca: Método de busca em linha ('secao_aurea', 'ajuste_quadratico',
            'particao_igual'). Padrão: 'secao_aurea'.
        tolerancia: Critério de parada na norma do gradiente (padrão: 1e-6).
        max_iteracoes: Número máximo de iterações (padrão: 1000).

    Returns:
        Tupla (theta_otimo, historico_perda, historico_norma_grad, num_iteracoes).
    """
    theta = theta_inicial.copy()
    historico_perda: list[float] = []
    historico_norma_grad: list[float] = []

    for k in range(max_iteracoes):
        grad = gradiente_perda(theta, X, y)
        norma_grad = float(np.linalg.norm(grad))
        historico_perda.append(funcao_perda(theta, X, y))
        historico_norma_grad.append(norma_grad)

        if norma_grad < tolerancia:
            return theta, historico_perda, historico_norma_grad, k

        # Direção de descida: oposto do gradiente
        direcao = -grad

        # Função unidimensional: g(α) = L(θₖ + α·d)
        def g_alpha(alpha: float, _t: np.ndarray = theta, _d: np.ndarray = direcao) -> float:
            return funcao_perda(_t + alpha * _d, X, y)

        # Intervalo que contém o mínimo de g, depois aplica busca em linha
        a, b = intervalo_inicial(g_alpha, alpha_inicial=0.0)
        alpha_otimo = _aplicar_busca_linha(g_alpha, a, b, metodo_busca)

        # Atualização: θₖ₊₁ = θₖ + α*·d
        theta = theta + alpha_otimo * direcao

    # Registra o estado final se atingiu max_iteracoes sem convergir
    grad_final = gradiente_perda(theta, X, y)
    historico_perda.append(funcao_perda(theta, X, y))
    historico_norma_grad.append(float(np.linalg.norm(grad_final)))

    return theta, historico_perda, historico_norma_grad, max_iteracoes


def metodo_newton(
    funcao_perda: Callable,
    gradiente_perda: Callable,
    hessiana_perda: Callable,
    theta_inicial: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metodo_busca: str = 'secao_aurea',
    tolerancia: float = 1e-6,
    max_iteracoes: int = 200,
) -> tuple[np.ndarray, list[float], list[float], int]:
    """
    Minimiza L(θ) pelo método de Newton com busca em linha.

    Usa informação de segunda ordem (Hessiana) para calcular a direção de
    Newton, que converge quadraticamente perto do mínimo.

    A cada iteração k:
        1. Calcula gradiente: g = ∇L(θₖ)
        2. Calcula Hessiana: H = H(θₖ)
        3. Resolve: H·d = -g  → direção de Newton d
        4. Define g(α) = L(θₖ + α·d)
        5. Encontra α* via busca em linha
        6. Atualiza: θₖ₊₁ = θₖ + α*·d
        7. Para se ‖∇L(θₖ₊₁)‖ < tolerância

    Args:
        funcao_perda: Callable(theta, X, y) → float — função L(θ).
        gradiente_perda: Callable(theta, X, y) → ndarray — gradiente ∇L(θ).
        hessiana_perda: Callable(theta, X, y) → ndarray — Hessiana H(θ).
        theta_inicial: Vetor de parâmetros inicial (p+1,).
        X: Matriz de features com bias (n × (p+1)).
        y: Vetor de rótulos binários (n,).
        metodo_busca: Método de busca em linha (padrão: 'secao_aurea').
        tolerancia: Critério de parada na norma do gradiente (padrão: 1e-6).
        max_iteracoes: Número máximo de iterações (padrão: 200).

    Returns:
        Tupla (theta_otimo, historico_perda, historico_norma_grad, num_iteracoes).
    """
    theta = theta_inicial.copy()
    historico_perda: list[float] = []
    historico_norma_grad: list[float] = []

    for k in range(max_iteracoes):
        grad = gradiente_perda(theta, X, y)
        norma_grad = float(np.linalg.norm(grad))
        historico_perda.append(funcao_perda(theta, X, y))
        historico_norma_grad.append(norma_grad)

        if norma_grad < tolerancia:
            return theta, historico_perda, historico_norma_grad, k

        # Hessiana com regularização de Tikhonov para estabilidade numérica
        H = hessiana_perda(theta, X, y)
        H_reg = H + 1e-8 * np.eye(H.shape[0])

        try:
            # Direção de Newton: resolve H·d = -∇L(θ)
            direcao = np.linalg.solve(H_reg, -grad)
        except np.linalg.LinAlgError:
            # Fallback para descida do gradiente se H for singular
            direcao = -grad

        # Função unidimensional: g(α) = L(θₖ + α·d)
        def g_alpha(alpha: float, _t: np.ndarray = theta, _d: np.ndarray = direcao) -> float:
            return funcao_perda(_t + alpha * _d, X, y)

        # Intervalo que contém o mínimo de g, depois aplica busca em linha
        a, b = intervalo_inicial(g_alpha, alpha_inicial=0.0)
        alpha_otimo = _aplicar_busca_linha(g_alpha, a, b, metodo_busca)

        # Atualização: θₖ₊₁ = θₖ + α*·d
        theta = theta + alpha_otimo * direcao

    # Registra o estado final se atingiu max_iteracoes sem convergir
    grad_final = gradiente_perda(theta, X, y)
    historico_perda.append(funcao_perda(theta, X, y))
    historico_norma_grad.append(float(np.linalg.norm(grad_final)))

    return theta, historico_perda, historico_norma_grad, max_iteracoes


# =============================================================================
# SEÇÃO: Métricas de Avaliação
# =============================================================================


def calcular_matriz_confusao(
    y_real: np.ndarray,
    y_predito: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Calcula os quatro componentes da matriz de confusão.

    Classificação binária:
        VP (Verdadeiro Positivo): predito=1 e real=1
        FP (Falso Positivo):     predito=1 e real=0
        FN (Falso Negativo):     predito=0 e real=1
        VN (Verdadeiro Negativo): predito=0 e real=0

    Args:
        y_real: Vetor de rótulos verdadeiros (n,).
        y_predito: Vetor de predições binárias (n,).

    Returns:
        Tupla (VP, FP, FN, VN) com os quatro contagens inteiras.
    """
    y_r = y_real.astype(int)
    y_p = y_predito.astype(int)

    VP = int(np.sum((y_p == 1) & (y_r == 1)))
    FP = int(np.sum((y_p == 1) & (y_r == 0)))
    FN = int(np.sum((y_p == 0) & (y_r == 1)))
    VN = int(np.sum((y_p == 0) & (y_r == 0)))

    return VP, FP, FN, VN


def calcular_acuracia(
    y_real: np.ndarray,
    y_predito: np.ndarray,
) -> float:
    """
    Calcula a acurácia do classificador.

    Acurácia = (VP + VN) / (VP + FP + FN + VN)

    Args:
        y_real: Vetor de rótulos verdadeiros (n,).
        y_predito: Vetor de predições binárias (n,).

    Returns:
        Acurácia no intervalo [0, 1].
    """
    VP, FP, FN, VN = calcular_matriz_confusao(y_real, y_predito)
    return (VP + VN) / (VP + FP + FN + VN)


def calcular_precisao(
    y_real: np.ndarray,
    y_predito: np.ndarray,
) -> float:
    """
    Calcula a precisão (valor preditivo positivo) do classificador.

    Precisão = VP / (VP + FP)

    Retorna 0.0 se nenhum exemplo for predito como positivo.

    Args:
        y_real: Vetor de rótulos verdadeiros (n,).
        y_predito: Vetor de predições binárias (n,).

    Returns:
        Precisão no intervalo [0, 1].
    """
    VP, FP, FN, VN = calcular_matriz_confusao(y_real, y_predito)
    if VP + FP == 0:
        return 0.0
    return VP / (VP + FP)


def calcular_recall(
    y_real: np.ndarray,
    y_predito: np.ndarray,
) -> float:
    """
    Calcula o recall (sensibilidade) do classificador.

    Recall = VP / (VP + FN)

    Retorna 0.0 se não houver nenhum positivo real.

    Args:
        y_real: Vetor de rótulos verdadeiros (n,).
        y_predito: Vetor de predições binárias (n,).

    Returns:
        Recall no intervalo [0, 1].
    """
    VP, FP, FN, VN = calcular_matriz_confusao(y_real, y_predito)
    if VP + FN == 0:
        return 0.0
    return VP / (VP + FN)


def calcular_f1(
    y_real: np.ndarray,
    y_predito: np.ndarray,
) -> float:
    """
    Calcula o F1-score, média harmônica entre precisão e recall.

    F1 = 2 · (precisão · recall) / (precisão + recall)

    Retorna 0.0 se precisão e recall forem ambos zero.

    Args:
        y_real: Vetor de rótulos verdadeiros (n,).
        y_predito: Vetor de predições binárias (n,).

    Returns:
        F1-score no intervalo [0, 1].
    """
    precisao = calcular_precisao(y_real, y_predito)
    recall = calcular_recall(y_real, y_predito)
    if precisao + recall == 0.0:
        return 0.0
    return 2.0 * precisao * recall / (precisao + recall)


def calcular_roc_auc(
    y_real: np.ndarray,
    probabilidades: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calcula a curva ROC e a área sob ela (AUC) sem uso de bibliotecas externas.

    Ordena os exemplos por probabilidade decrescente e percorre os limiares,
    acumulando as taxas de verdadeiros e falsos positivos. A AUC é calculada
    pela regra do trapézio.

    Args:
        y_real: Vetor de rótulos verdadeiros binários (n,).
        probabilidades: Vetor de probabilidades estimadas (n,), em [0, 1].

    Returns:
        Tupla (taxas_fp, taxas_vp, auc) onde taxas_fp e taxas_vp são arrays
        com os pontos da curva ROC e auc é o escalar AUC ∈ [0, 1].
    """
    # Ordena exemplos por probabilidade decrescente (limiar mais alto primeiro)
    indices_ordenados = np.argsort(probabilidades)[::-1]
    y_ordenado = y_real[indices_ordenados]

    total_positivos = int(y_real.sum())
    total_negativos = len(y_real) - total_positivos

    taxas_vp = [0.0]
    taxas_fp = [0.0]
    vp, fp = 0, 0

    for rotulo in y_ordenado:
        if rotulo == 1:
            vp += 1
        else:
            fp += 1
        taxas_vp.append(vp / total_positivos)
        taxas_fp.append(fp / total_negativos)

    taxas_vp_arr = np.array(taxas_vp)
    taxas_fp_arr = np.array(taxas_fp)

    # AUC via regra do trapézio: integral de TPR em função de FPR
    auc = float(np.trapezoid(taxas_vp_arr, taxas_fp_arr))

    return taxas_fp_arr, taxas_vp_arr, auc


# =============================================================================
# SEÇÃO: Visualizações
# =============================================================================


def plotar_convergencia(
    historico_perda: list[float],
    historico_norma_grad: list[float],
    titulo: str = 'Convergência do Método de Otimização',
) -> None:
    """
    Plota as curvas de convergência da perda e da norma do gradiente.

    Dois subplots lado a lado: evolução de L(θ) por iteração (escala linear)
    e evolução de ‖∇L(θ)‖ por iteração (escala logarítmica).

    Args:
        historico_perda: Lista de valores de L(θ) por iteração.
        historico_norma_grad: Lista de ‖∇L(θ)‖ por iteração.
        titulo: Título geral da figura.
    """
    iteracoes = range(len(historico_perda))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iteracoes, historico_perda, color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Perda L(θ)')
    ax1.set_title('Evolução da Função de Perda')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(iteracoes, historico_norma_grad, color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('‖∇L(θ)‖  (escala log)')
    ax2.set_title('Evolução da Norma do Gradiente')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plotar_curva_roc(
    taxas_fp: np.ndarray,
    taxas_vp: np.ndarray,
    auc: float,
) -> None:
    """
    Plota a curva ROC com a linha de referência do classificador aleatório.

    Args:
        taxas_fp: Array com as taxas de falsos positivos (eixo x).
        taxas_vp: Array com as taxas de verdadeiros positivos (eixo y).
        auc: Área sob a curva ROC (exibida na legenda).
    """
    plt.figure(figsize=(6, 6))
    plt.plot(taxas_fp, taxas_vp, color='steelblue', linewidth=2,
             label=f'Modelo (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
             label='Classificador aleatório (AUC = 0.50)')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotar_matriz_confusao(
    matriz: tuple[int, int, int, int],
) -> None:
    """
    Plota a matriz de confusão como um mapa de calor 2×2.

    Layout:
        [[VN  FP],
         [FN  VP]]

    Args:
        matriz: Tupla (VP, FP, FN, VN) retornada por calcular_matriz_confusao.
    """
    VP, FP, FN, VN = matriz
    valores = np.array([[VN, FP], [FN, VP]])
    rotulos = [['VN', 'FP'], ['FN', 'VP']]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(valores, cmap='Blues')

    for i in range(2):
        for j in range(2):
            cor_texto = 'white' if valores[i, j] > valores.max() / 2.0 else 'black'
            ax.text(j, i, f'{rotulos[i][j]}\n{valores[i, j]}',
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color=cor_texto)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predito: 0', 'Predito: 1'])
    ax.set_yticklabels(['Real: 0', 'Real: 1'])
    ax.set_title('Matriz de Confusão')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plotar_coeficientes(
    theta: np.ndarray,
    nomes_features: list[str],
) -> None:
    """
    Plota os coeficientes θ em gráfico de barras horizontais, ordenados por |θᵢ|.

    O coeficiente de bias (θ₀) é excluído da visualização. Barras em azul
    indicam coeficientes positivos (associação positiva com inadimplência)
    e em vermelho indicam coeficientes negativos.

    Args:
        theta: Vetor de parâmetros treinados (p+1,), incluindo o bias.
        nomes_features: Lista com os nomes das p features (sem o bias).
    """
    # Exclui o intercepto θ₀
    coeficientes = theta[1:]
    indices_ordenados = np.argsort(np.abs(coeficientes))
    coef_ordenado = coeficientes[indices_ordenados]
    nomes_ordenados = [nomes_features[i] for i in indices_ordenados]
    cores = ['tomato' if v < 0 else 'steelblue' for v in coef_ordenado]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(coef_ordenado)), coef_ordenado, color=cores)
    ax.set_yticks(range(len(coef_ordenado)))
    ax.set_yticklabels(nomes_ordenados, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coeficiente θᵢ')
    ax.set_title('Importância das Features (por magnitude do coeficiente)')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
