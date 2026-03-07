"""
regressao_logistica.py
======================
Implementação do modelo de Regressão Logística para previsão de inadimplência
bancária, formulado como problema de otimização não linear irrestrito
(maximização da log-verossimilhança).

Disciplina: Otimização Não-Linear — UFPB
Prof. Felipe A. G. Moreno
"""

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
