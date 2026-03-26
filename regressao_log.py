from __future__ import annotations
from typing import Callable
import numpy as np


class RegressaoLogistica:
    """
    Regressão Logística para Previsão de Inadimplência Bancária.

    Parâmetros
    metodo_otimizacao : str
        'gradiente_descendente' ou 'newton'. Padrão: 'gradiente_descendente'.
    metodo_busca : str
        'secao_aurea', 'particao_igual' ou 'ajuste_quadratico'.
        Padrão: 'secao_aurea'.
    tmax : int
        Número máximo de iterações (equivalente ao tmax original).
    tolerancia : float
        Critério de parada: ‖∇L(θ)‖ < tolerancia.
    lambda_ : float
        Coeficiente de regularização L2 (weight decay). Padrão: 0.01.

    Atributos pós-treinamento
    w                   : np.ndarray — parâmetros θ ótimos (com bias)
    historico_perda_    : list[float] — L(θ) por iteração
    historico_norma_grad_ : list[float] — ‖∇L(θ)‖ por iteração
    n_iteracoes_        : int — iterações realizadas até convergir
    """

    METODOS_OTIMIZACAO = ('gradiente_descendente', 'newton')
    METODOS_BUSCA      = ('secao_aurea', 'particao_igual', 'ajuste_quadratico')

    # Usa busca em linha

    def __init__(
        self,
        metodo_otimizacao: str = 'gradiente_descendente',
        metodo_busca: str      = 'secao_aurea',
        tmax: int              = 1000,
        tolerancia: float      = 1e-6,
        lambda_: float         = 0.01,
    ) -> None:
        
        if metodo_otimizacao not in self.METODOS_OTIMIZACAO:
            raise ValueError(
                f"metodo_otimizacao deve ser um de {self.METODOS_OTIMIZACAO}. "
                f"Recebido: '{metodo_otimizacao}'."
            )
        if metodo_busca not in self.METODOS_BUSCA:
            raise ValueError(
                f"metodo_busca deve ser um de {self.METODOS_BUSCA}. "
                f"Recebido: '{metodo_busca}'."
            )

        self.metodo_otimizacao = metodo_otimizacao
        self.metodo_busca      = metodo_busca
        self.tmax              = tmax
        self.tolerancia        = tolerancia
        self.lambda_           = lambda_

        self.w: np.ndarray = None  # parâmetros θ (com bias) — None indica modelo não treinado

        # variáveis para análise de convergência
        self.historico_perda_: list[float]       = []
        self.historico_norma_grad_: list[float]  = []
        self.n_iteracoes_: int                   = 0

    # Adaptado: delega para o otimizador selecionado, que usa busca em
    # linha para determinar o passo ótimo α* a cada iteração.
    
    def fit(self, _X, _y) -> RegressaoLogistica:
        """
        Estima os parâmetros θ minimizando L(θ) sobre os dados de treino.

        Adiciona coluna de bias, inicializa w = 0 e chama o otimizador
        configurado. Ao final, w contém o θ* que minimiza L(θ).

        Parâmetros
        _X : array-like, shape (n, p)   — features (sem bias)
        _y : array-like, shape (n,)     — rótulos binários {0, 1}

        Retorna
        self  (permite encadeamento: modelo.fit(X, y).predict(X_test))
        """
        
        # Converte para float para evitar matrizes dtype=object no Newton.
        X_base = np.asarray(_X, dtype=float)
        y = np.asarray(_y, dtype=float)

        #adiciona coluna do bias
        X = np.c_[np.ones((X_base.shape[0], 1), dtype=float), X_base]

        w0 = np.zeros(X.shape[1]) 

        if self.metodo_otimizacao == 'gradiente_descendente':
            w, hist_l, hist_g, n = self._gradiente_descendente(w0, X, y)
        else:
            w, hist_l, hist_g, n = self._metodo_newton(w0, X, y)

        self.w                     = w
        self.historico_perda_      = hist_l
        self.historico_norma_grad_ = hist_g
        self.n_iteracoes_          = n
        return self

    def predict_prob(self, _X):
        """
        Retorna P(y=1|x) = σ(xᵀw) para cada exemplo.
        """
        self._checar_treinado()
        X_base = np.asarray(_X, dtype=float)
        X = np.c_[np.ones((X_base.shape[0], 1), dtype=float), X_base]
        return self._sigmoide(X @ self.w)

    def predict(self, _X):
        """
        Classifica cada exemplo como adimplente (0) ou inadimplente (1).

        ADAPTAÇÃO: original usava np.sign(xᵀw) que retorna {-1, 0, +1},
        adequado quando y ∈ {-1, +1}. Como o problema usa y ∈ {0, 1},
        substituímos pelo limiar de 0.5 sobre a probabilidade.

        """
        self._checar_treinado()
        X_base = np.asarray(_X, dtype=float)
        X = np.c_[np.ones((X_base.shape[0], 1), dtype=float), X_base]
        prod_esc = X @ self.w
        return (self._sigmoide(prod_esc) >= 0.5).astype(float)
        # Para y ∈ {-1,+1}: return np.sign(prod_esc)   ← linha original

    def getW(self) :
        """Retorna o vetor de parâmetros w. Idêntico ao original."""
        self._checar_treinado()
        return self.w

    def getRegressionY(self, regressionX, shift=0) :
        """
        Retorna y da reta de decisão para um dado x₁ (uso em 2D).

        Reta de decisão: w₀ + w₁x₁ + w₂x₂ = 0  →  x₂ = (−w₀ − w₁x₁) / w₂

        Idêntico ao original. O parâmetro shift desloca a reta verticalmente,
        útil para visualizar margem.
        """
        self._checar_treinado()
        return (-self.w[0] + shift - self.w[1] * regressionX) / self.w[2]


    @staticmethod
    def _sigmoide(z: float | np.ndarray):
        """
        Função sigmoide: σ(z) = 1 / (1 + e^{−z})

        Transforma a combinação linear xᵀθ em probabilidade P(y=1|x) ∈ (0,1).

        Garantia de Estabilidade numérica

        Relação com a classe original

        """
        z = np.asarray(z, dtype=float)  # Garante que z é um array NumPy float
        z_clip = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    @staticmethod
    def _funcao_perda(w, X, y, lambda_ ):
        """
        Função de perda com regularização L2 (negativo da log-verossimilhança).

        L(θ) = −Σᵢ [yᵢ log σ(xᵢᵀθ) + (1−yᵢ) log(1−σ(xᵢᵀθ))] + (λ/2)‖θ₁:‖²

        O segundo termo é a regularização L2 — penaliza pesos grandes para
        evitar overfitting. O bias w[0] NÃO é regularizado (convenção padrão).

        Esta função é necessária para a busca em linha: g(α) = L(θ + α·d)
        precisa avaliar L em pontos arbitrários ao longo da direção d.
        """
        eps  = 1e-15
        prob = RegressaoLogistica._sigmoide(X @ w)
        prob = np.clip(prob, eps, 1.0 - eps)

        # Termo de verossimilhança
        perda_ll = -np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob))

        # Regularização L2 (exclui bias w[0])
        reg = (lambda_ / 2.0) * np.dot(w[1:], w[1:])

        return float(perda_ll + reg)

    @staticmethod
    def _gradiente(w, X, y, lambda_):
        """
        Gradiente de L(θ) com regularização L2.

        ∇L(θ) = (1/N) Xᵀ(σ(Xθ) − y)  +  λ·[0, θ₁, …, θₚ]


            num = X * y[:, np.newaxis]      # cada linha i: xᵢ·yᵢ
            den = 1 + exp(y * Xw)           
            grad = −(1/N) Σ num/den + λ·w[1:]
        """
        N    = X.shape[0]
        erro = RegressaoLogistica._sigmoide(X @ w) - y   # ŷ − y
        grad = (1.0 / N) * (X.T @ erro)

        grad[1:] += lambda_ * w[1:]
        return grad

    @staticmethod
    def _hessiana(w, X, lambda_):
        """
        Hessiana de L(θ) com regularização L2.

        H(θ) = (1/N) XᵀWX  +  λ·diag(0, 1, 1, …, 1)

        onde W = diag(σ(xᵢᵀθ) · (1 − σ(xᵢᵀθ))) é a matriz diagonal de pesos.

        Cada peso wᵢ = ŷᵢ(1−ŷᵢ) ∈ (0, 0.25] mede a "incerteza" do modelo
        no exemplo i: máximo em ŷ = 0.5 (mais incerto), zero quando
        ŷ → 0 ou ŷ → 1 (modelo confiante).

        A regularização L2 adiciona λ à diagonal (exceto bias), tornando H
        estritamente positiva definida, o que:
          1. Garante solução única do sistema H·d = −∇L.
          2. Melhora o condicionamento numérico.

       Necessário apenas para o Método de Newton.
        """
        N    = X.shape[0]
        prob = RegressaoLogistica._sigmoide(X @ w)
        pesos = prob * (1.0 - prob)                   # wᵢ = ŷᵢ(1−ŷᵢ)
        H = (1.0 / N) * (X.T @ (pesos[:, np.newaxis] * X))

        # Regularização L2 na diagonal (exclui bias → H[0,0] não muda)
        reg_diag       = lambda_ * np.ones(w.shape[0])
        reg_diag[0]    = 0.0
        H += np.diag(reg_diag)
        return H


    @staticmethod
    def _intervalo_inicial(g,passo = 1e-2,fator = 2.0,max_iter =200):
        """
        Encontra [a, b] que contém um mínimo de g(α), partindo de α=0.

        Por que é necessário?
        ---------------------
        Os métodos de busca (seção áurea, trisseção, parábola) precisam de
        um intervalo que *garanta* a presença do mínimo. Não dá para usar
        [0, ∞) diretamente.

        Como funciona?
        --------------
        Parte de α=0 (posição atual de θ) e avança na direção de descida
        de g, dobrando o passo a cada iteração até g começar a subir.
        Nesse ponto, o mínimo ficou entre o penúltimo e o próximo ponto.

        """
        a, ga = 0.0, g(0.0)
        b, gb = passo, g(passo)

        if ga < gb:                        # g cresce para direita: vai para esquerda
            a, b   = b, a
            ga, gb = gb, ga
            passo  = -passo

        for _ in range(max_iter):
            c = b + passo
            gc = g(c)
            if gb < gc:                    # g subiu: mínimo está entre a e c
                return (a, c) if a < c else (c, a)
            a, ga, b, gb = b, gb, c, gc
            passo *= fator

        return (a, b) if a < b else (b, a)

    @staticmethod
    def _busca_secao_aurea(g, a, b, n_iter = 50):
        """
        Minimiza g em [a, b] pela Seção Áurea. Retorna α* central.

        Usa φ = (√5−1)/2 ≈ 0.618 para posicionar c e d dentro de [a,b]:
            c = b − φ(b−a),   d = a + φ(b−a)

        A cada passo, descarta a metade sem o mínimo (g(c) vs g(d)).
        Vantagem: só 1 nova avaliação de g por iteração (o outro ponto é
        reaproveitado), reduzindo o intervalo por φ ≈ 0.618 por iteração.
        """
        phi    = (np.sqrt(5.0) - 1.0) / 2.0
        c, d   = b - phi*(b-a), a + phi*(b-a)
        gc, gd = g(c), g(d)

        for _ in range(n_iter - 2):
            if gc < gd:
                b, d, gd = d, c, gc
                c  = b - phi*(b-a); gc = g(c)
            else:
                a, c, gc = c, d, gd
                d  = a + phi*(b-a); gd = g(d)

        return (a + b) / 2.0

    @staticmethod
    def _busca_particao_igual(g, a, b, tol = 1e-6):
        """
        Minimiza g em [a, b] por trisseção (partição em 3 partes iguais).

        Divide [a,b] em u = a+(b−a)/3 e v = a+2(b−a)/3.
        Descarta o terço sem o mínimo. Reduz por 2/3 por iteração com
        2 avaliações de g — menos eficiente que seção áurea, mas simples.
        """
        while b - a >= tol:
            u = a + (b-a)/3.0
            v = a + 2*(b-a)/3.0
            if g(u) < g(v):
                b = v
            else:
                a = u
        return (a + b) / 2.0

    @staticmethod
    def _busca_ajuste_quadratico(g, a, b, n_iter = 50):
        """
        Minimiza g interpolando uma parábola por três pontos (a, mid, b).

        O vértice x* da parábola que passa por (a,g(a)), (b,g(b)), (c,g(c)):

            x* = ½ · [g(a)(b²−c²) + g(b)(c²−a²) + g(c)(a²−b²)]
                    / [g(a)(b−c)   + g(b)(c−a)   + g(c)(a−b)  ]

        Substitui o pior dos três pontos por x* a cada iteração.
        Converge mais rápido que seção áurea quando g é suave (quadrática),
        pois usa informação de curvatura implícita.
        """
        c        = (a + b) / 2.0
        ga, gb, gc = g(a), g(b), g(c)

        for _ in range(n_iter - 3):
            denom = ga*(b-c) + gb*(c-a) + gc*(a-b)
            if abs(denom) < 1e-14:
                break
            x  = 0.5*(ga*(b**2-c**2) + gb*(c**2-a**2) + gc*(a**2-b**2)) / denom
            gx = g(x)
            if x > b:
                (c, gc, b, gb) if gx > gb else (a, ga, b, gb).__class__ 
                if gx > gb: c, gc = x, gx
                else:        a, ga, b, gb = b, gb, x, gx
            elif x < b:
                if gx > gb: a, ga = x, gx
                else:        c, gc, b, gb = b, gb, x, gx

        return b   # b é sempre o melhor ponto

    def _passo_otimo(self, g):
        """
        Encontra o intervalo e aplica o método de busca configurado.

        Esta função concentra toda a lógica de escolha do método,
        mantendo os otimizadores limpos. Substitui diretamente o 'eta'
        da classe original por um α* calculado numericamente.

        Fluxo:
            1. _intervalo_inicial  →  [a, b] que contém o mínimo de g
            2. método de busca     →  α* ∈ [a, b]
        """
        a, b = self._intervalo_inicial(g)

        if self.metodo_busca == 'secao_aurea':
            return self._busca_secao_aurea(g, a, b)
        elif self.metodo_busca == 'particao_igual':
            return self._busca_particao_igual(g, a, b)
        else:   # 'ajuste_quadratico'
            return self._busca_ajuste_quadratico(g, a, b)


    def _gradiente_descendente(self, w0, X, y):
        """
        Gradiente Descendente com busca em linha exata.

        Equivalente ao loop original, com UMA mudança estrutural:
        em vez de  w ← w − eta·∇L,  fazemos:

            d   = −∇L(w)          direção de descida (igual ao original)
            g(α) = L(w + α·d)     problema 1D de busca em linha    [NOVO]
            α*  = argmin g(α)     passo ótimo via busca em linha   [NOVO]
            w   ← w + α*·d        atualização com passo ótimo

        O cálculo de ∇L (gradiente) é idêntico ao da classe original.

        Por que busca em linha?
        -----------------------
        O eta fixo exige ajuste manual e pode:
        - Ser muito pequeno  → convergência lenta (muitas iterações)
        - Ser muito grande   → divergência (w explode)
        Com α*, o passo é o melhor possível *naquela direção*,
        garantindo a maior redução de L em cada iteração.
        """
        w = w0.copy()
        hist_perda = []
        hist_grad = []

        for k in range(self.tmax):
            grad = self._gradiente(w, X, y, self.lambda_)
            norma = np.linalg.norm(grad)

            hist_perda.append(self._funcao_perda(w, X, y, self.lambda_))
            hist_grad.append(norma)

            # Critério de parada — versão numérica do "norma == 0" original
            if norma < self.tolerancia:
                return w, hist_perda, hist_grad, k

            # direção de descida: idêntica ao original
            direcao = -grad

            # -------------------------------------------------------
            # DIFERENÇA CENTRAL em relação ao original:
            # em vez de  w -= eta * grad,  buscamos o passo ótimo α*
            # -------------------------------------------------------
            g = lambda alpha: self._funcao_perda(
                w + alpha * direcao, X, y, self.lambda_
            )

            alpha = self._passo_otimo(g)
            w = w + alpha * direcao

        # Estado final se atingir tmax sem convergir
        grad_f = self._gradiente(w, X, y, self.lambda_)
        hist_perda.append(self._funcao_perda(w, X, y, self.lambda_))
        hist_grad.append(np.linalg.norm(grad_f))

        return w, hist_perda, hist_grad, self.tmax

    def _metodo_newton(self, w0, X, y):
        """
        Método de Newton com busca em linha exata.

        NOVO em relação à classe original — segundo otimizador disponível.

        Diferença fundamental em relação ao gradiente descendente:
        a direção d não é simplesmente −∇L, mas resolve o sistema linear:

            H(θ) · d = −∇L(θ)    →    d = −H(θ)⁻¹ ∇L(θ)

        onde H(θ) = (1/N) XᵀWX + λI é a Hessiana da função de perda.

        Por que isso é melhor?
        ----------------------
        A Hessiana H codifica a *curvatura* de L em θ. A direção de Newton
        d "escala" o gradiente pela curvatura local:
        - Em direções com alta curvatura (L muda rápido): d pequeno
        - Em direções com baixa curvatura (L muda devagar): d grande
        Isso elimina o "zigzag" do gradiente descendente e converge em
        muito menos iterações (convergência quadrática vs. linear).

        Regularização de Tikhonov (H += εI):
        Garante que H seja positiva definida mesmo quando mal condicionada,
        tornando np.linalg.solve sempre numericamente estável.
        """
        w = w0.copy()
        hist_perda = []
        hist_grad = []

        for k in range(self.tmax):
            grad = self._gradiente(w, X, y, self.lambda_)
            norma = np.linalg.norm(grad)

            hist_perda.append(self._funcao_perda(w, X, y, self.lambda_))
            hist_grad.append(norma)

            if norma < self.tolerancia:
                return w, hist_perda, hist_grad, k

            H = self._hessiana(w, X, self.lambda_)
            H_reg = H + 1e-8 * np.eye(H.shape[0])  # Tikhonov: estabilidade

            try:
                # d = −H⁻¹∇L  → resolvido como sistema linear (mais estável que inverter)
                direcao = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                # fallback: se H for singular, usamos gradiente descendente
                direcao = -grad

            g = lambda alpha: self._funcao_perda(
                w + alpha * direcao, X, y, self.lambda_
            )

            alpha = self._passo_otimo(g)
            w = w + alpha * direcao

        # Estado final se atingir tmax sem convergir
        grad_f = self._gradiente(w, X, y, self.lambda_)
        hist_perda.append(self._funcao_perda(w, X, y, self.lambda_))
        hist_grad.append(np.linalg.norm(grad_f))

        return w, hist_perda, hist_grad, self.tmax

    def _checar_treinado(self) -> None:
        if self.w is None:
            raise RuntimeError("Chame fit(X, y) antes de usar o modelo.")

    def __repr__(self) -> str:
        status = "treinado" if self.w is not None else "não treinado"
        return (
            f"RegressaoLogistica("
            f"metodo='{self.metodo_otimizacao}', "
            f"busca='{self.metodo_busca}', "
            f"tmax={self.tmax}, "
            f"λ={self.lambda_}, "
            f"status='{status}')"
        )


