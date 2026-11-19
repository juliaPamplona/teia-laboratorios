# -*- coding: utf-8 -*-
r"""Classificador de Bayes ótimo (gaussiano) e Naive Bayes gaussiano

Classificação pelo máximo a posteriori (MAP):

    C* = argmax_j P(C_j | x) = argmax_j P(x | C_j) P(C_j).

Como as probabilidades a priori são iguais, P(C_1) = P(C_2) = P(C_3), elas
são suprimidas, e a decisão passa a depender só da verossimilhança P(x | C_j).

1. Bayes ótimo: P(x | C_j) é a normal multivariada N(m_j, Σ_j), com vetor de
   médias m_j e matriz de covariância completa Σ_j de cada classe. Tomando o
   logaritmo e descartando o termo -(n/2) ln 2π, comum a todas as classes,
   a função discriminante é

       g_j(x) = -1/2 ln|Σ_j| - 1/2 (x - m_j)^T Σ_j^{-1} (x - m_j).

2. Naive Bayes: supõe os atributos condicionalmente independentes dentro de
   cada classe, P(x | C_j) = Π_k P(x_k | C_j), com cada P(x_k | C_j) normal
   de média m_jk e variância s²_jk. É o caso anterior com Σ_j diagonal:

       g_j(x) = -1/2 Σ_k [ ln s²_jk + (x_k - m_jk)² / s²_jk ].

Os parâmetros são as estimativas de máxima verossimilhança: média amostral e
covariância (ou variância) com divisor N, e não N - 1.

Superfície de decisão entre as classes i e j: d_ij(x) = g_i(x) - g_j(x) = 0.
Expandindo g_j, com P_j = Σ_j^{-1},

    d_ij(x) = x^T A x + b^T x + c,
    A = -1/2 (P_i - P_j),
    b = P_i m_i - P_j m_j,
    c = -1/2 (m_i^T P_i m_i - m_j^T P_j m_j) - 1/2 (ln|Σ_i| - ln|Σ_j|).

É uma superfície quadrática (hiperquádrica em 4 dimensões). d_ij(x) > 0
indica a classe i e d_ij(x) < 0, a classe j.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Bayes ótimo (covariância completa)
# ---------------------------------------------------------------------------

def treinar_bayes_gaussiano(X_treino, y_treino):
    parametros = {}

    for classe in np.unique(y_treino):
        X_classe = X_treino[y_treino == classe]

        media = np.mean(X_classe, axis=0)
        diferencas = X_classe - media
        covariancia = (diferencas.T @ diferencas) / len(X_classe)

        sinal, logdet = np.linalg.slogdet(covariancia)

        # Correção só se a matriz não for positiva definida: Σ' = Σ + 1e-8 I
        if sinal <= 0:
            covariancia = covariancia + 1e-8 * np.eye(X_treino.shape[1])
            sinal, logdet = np.linalg.slogdet(covariancia)

        if sinal <= 0:
            raise ValueError(f"Matriz de covariância inválida para a classe {classe}")

        parametros[int(classe)] = {
            "media": media,
            "covariancia": covariancia,
            "logdet": logdet
        }

    return parametros


def pontuar_bayes_gaussiano(X_amostras, parametros):
    """g_j(x) = -1/2 ln|Σ_j| - 1/2 (x - m_j)^T Σ_j^{-1} (x - m_j)."""
    X_amostras = np.atleast_2d(np.asarray(X_amostras, dtype=float))
    diferencas = X_amostras - parametros["media"]

    # Distância de Mahalanobis ao quadrado, sem inverter Σ explicitamente
    resolvido = np.linalg.solve(parametros["covariancia"], diferencas.T).T
    mahalanobis = np.sum(diferencas * resolvido, axis=1)

    return -0.5 * (parametros["logdet"] + mahalanobis)


# ---------------------------------------------------------------------------
# Naive Bayes (covariância diagonal)
# ---------------------------------------------------------------------------

def treinar_naive_bayes_gaussiano(X_treino, y_treino):
    parametros = {}

    for classe in np.unique(y_treino):
        X_classe = X_treino[y_treino == classe]

        media = np.mean(X_classe, axis=0)
        variancias = np.var(X_classe, axis=0, ddof=0)
        variancias = np.maximum(variancias, 1e-9)  # evita divisão por zero

        parametros[int(classe)] = {
            "media": media,
            "variancias": variancias
        }

    return parametros


def pontuar_naive_bayes_gaussiano(X_amostras, parametros):
    """Soma dos logaritmos das densidades normais univariadas."""
    X_amostras = np.atleast_2d(np.asarray(X_amostras, dtype=float))
    diferencas = X_amostras - parametros["media"]

    return -0.5 * np.sum(
        np.log(parametros["variancias"]) + diferencas ** 2 / parametros["variancias"],
        axis=1
    )


# ---------------------------------------------------------------------------
# Classificação MAP
# ---------------------------------------------------------------------------

def prever_modelo(X_amostras, parametros, funcao_pontuacao, classes=None):
    """Atribui cada amostra à classe de maior g_j(x).

    `classes` restringe a decisão a um subconjunto (por exemplo, a um par).
    """
    if classes is None:
        classes = list(parametros.keys())

    pontuacoes = np.vstack([
        funcao_pontuacao(X_amostras, parametros[classe])
        for classe in classes
    ])
    return np.asarray(classes)[np.argmax(pontuacoes, axis=0)]


MODELOS = {
    "Bayes ótimo": {
        "treinar": treinar_bayes_gaussiano,
        "pontuar": pontuar_bayes_gaussiano
    },
    "Naive Bayes": {
        "treinar": treinar_naive_bayes_gaussiano,
        "pontuar": pontuar_naive_bayes_gaussiano
    }
}


# ---------------------------------------------------------------------------
# Superfície de decisão d_ij(x) = x^T A x + b^T x + c
# ---------------------------------------------------------------------------

def _media_covariancia(parametros_classe):
    """Média e Σ de uma classe (Σ diagonal no Naive Bayes)."""
    if "covariancia" in parametros_classe:
        return parametros_classe["media"], parametros_classe["covariancia"]
    return parametros_classe["media"], np.diag(parametros_classe["variancias"])


def superficie_decisao(parametros, classe_i, classe_j):
    """Coeficientes (A, b, c) de d_ij(x) = g_i(x) - g_j(x)."""
    m_i, S_i = _media_covariancia(parametros[classe_i])
    m_j, S_j = _media_covariancia(parametros[classe_j])

    P_i = np.linalg.inv(S_i)
    P_j = np.linalg.inv(S_j)

    A = -0.5 * (P_i - P_j)
    b = P_i @ m_i - P_j @ m_j
    c = (
        -0.5 * (m_i @ P_i @ m_i - m_j @ P_j @ m_j)
        - 0.5 * (np.linalg.slogdet(S_i)[1] - np.linalg.slogdet(S_j)[1])
    )
    return A, b, c


def avaliar_superficie(X, A, b, c):
    X = np.atleast_2d(X)
    return np.einsum("nk,kl,nl->n", X, A, X) + X @ b + c


def fatia_superficie(A, b, c, indices_visuais, valores_fixos):
    """Restringe d_ij(x) a dois atributos, com os demais fixos.

    Com x = (u, f), em que u são os atributos visualizados e f os fixos:
        d(u) = u^T A_uu u + (b_u + 2 A_uf f)^T u + (c + b_f^T f + f^T A_ff f),
    uma cônica no plano dos atributos visualizados.
    """
    u = list(indices_visuais)
    f = [k for k in range(len(b)) if k not in u]
    valores_fixos = np.asarray(valores_fixos, dtype=float)

    A_2d = A[np.ix_(u, u)]
    b_2d = b[u] + 2 * A[np.ix_(u, f)] @ valores_fixos
    c_2d = c + b[f] @ valores_fixos + valores_fixos @ A[np.ix_(f, f)] @ valores_fixos
    return A_2d, b_2d, c_2d


def formatar_equacao(A, b, c, nomes=None, casas=4):
    """Escreve d(x) = Σ a_kl x_k x_l + Σ b_k x_k + c = 0 por extenso."""
    n = len(b)
    if nomes is None:
        nomes = [f"x{k + 1}" for k in range(n)]

    termos = []
    for k in range(n):
        termos.append((A[k, k], f"{nomes[k]}²"))
    for k in range(n):
        for l in range(k + 1, n):
            termos.append((2 * A[k, l], f"{nomes[k]}·{nomes[l]}"))
    for k in range(n):
        termos.append((b[k], nomes[k]))
    termos.append((c, ""))

    texto = ""
    for valor, variavel in termos:
        if abs(valor) < 10 ** (-casas):
            continue
        sinal = "-" if valor < 0 else "+"
        parcela = f"{abs(valor):.{casas}f}" + (f" {variavel}" if variavel else "")
        texto += f" {sinal} {parcela}" if texto else ("-" if valor < 0 else "") + parcela
    return texto + " = 0"
