# -*- coding: utf-8 -*-
r"""Testes de normalidade multivariada: Henze-Zirkler e Mardia

Implementação própria (só NumPy e math), seguindo as fórmulas do pacote MVN
do R (Korkmaz, Goksuluk e Zararsiz, 2014), para conferir em Python o
resultado do script normalidade.R.

Em ambos os testes, H0: os dados vêm de uma população normal multivariada.
Se p > 0,05, não se rejeita H0.

Notação: n amostras, p atributos, m = média amostral e S = covariância de
máxima verossimilhança (divisor n). D_i = (x_i - m)^T S^{-1} (x_i - m) é a
distância de Mahalanobis ao quadrado de cada amostra à média, e
D_ik = (x_i - x_k)^T S^{-1} (x_i - x_k), a distância entre duas amostras.

1. Henze-Zirkler (1990)
   β  = (1/√2) ((2p + 1)/4)^{1/(p+4)} n^{1/(p+4)}
   HZ = n [ (1/n²) Σ_i Σ_k e^{-β² D_ik / 2}
            - 2 (1 + β²)^{-p/2} (1/n) Σ_i e^{-β² D_i / (2(1 + β²))}
            + (1 + 2β²)^{-p/2} ]
   Sob H0, HZ segue aproximadamente uma log-normal, cujos parâmetros vêm da
   média μ e da variância σ² de HZ; p = 1 - F_lognormal(HZ).

2. Mardia (1970)
   b1 = (1/n²) Σ_i Σ_k [(x_i - m)^T S^{-1} (x_k - m)]³   (assimetria)
   b2 = (1/n) Σ_i D_i²                                  (curtose)
   n b1 / 6 ~ χ² com p(p+1)(p+2)/6 graus de liberdade;
   (b2 - p(p+2)) / √(8p(p+2)/n) ~ N(0, 1).
   Para n < 20 usa-se a correção de pequenas amostras na assimetria.
   Os dados são considerados normais quando os dois p-valores são > 0,05.
"""
import math

import numpy as np


# ---------------------------------------------------------------------------
# Distribuições (sem scipy)
# ---------------------------------------------------------------------------

def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def qui_quadrado_cdf(x, gl):
    """P(χ²_gl <= x) pela série da função gama incompleta regularizada."""
    if x <= 0:
        return 0.0
    a, t = gl / 2, x / 2
    termo = 1.0 / a
    soma = termo
    for n in range(1, 10000):
        termo *= t / (a + n)
        soma += termo
        if termo < soma * 1e-15:
            break
    return min(1.0, math.exp(-t + a * math.log(t) - math.lgamma(a)) * soma)


def qui_quadrado_quantil(prob, gl):
    """Inversa da χ² por bisseção (usada no gráfico Q-Q)."""
    baixo, alto = 0.0, max(1.0, gl * 10)
    while qui_quadrado_cdf(alto, gl) < prob:
        alto *= 2
    for _ in range(200):
        meio = (baixo + alto) / 2
        if qui_quadrado_cdf(meio, gl) < prob:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


# ---------------------------------------------------------------------------
# Distâncias de Mahalanobis
# ---------------------------------------------------------------------------

def _centrar(X):
    X = np.asarray(X, dtype=float)
    n = len(X)
    centrado = X - X.mean(axis=0)
    S = centrado.T @ centrado / n
    G = centrado @ np.linalg.solve(S, centrado.T)  # G_ik = (x_i - m)^T S^-1 (x_k - m)
    return n, X.shape[1], G


def distancias_mahalanobis(X):
    """D_i de cada amostra à média (diagonal de G)."""
    _, _, G = _centrar(X)
    return np.diag(G).copy()


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def henze_zirkler(X, alpha=0.05):
    n, p, G = _centrar(X)
    Di = np.diag(G)
    Dik = Di[:, None] + Di[None, :] - 2 * G

    beta = (1 / math.sqrt(2)) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * n ** (1 / (p + 4))
    b2 = beta ** 2

    hz = n * (
        np.sum(np.exp(-b2 / 2 * Dik)) / n ** 2
        - 2 * (1 + b2) ** (-p / 2) * np.sum(np.exp(-b2 / (2 * (1 + b2)) * Di)) / n
        + (1 + 2 * b2) ** (-p / 2)
    )

    a = 1 + 2 * b2
    wb = (1 + b2) * (1 + 3 * b2)
    mu = 1 - a ** (-p / 2) * (1 + p * b2 / a + p * (p + 2) * b2 ** 2 / (2 * a ** 2))
    si2 = (
        2 * (1 + 4 * b2) ** (-p / 2)
        + 2 * a ** (-p) * (1 + 2 * p * b2 ** 2 / a ** 2 + 3 * p * (p + 2) * b2 ** 4 / (4 * a ** 4))
        - 4 * wb ** (-p / 2) * (1 + 3 * p * b2 ** 2 / (2 * wb) + p * (p + 2) * b2 ** 4 / (2 * wb ** 2))
    )
    pmu = math.log(math.sqrt(mu ** 4 / (si2 + mu ** 2)))
    psi = math.sqrt(math.log((si2 + mu ** 2) / mu ** 2))

    p_valor = 1 - normal_cdf((math.log(hz) - pmu) / psi)
    return {
        "teste": "Henze-Zirkler",
        "estatistica": hz,
        "p_valor": p_valor,
        "normal": p_valor > alpha
    }


def mardia(X, alpha=0.05):
    n, p, G = _centrar(X)
    b1 = np.sum(G ** 3) / n ** 2
    b2 = np.sum(np.diag(G) ** 2) / n

    gl = p * (p + 1) * (p + 2) / 6
    if n < 20:
        k = (p + 1) * (n + 1) * (n + 3) / (n * ((n + 1) * (p + 1) - 6))
        estat_assimetria = n * k * b1 / 6
    else:
        estat_assimetria = n * b1 / 6
    p_assimetria = 1 - qui_quadrado_cdf(estat_assimetria, gl)

    estat_curtose = (b2 - p * (p + 2)) / math.sqrt(8 * p * (p + 2) / n)
    p_curtose = 2 * (1 - normal_cdf(abs(estat_curtose)))

    return {
        "teste": "Mardia",
        "assimetria": estat_assimetria,
        "p_assimetria": p_assimetria,
        "curtose": estat_curtose,
        "p_curtose": p_curtose,
        "normal": p_assimetria > alpha and p_curtose > alpha
    }


if __name__ == "__main__":
    # Autoteste: amostras de uma normal multivariada costumam passar nos
    # testes, e amostras exponenciais (assimétricas) costumam ser rejeitadas.
    rng = np.random.default_rng(0)
    normal = rng.multivariate_normal(np.zeros(4), np.eye(4) + 0.5, size=200)
    exponencial = rng.exponential(size=(200, 4))
    print("Normal      HZ p =", round(henze_zirkler(normal)["p_valor"], 4),
          "| Mardia p =", round(mardia(normal)["p_assimetria"], 4), round(mardia(normal)["p_curtose"], 4))
    print("Exponencial HZ p =", round(henze_zirkler(exponencial)["p_valor"], 6),
          "| Mardia p =", round(mardia(exponencial)["p_assimetria"], 6))
    assert abs(qui_quadrado_cdf(qui_quadrado_quantil(0.95, 4), 4) - 0.95) < 1e-9
    assert abs(qui_quadrado_cdf(9.487729, 4) - 0.95) < 1e-6
