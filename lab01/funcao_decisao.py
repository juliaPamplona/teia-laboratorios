"""(ii) Classificador pelo máximo das funções de decisão.

d_j(x) = x^t m_j - 1/2 m_j^t m_j

Obs.: menor D(x, m_j) equivale a maior d_j(x), pois D^2 = x^t x - 2 d_j(x),
e x^t x é igual para todas as classes. Logo, este classificador deve produzir
exatamente as mesmas predições que o de distância mínima.
"""
import numpy as np


def func_decisao(x, m):
    return np.dot(x, m) - 0.5 * np.dot(m, m)


def classificar(X, prototipos):
    """Atribui cada amostra de X à classe com maior valor numérico de d_j(x)."""
    pred = []
    for x in X:
        decisoes = {c: func_decisao(x, m) for c, m in prototipos.items()}
        pred.append(max(decisoes, key=decisoes.get))
    return pred


def formatar(m):
    """Expressão de d_j(x) com os coeficientes numéricos do protótipo m."""
    termos = " + ".join(f"{m[k]:.3f}*x{k+1}" for k in range(len(m)))
    return f"{termos} - {0.5 * np.dot(m, m):.3f}"
