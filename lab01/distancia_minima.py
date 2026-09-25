"""(i) Classificador de distância mínima (distância euclidiana).

Cada classe w_j é representada por um vetor protótipo m_j (vetor média) e o
padrão x é atribuído à classe do protótipo mais próximo.
"""
import numpy as np


def calcular_prototipos(dados, atributos, target, classes):
    """Vetor protótipo de cada classe: m_j = (1/N_j) * soma(x), x pertencente a w_j."""
    return {c: dados[dados[target] == c][atributos].mean().values for c in classes}


def dist_euclidiana(x, m):
    """D(x, m_j) = sqrt((x - m_j)^t (x - m_j))"""
    return np.sqrt(np.sum((x - m)**2))


def classificar(X, prototipos):
    """Atribui cada amostra de X à classe de menor distância ao protótipo."""
    pred = []
    for x in X:
        distancias = {c: dist_euclidiana(x, m) for c, m in prototipos.items()}
        pred.append(min(distancias, key=distancias.get))
    return pred
