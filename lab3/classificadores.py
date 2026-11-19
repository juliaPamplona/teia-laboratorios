# -*- coding: utf-8 -*-
r"""Classificadores das aulas anteriores, estendidos para as três classes

Todos recebem X (n × atributos) e y (rótulos inteiros 0, 1, 2) de treino e
devolvem uma função `prever(X)` que atribui uma das três classes a cada
amostra. Nenhuma biblioteca de aprendizagem de máquina é usada.

1. Distância mínima (lab01): protótipo m_j = média da classe; x vai para a
   classe de menor D(x, m_j) = ||x - m_j||.
2. Máximo da função de decisão (lab01): d_j(x) = x^T m_j - 1/2 m_j^T m_j;
   x vai para a classe de maior d_j(x). Como D² = x^T x - 2 d_j(x), as
   predições são idênticas às da distância mínima.
3. Superfície de decisão um contra todos: para cada classe i, a superfície
   do lab01 entre o protótipo m_i e o protótipo do "resto" m_r (média das
   amostras das outras duas classes):
       d_i(x) = (m_i - m_r)^T x - 1/2 (m_i - m_r)^T (m_i + m_r).
   d_i(x) > 0 indica "classe i" e d_i(x) < 0 indica "resto". Quando mais de
   uma (ou nenhuma) superfície responde positivo, x vai para a classe de
   maior d_i(x).
4. Perceptron simples um contra todos (lab2): um perceptron por classe, com
   resposta desejada +1 para a classe i e -1 para o resto; x vai para a
   classe de maior g_i(x) = w_i^T x.
5. Perceptron com regra delta um contra todos (lab3): igual ao anterior,
   treinando cada w_i com a regra delta de `regra_delta.py`.

Nos dois perceptrons os atributos são padronizados (z-score) com a média e
o desvio do treino, como no restante do lab3. Para comparar os g_i(x) entre
si, cada w_i é normalizado pela norma da parte sem o limiar (w_1, ..., w_n),
de modo que g_i(x) passa a ser a distância com sinal de x ao hiperplano i.
"""
import numpy as np

from regra_delta import aumentar, treinar_regra_delta


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def padronizador(X_treino):
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    desvio[desvio == 0] = 1
    return lambda X: (X - media) / desvio


def prototipos(X, y, classes):
    return np.array([X[y == c].mean(axis=0) for c in classes])


def _normalizar(W):
    """Divide cada w_i pela norma de (w_1, ..., w_n), sem o limiar w_0."""
    return W / np.linalg.norm(W[:, :-1], axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 1 e 2. Distância mínima e máximo da função de decisão
# ---------------------------------------------------------------------------

def distancia_minima(X_treino, y_treino, classes):
    M = prototipos(X_treino, y_treino, classes)

    def prever(X):
        distancias = np.sqrt(((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2))
        return np.asarray(classes)[np.argmin(distancias, axis=1)]

    return prever, {"prototipos": M}


def maximo_funcao_decisao(X_treino, y_treino, classes):
    M = prototipos(X_treino, y_treino, classes)

    def prever(X):
        # d_j(x) = x^T m_j - 1/2 m_j^T m_j
        d = X @ M.T - 0.5 * np.sum(M ** 2, axis=1)
        return np.asarray(classes)[np.argmax(d, axis=1)]

    return prever, {"prototipos": M}


# ---------------------------------------------------------------------------
# 3. Superfície de decisão um contra todos
# ---------------------------------------------------------------------------

def superficie_um_contra_todos(X_treino, y_treino, classes):
    W = []
    for c in classes:
        m_i = X_treino[y_treino == c].mean(axis=0)
        m_r = X_treino[y_treino != c].mean(axis=0)
        w = m_i - m_r
        w0 = -0.5 * np.dot(m_i - m_r, m_i + m_r)
        W.append(np.r_[w, w0])
    W = np.array(W)

    def prever(X):
        d = aumentar(X) @ W.T
        return np.asarray(classes)[np.argmax(d, axis=1)]

    return prever, {"W": W}


# ---------------------------------------------------------------------------
# 4. Perceptron simples um contra todos
# ---------------------------------------------------------------------------

def treinar_perceptron(X_amostras, respostas, taxa=1.0, max_epocas=1000):
    """Perceptron do slide 9 da PR41: corrige só quando r · w^T x <= 0.

    Se as classes não forem linearmente separáveis, o algoritmo não termina
    sozinho; ele para em `max_epocas` com os pesos da última correção.
    """
    X_aumentado = aumentar(X_amostras)
    w = np.zeros(X_aumentado.shape[1])

    for epoca in range(1, max_epocas + 1):
        houve_correcao = False
        for x, r in zip(X_aumentado, respostas):
            if r * (w @ x) <= 0:
                w = w + taxa * r * x
                houve_correcao = True
        if not houve_correcao:
            return w, epoca, True

    return w, max_epocas, False


def perceptron_um_contra_todos(X_treino, y_treino, classes, max_epocas=1000):
    transformar = padronizador(X_treino)
    Z = transformar(X_treino)

    W, info = [], []
    for c in classes:
        r = np.where(y_treino == c, 1, -1)
        w, epocas, convergiu = treinar_perceptron(Z, r, max_epocas=max_epocas)
        W.append(w)
        info.append({"classe": c, "epocas": epocas, "convergiu": convergiu})
    W = _normalizar(np.array(W))

    def prever(X):
        g = aumentar(transformar(X)) @ W.T
        return np.asarray(classes)[np.argmax(g, axis=1)]

    return prever, {"W": W, "treino": info}


# ---------------------------------------------------------------------------
# 5. Perceptron com regra delta um contra todos
# ---------------------------------------------------------------------------

def regra_delta_um_contra_todos(X_treino, y_treino, classes):
    transformar = padronizador(X_treino)
    Z = transformar(X_treino)

    W, info = [], []
    for c in classes:
        r = np.where(y_treino == c, 1, -1)
        resultado = treinar_regra_delta(Z, r)
        W.append(resultado["w"])
        info.append({
            "classe": c,
            "epocas": resultado["epocas"],
            "convergiu": resultado["eqm_estabilizou"],
            "eqm_final": resultado["eqm"][-1]
        })
    W = _normalizar(np.array(W))

    def prever(X):
        g = aumentar(transformar(X)) @ W.T
        return np.asarray(classes)[np.argmax(g, axis=1)]

    return prever, {"W": W, "treino": info}


CLASSIFICADORES = {
    "Distância mínima": distancia_minima,
    "Máximo função decisão": maximo_funcao_decisao,
    "Superfície 1×todos": superficie_um_contra_todos,
    "Perceptron 1×todos": perceptron_um_contra_todos,
    "Regra delta 1×todos": regra_delta_um_contra_todos
}
