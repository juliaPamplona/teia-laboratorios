# -*- coding: utf-8 -*-
r"""Leitura da base Iris e divisão 70% treinamento / 30% teste

Cada flor é representada pelo vetor de quatro atributos x = [x1, x2, x3, x4]:
- x1 = comprimento da sépala;
- x2 = largura da sépala;
- x3 = comprimento da pétala;
- x4 = largura da pétala.

A espécie é transformada em rótulos numéricos: 0 = Setosa, 1 = Versicolor e
2 = Virginica.
"""
import os

import numpy as np
import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_IRIS = os.path.join(PASTA, "Iris data.xls")

SEED = 42

ATRIBUTOS = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
]

NOME_PARA_CLASSE = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}

NOMES_CLASSES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

CLASSES = [0, 1, 2]


def carregar_iris(arquivo=ARQUIVO_IRIS):
    """Devolve X (150 × 4) e y (150,) com os rótulos 0, 1 e 2."""
    iris = pd.read_excel(arquivo, engine="xlrd")
    iris.columns = [coluna.strip() for coluna in iris.columns]

    y = (
        iris["Species"].str.strip().str.lower()
        .map(NOME_PARA_CLASSE)
        .to_numpy(dtype=int)
    )
    X = iris[ATRIBUTOS].to_numpy(dtype=float)
    return X, y


def divisao_treino_teste_estratificada(X, y, percentual_treino=0.70, seed=SEED):
    """Divisão estratificada feita à mão, sem train_test_split.

    70% de cada classe ficam no treinamento e 30% no teste. Não há conjunto
    de validação nesta atividade.
    """
    rng = np.random.default_rng(seed)

    indices_treino = []
    indices_teste = []

    for classe in np.unique(y):
        indices = np.where(y == classe)[0].copy()
        rng.shuffle(indices)

        n_treino = int(round(len(indices) * percentual_treino))

        indices_treino.extend(indices[:n_treino])
        indices_teste.extend(indices[n_treino:])

    indices_treino = np.array(indices_treino)
    indices_teste = np.array(indices_teste)

    # Embaralha para que as classes não fiquem em blocos
    rng.shuffle(indices_treino)
    rng.shuffle(indices_teste)

    return (
        X[indices_treino],
        X[indices_teste],
        y[indices_treino],
        y[indices_teste]
    )
