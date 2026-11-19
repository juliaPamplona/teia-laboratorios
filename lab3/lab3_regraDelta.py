# -*- coding: utf-8 -*-
"""Laboratório 03 — Perceptron com regra delta (aula PR41)

Como executar (a partir da raiz do repositório):

    pip install numpy pandas matplotlib xlrd
    python lab3/lab3_regraDelta.py

A base "Iris data.xls" é lida da pasta deste arquivo. As figuras são salvas
em figuras/, na mesma pasta, e exibidas ao final.

Atividade:
- implementar, sem bibliotecas de aprendizagem de máquina, o perceptron com
  regra delta visto na aula PR41;
- base Iris com 70% dos dados para treinamento e 30% para teste;
- pares Setosa × Virginica, Setosa × Versicolor e Virginica × Versicolor;
- tabelas verdade OR, AND e XOR (o algoritmo converge?);
- gráfico de convergência ÉPOCA × EQM (MSE) para todos os casos.

Organização do código:
- regra_delta.py: o algoritmo (regra delta, EQM, classificação, teste de
  separabilidade e critério de convergência);
- visualizacao.py: impressões no terminal e gráficos;
- lab3_regraDelta.py (este arquivo): leitura e preparo dos dados, execução
  dos experimentos e conclusões.

Complemento (aula PR51, métricas de qualidade):
- metricas.py, classificadores.py e lab3_metricas.py: comparação dos cinco
  classificadores nas três classes da Iris;
- exercicios_pr51.py: exercícios resolvidos da aula.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regra_delta import (
    aumentar,
    calcular_eqm,
    classificar,
    treinar_regra_delta,
    verificar_separabilidade
)
from visualizacao import (
    imprimir_resultados_iris,
    imprimir_tabela_verdade,
    imprimir_resumo,
    plotar_eqm_por_experimento,
    plotar_eqm_comparacao,
    plotar_dispersao_2d,
    plotar_fronteiras_tabelas
)

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", None)

SEED = 42
PASTA = os.path.dirname(os.path.abspath(__file__))

"""## 1. Leitura da base Iris

Cada flor é representada pelo vetor de quatro atributos x = [x_1, x_2, x_3, x_4]:
- `x1` = comprimento da sépala;
- `x2` = largura da sépala;
- `x3` = comprimento da pétala;
- `x4` = largura da pétala.
"""

iris = pd.read_excel(os.path.join(PASTA, "Iris data.xls"), engine="xlrd")
iris.columns = [coluna.strip() for coluna in iris.columns]

print(iris.shape)
print(iris.columns.tolist())
print(iris.head())

nome_para_classe = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}

nomes_classes = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

atributos = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
]

y = iris["Species"].str.strip().str.lower().map(nome_para_classe).to_numpy(dtype=int)
X = iris[atributos].to_numpy(dtype=float)

print("\nDimensão de X:", X.shape)
print("Dimensão de y:", y.shape)

r"""## 2. Separação 70% treinamento / 30% teste

A divisão é feita manualmente, sem `train_test_split`, e é estratificada:
70% de cada classe ficam no treinamento e 30% no teste. A mesma divisão é
usada por todos os pares, então cada flor de teste nunca é vista no treino.
"""


def divisao_treino_teste_estratificada(y, percentual_treino=0.70, seed=SEED):
    rng = np.random.default_rng(seed)

    indices_treino = []
    indices_teste = []

    for classe in np.unique(y):
        indices = rng.permutation(np.where(y == classe)[0])
        n_treino = int(round(len(indices) * percentual_treino))

        indices_treino.extend(indices[:n_treino])
        indices_teste.extend(indices[n_treino:])

    # Embaralha para que as classes não fiquem em blocos durante o treino
    return rng.permutation(indices_treino), rng.permutation(indices_teste)


indices_treino, indices_teste = divisao_treino_teste_estratificada(y)

X_treino, X_teste = X[indices_treino], X[indices_teste]
y_treino, y_teste = y[indices_treino], y[indices_teste]

print("Treinamento:", len(X_treino), "amostras")
print("Teste:", len(X_teste), "amostras")

for c in np.unique(y):
    print(
        f"{nomes_classes[c]:<11} treino: {np.sum(y_treino == c)}"
        f" | teste: {np.sum(y_teste == c)}"
    )

r"""## 3. Experimentos com a Iris usando os 4 atributos

Cada par de classes recebe um perceptron binário próprio, treinado somente
com as amostras de treinamento das duas classes. A primeira classe do par
recebe a resposta desejada +1 e a segunda, -1.

Os atributos são padronizados (z-score) com a média e o desvio padrão do
treinamento. A aula não exige essa etapa, mas ela deixa os atributos na
mesma escala, o que permite usar o mesmo α nos três pares sem oscilação.
"""

pares_classes = [
    (0, 2),  # Setosa x Virginica
    (0, 1),  # Setosa x Versicolor
    (2, 1)   # Virginica x Versicolor
]


def nome_par(classe_a, classe_b):
    return f"{nomes_classes[classe_a]} × {nomes_classes[classe_b]}"


def padronizar(X_tr, X_te):
    media = X_tr.mean(axis=0)
    desvio = X_tr.std(axis=0)
    desvio[desvio == 0] = 1
    return (X_tr - media) / desvio, (X_te - media) / desvio, media, desvio


def rodar_par(classe_a, classe_b, colunas):
    mascara_treino = (y_treino == classe_a) | (y_treino == classe_b)
    mascara_teste = (y_teste == classe_a) | (y_teste == classe_b)

    Xtr = X_treino[mascara_treino][:, colunas]
    Xte = X_teste[mascara_teste][:, colunas]
    rtr = np.where(y_treino[mascara_treino] == classe_a, 1, -1)
    rte = np.where(y_teste[mascara_teste] == classe_a, 1, -1)

    Ztr, Zte, media, desvio = padronizar(Xtr, Xte)

    resultado = treinar_regra_delta(Ztr, rtr)
    w = resultado["w"]

    # Pesos nas unidades originais dos atributos:
    # g(x) = Σ w_j (x_j - m_j) / s_j + w_0
    w_original = np.r_[
        w[:-1] / desvio,
        w[-1] - np.sum(w[:-1] * media / desvio)
    ]

    pred_teste = classificar(aumentar(Zte), w)

    resultado.update({
        "nome": nome_par(classe_a, classe_b),
        "classe_a": classe_a,
        "classe_b": classe_b,
        "w_original": w_original,
        "X_treino": Xtr,
        "X_teste": Xte,
        "r_treino": rtr,
        "r_teste": rte,
        "erros_teste": int(np.sum(pred_teste != rte)),
        "acuracia_teste": np.mean(pred_teste == rte),
        "separavel": verificar_separabilidade(Ztr, rtr)
    })
    return resultado


resultados_4d = [rodar_par(a, b, [0, 1, 2, 3]) for a, b in pares_classes]
imprimir_resultados_iris(resultados_4d, "4 atributos")

r"""### 3.1 Gráficos ÉPOCA × EQM da Iris

Um gráfico por par e um gráfico com os três pares juntos, para comparar com a
Fig. 12.26 do slide 18. A legenda da própria figura identifica (a) Setosa ×
Versicolor como linearmente separáveis e (b) Versicolor × Virginica como não
separáveis. O texto do slide cita "Setosa e Virginica" em (b), mas isso é um
erro de digitação: Setosa é separável das outras duas espécies.
"""

plotar_eqm_por_experimento(
    resultados_4d,
    "Convergência da regra delta na Iris (4 atributos)",
    "mse_iris_pares"
)
plotar_eqm_comparacao(
    resultados_4d,
    "Convergência da regra delta — comparação dos pares (4 atributos)",
    "mse_iris_comparacao"
)

r"""## 4. Visualização 2D com os dois primeiros atributos

Com 4 atributos a superfície de decisão é um hiperplano em 4 dimensões e não
pode ser desenhada como uma reta. Para a visualização, o perceptron é treinado
novamente usando apenas os dois primeiros atributos (comprimento e largura da
sépala).

No espaço da sépala, Versicolor e Virginica se sobrepõem bastante, então esse
par fica bem pior do que com os 4 atributos. A comparação mostra como a escolha
dos atributos afeta a separabilidade.
"""

resultados_2d = [rodar_par(a, b, [0, 1]) for a, b in pares_classes]
imprimir_resultados_iris(resultados_2d, "2 primeiros atributos")

plotar_dispersao_2d(
    resultados_2d,
    nomes_classes,
    atributos[:2],
    "dispersao_iris_2d"
)
plotar_eqm_por_experimento(
    resultados_2d,
    "Convergência da regra delta na Iris (2 primeiros atributos)",
    "mse_iris_2d"
)

r"""## 5. Tabelas verdade OR, AND e XOR

Entradas em {0, 1}. A saída 1 da tabela vira a resposta desejada +1 e a
saída 0 vira -1, como no perceptron da aula. Aqui não há padronização nem
divisão treino/teste: os quatro padrões são o conjunto de treinamento.

Para conferir a regra delta, o EQM final é comparado com o menor EQM possível
para uma função linear, obtido pela solução de mínimos quadrados
(`np.linalg.lstsq`, álgebra linear, não aprendizagem de máquina). Com α
constante e atualização a cada padrão, os pesos ficam oscilando perto desse
mínimo, por isso o EQM final fica um pouco acima dele.
"""

X_tabela = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
], dtype=float)

tabelas_verdade = {
    "OR":  np.array([1, 1, 1, 0]),
    "AND": np.array([1, 0, 0, 0]),
    "XOR": np.array([0, 1, 1, 0])
}


def rodar_tabela_verdade(nome, saidas):
    respostas = np.where(saidas == 1, 1, -1)

    resultado = treinar_regra_delta(X_tabela, respostas)

    X_aumentado = aumentar(X_tabela)
    pred = classificar(X_aumentado, resultado["w"])

    w_otimo = np.linalg.lstsq(X_aumentado, respostas, rcond=None)[0]

    resultado.update({
        "nome": nome,
        "respostas": respostas,
        "eqm_minimo": calcular_eqm(X_aumentado, respostas, w_otimo),
        "acertos": int(np.sum(pred == respostas)),
        "separavel": verificar_separabilidade(X_tabela, respostas)
    })
    return resultado


resultados_tabelas = [
    rodar_tabela_verdade(nome, saidas)
    for nome, saidas in tabelas_verdade.items()
]

for resultado in resultados_tabelas:
    imprimir_tabela_verdade(resultado, X_tabela)

r"""### 5.1 Gráficos ÉPOCA × EQM e fronteiras de decisão das tabelas verdade"""

plotar_eqm_por_experimento(
    resultados_tabelas,
    "Convergência da regra delta nas tabelas verdade",
    "mse_tabelas_verdade"
)
plotar_fronteiras_tabelas(
    resultados_tabelas,
    X_tabela,
    "fronteiras_tabelas_verdade"
)

r"""## 6. Resumo — o algoritmo converge?"""

imprimir_resumo(
    [("4 atrib.", resultados_4d), ("2 atrib.", resultados_2d)],
    resultados_tabelas
)

r"""## 7. Conclusões

- **Setosa × Virginica e Setosa × Versicolor (4 atributos):** a Setosa é
  linearmente separável das outras duas espécies. O EQM estabiliza e o treino
  chega a zero erros, então o algoritmo converge.
- **Os mesmos pares com os 2 atributos da sépala:** as classes continuam
  linearmente separáveis no treino (o perceptron simples encontra uma reta sem
  erros), mas a solução de EQM mínimo da regra delta deixa alguns padrões do
  lado errado. Isso mostra que a regra delta minimiza o erro quadrático, não o
  número de erros, e que a solução dela nem sempre coincide com uma reta
  separadora.
- **Virginica × Versicolor:** o EQM converge para um mínimo, mas sempre sobram
  erros de classificação no treino. As classes não são linearmente separáveis,
  então o algoritmo não converge para uma separação perfeita. É o caso da
  Fig. 12.26(b): o EQM estabiliza num patamar mais alto. Com apenas os 2
  atributos da sépala a sobreposição é muito maior e o desempenho cai bastante.
- **OR e AND:** são linearmente separáveis; a regra delta encontra uma reta
  que classifica os quatro padrões corretamente e o algoritmo converge.
- **XOR:** o EQM estabiliza, mas no maior patamar entre as tabelas. A melhor
  solução linear é w = 0 (EQM = 0,5), em que g(x) = 0 para todas as entradas
  e a classificação fica indefinida. Como os pesos iniciais já são w = 0, a
  curva do XOR sobe um pouco (de 0,500 para cerca de 0,516): a correção feita
  a cada padrão, com α constante, afasta os pesos levemente do mínimo e eles
  passam a oscilar em torno dele. Nenhuma reta separa as classes, então o
  algoritmo não converge para uma solução. Para resolver o XOR são necessárias
  duas retas, isto é, uma camada intermediária de perceptrons (slide 19,
  Fig. 12.27(b) e Fig. 12.28).
- **Regra delta × perceptron simples:** o perceptron simples só termina quando
  todos os padrões são classificados corretamente; com classes não separáveis
  ele corrige os pesos para sempre. A regra delta minimiza o erro quadrático,
  então o EQM sempre estabiliza e ela devolve a melhor fronteira linear no
  sentido do erro quadrático. Ela não separa classes não separáveis, mas
  termina com um classificador útil mesmo nesse caso.
"""

# Exibe todas as figuras geradas
plt.show()
