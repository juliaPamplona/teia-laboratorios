# -*- coding: utf-8 -*-
r"""Laboratório 03 (complemento) — Métricas de qualidade (aula PR51)

Como executar (a partir da raiz do repositório):

    pip install numpy pandas matplotlib xlrd
    python lab3/lab3_metricas.py

Atividade:
1) Na base Iris, classificar as três classes com:
   - distância mínima;
   - máximo da função de decisão;
   - superfície de decisão (um contra todos);
   - perceptron (um contra todos);
   - perceptron com regra delta (um contra todos);
   e calcular, para cada um: acerto global, acurácia do produtor, acurácia
   do usuário, Kappa, Tau e, no problema de duas classes, MCC e F_b com
   b = 1 e b = 2.
2) Teste de significância de Kappa entre a superfície de decisão e o
   perceptron, e de Kappa e Tau entre a regra delta e o perceptron.

Organização:
- metricas.py: matriz de confusão e todas as métricas da PR51;
- classificadores.py: os cinco classificadores para as três classes;
- lab3_metricas.py (este arquivo): dados, execução e conclusões;
- exercicios_pr51.py: exercícios resolvidos da aula.

As métricas são calculadas no conjunto de teste (30%, 45 amostras), com a
mesma divisão estratificada (semente 42) do lab3_regraDelta.py.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import metricas as M
from classificadores import CLASSIFICADORES

np.set_printoptions(precision=4, suppress=True)

SEED = 42
PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_FIGURAS = os.path.join(PASTA, "figuras")

CLASSES = [0, 1, 2]
NOMES = ["Setosa", "Versicolor", "Virginica"]

"""## 1. Dados e divisão 70% / 30%"""

iris = pd.read_excel(os.path.join(PASTA, "Iris data.xls"), engine="xlrd")
iris.columns = [coluna.strip() for coluna in iris.columns]

nome_para_classe = {"setosa": 0, "versicolor": 1, "virginica": 2}
atributos = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

y = iris["Species"].str.strip().str.lower().map(nome_para_classe).to_numpy(dtype=int)
X = iris[atributos].to_numpy(dtype=float)


def divisao_treino_teste_estratificada(y, percentual_treino=0.70, seed=SEED):
    """Mesma divisão usada em lab3_regraDelta.py."""
    rng = np.random.default_rng(seed)
    indices_treino, indices_teste = [], []
    for classe in np.unique(y):
        indices = rng.permutation(np.where(y == classe)[0])
        n_treino = int(round(len(indices) * percentual_treino))
        indices_treino.extend(indices[:n_treino])
        indices_teste.extend(indices[n_treino:])
    return rng.permutation(indices_treino), rng.permutation(indices_teste)


indices_treino, indices_teste = divisao_treino_teste_estratificada(y)
X_treino, X_teste = X[indices_treino], X[indices_teste]
y_treino, y_teste = y[indices_treino], y[indices_teste]

print(f"Treinamento: {len(X_treino)} amostras | Teste: {len(X_teste)} amostras")

"""## 2. Treinamento e matrizes de confusão (linhas = predição, colunas = referência)"""


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def imprimir_matriz(mc):
    largura = max(len(n) for n in NOMES) + 2
    print(" " * (largura + 6) + "Referência")
    print("Pred. ".ljust(largura + 6) + "".join(n[:10].rjust(12) for n in NOMES) + "Total".rjust(8))
    for i, nome in enumerate(NOMES):
        linha = "".join(f"{v:12d}" for v in mc[i])
        print(f"      {nome:<{largura}}{linha}{mc[i].sum():8d}")
    print(f"      {'Total':<{largura}}" + "".join(f"{v:12d}" for v in mc.sum(axis=0))
          + f"{mc.sum():8d}")


resultados = {}

for nome, construtor in CLASSIFICADORES.items():
    prever, info = construtor(X_treino, y_treino, CLASSES)
    mc = M.matriz_confusao(y_teste, prever(X_teste), CLASSES)
    resultados[nome] = {"mc": mc, "info": info, **M.metricas_multiclasse(mc)}

    titulo(nome)
    if "treino" in info:
        for t in info["treino"]:
            situacao = "convergiu" if t["convergiu"] else "NÃO convergiu"
            print(f"  {NOMES[t['classe']]:<10} × resto: {t['epocas']:4d} épocas ({situacao})")
    imprimir_matriz(mc)

r"""## 3. Questão 1 — métricas multiclasse

i)   acerto global A_g;
ii)  acurácia do produtor A_pi = a_ii / a_+i (sensibilidade de cada classe);
iii) acurácia do usuário  A_ui = a_ii / a_i+ (precisão de cada classe);
iv)  Kappa e sua variância;
v)   Tau e sua variância.
"""

titulo("Questão 1 (i a v) — métricas multiclasse no conjunto de teste")

cabecalho = f"{'Classificador':<24}{'A_g':>8}{'Kappa':>9}{'σ²_K':>11}{'Tau':>9}{'σ²_τ':>11}  Concordância"
print(cabecalho)
print("-" * len(cabecalho))
for nome, r in resultados.items():
    print(f"{nome:<24}{r['A_g']:8.4f}{r['kappa']:9.4f}{r['var_kappa']:11.6f}"
          f"{r['tau']:9.4f}{r['var_tau']:11.6f}  {M.interpretar_kappa(r['kappa'])}")

print("\nAcurácia do produtor (A_p) e do usuário (A_u) por classe:")
cabecalho = f"{'Classificador':<24}" + "".join(f"{'A_p ' + n[:4]:>11}" for n in NOMES) \
    + "".join(f"{'A_u ' + n[:4]:>11}" for n in NOMES)
print(cabecalho)
print("-" * len(cabecalho))
for nome, r in resultados.items():
    print(f"{nome:<24}" + "".join(f"{v:11.4f}" for v in r["A_p"])
          + "".join(f"{v:11.4f}" for v in r["A_u"]))

r"""## 4. Questão 1 (vi e vii) — problema de duas classes

MCC e F_b se aplicam a duas classes. Cada matriz 3×3 é reduzida a três
problemas binários "classe i × resto": VP = a_ii, FP = a_i+ - a_ii,
FN = a_+i - a_ii e VN = m - VP - FP - FN.
"""

titulo("Questão 1 (vi e vii) — MCC, F1 e F2 por classe (classe × resto)")

cabecalho = f"{'Classificador':<24}{'Classe':<12}{'VP':>4}{'FP':>4}{'FN':>4}{'VN':>4}" \
    f"{'Sens.':>8}{'Espec.':>8}{'Prec.':>8}{'F1':>8}{'F2':>8}{'MCC':>8}"
print(cabecalho)
print("-" * len(cabecalho))
for nome, r in resultados.items():
    r["binarias"] = []
    for i, classe in enumerate(NOMES):
        b = M.metricas_binarias(*M.binaria_da_multiclasse(r["mc"], i))
        r["binarias"].append(b)
        print(f"{nome if i == 0 else '':<24}{classe:<12}{b['VP']:4d}{b['FP']:4d}{b['FN']:4d}{b['VN']:4d}"
              f"{b['sensibilidade']:8.4f}{b['especificidade']:8.4f}{b['precisao']:8.4f}"
              f"{b['F1']:8.4f}{b['F2']:8.4f}{b['MCC']:8.4f}")
    medias = {k: np.mean([b[k] for b in r["binarias"]]) for k in ("F1", "F2", "MCC")}
    print(f"{'':<24}{'média':<12}{'':>16}{'':>24}"
          f"{medias['F1']:8.4f}{medias['F2']:8.4f}{medias['MCC']:8.4f}")
    print()

r"""## 5. Questão 2 — teste de significância (α = 5%)

H0: não há diferença entre os coeficientes; H1: há diferença.
Z = (k1 - k2) / sqrt(σ1² + σ2²) ~ N(0,1); rejeita-se H0 se |Z| > 1,96.
p-valor = P(|Z| > |z|).
"""


def imprimir_teste(nome1, nome2, coeficiente):
    r1, r2 = resultados[nome1], resultados[nome2]
    if coeficiente == "Kappa":
        teste = M.comparar_kappa(r1["mc"], r2["mc"])
        c1, c2 = r1["kappa"], r2["kappa"]
    else:
        teste = M.comparar_tau(r1["mc"], r2["mc"])
        c1, c2 = r1["tau"], r2["tau"]
    decisao = "rejeita H0 (diferem)" if teste["rejeita_h0"] else "não rejeita H0"
    print(f"{coeficiente:<6}{nome1:<22} {c1:.4f}  ×  {nome2:<20} {c2:.4f}"
          f"  Z = {teste['z']:+.4f}  p = {teste['p_valor']:.4f}  → {decisao}")
    return teste


titulo("Questão 2 — testes de significância pedidos")
testes_q2 = {
    ("Superfície 1×todos", "Perceptron 1×todos", "Kappa"): None,
    ("Regra delta 1×todos", "Perceptron 1×todos", "Kappa"): None,
    ("Regra delta 1×todos", "Perceptron 1×todos", "Tau"): None
}
for (n1, n2, coef) in testes_q2:
    testes_q2[(n1, n2, coef)] = imprimir_teste(n1, n2, coef)

titulo("Complemento — Z de Kappa para todos os pares de classificadores")
nomes = list(resultados)
print(f"{'':<24}" + "".join(f"{n[:12]:>14}" for n in nomes))
for a in nomes:
    linha = ""
    for b in nomes:
        z = M.comparar_kappa(resultados[a]["mc"], resultados[b]["mc"])["z"]
        linha += f"{z:14.3f}"
    print(f"{a:<24}{linha}")
print("Nenhum |Z| passa de 1,96?",
      all(abs(M.comparar_kappa(resultados[a]["mc"], resultados[b]["mc"])["z"]) <= M.Z_CRITICO_5
          for a in nomes for b in nomes))

"""## 6. Figura das matrizes de confusão"""

fig, eixos = plt.subplots(1, len(resultados), figsize=(4.2 * len(resultados), 4.2))
for ax, (nome, r) in zip(eixos, resultados.items()):
    mc = r["mc"]
    ax.imshow(mc, cmap="Blues", vmin=0, vmax=mc.sum(axis=0).max())
    for i in range(len(mc)):
        for j in range(len(mc)):
            ax.text(j, i, mc[i, j], ha="center", va="center",
                    color="white" if mc[i, j] > mc.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(3), NOMES, fontsize=8)
    ax.set_yticks(range(3), NOMES, fontsize=8)
    ax.set_xlabel("Referência")
    ax.set_ylabel("Predição")
    ax.set_title(f"{nome}\nA_g = {r['A_g']:.3f} | K = {r['kappa']:.3f}", fontsize=10)
fig.suptitle("Matrizes de confusão no conjunto de teste (Iris, 4 atributos)")
fig.tight_layout()
os.makedirs(PASTA_FIGURAS, exist_ok=True)
fig.savefig(os.path.join(PASTA_FIGURAS, "matrizes_confusao.png"), dpi=150, bbox_inches="tight")

r"""## 7. Conclusões

Valores do conjunto de teste (45 amostras, semente 42).

**Questão 1**
- Setosa é classificada sem erro por todos os métodos (A_p = A_u = 1,
  F1 = MCC = 1): ela é linearmente separável das outras duas espécies.
- Todos os erros estão entre Versicolor e Virginica. Os classificadores
  erram em direções diferentes, o que aparece nas acurácias do produtor e do
  usuário:
  - distância mínima e máximo da função de decisão empurram Virginica para
    Versicolor (A_p Virginica = 0,60; A_u Versicolor = 0,71). As predições dos
    dois são idênticas, como previsto (D² = x^T x - 2 d_j(x));
  - superfície 1×todos e perceptron fazem o contrário: perdem Versicolor
    (A_p Versicolor = 0,73) e acertam quase toda Virginica.
- Ranking por A_g e Kappa: superfície 1×todos (0,911 / 0,867), perceptron
  (0,889 / 0,833), distância mínima = máximo (0,867 / 0,800) e regra delta
  (0,822 / 0,733). Pela escala de Landis e Koch, os dois primeiros têm
  concordância quase perfeita e os demais, substancial.
- **Kappa = Tau em todos os casos.** Não é coincidência: o conjunto de teste
  é balanceado (15 amostras por classe), então a_+i = m/C e o acerto casual
  vira A_a = Σ a_i+ (m/C) / m² = 1/C, que é exatamente a hipótese do Tau. As
  variâncias, porém, são diferentes, porque a do Kappa usa a distribuição
  das predições (a_i+) e a do Tau não.
- No problema de duas classes (classe × resto), o MCC é a métrica mais
  severa porque usa os quatro valores da matriz, inclusive VN. A regra delta
  tem os menores MCC (0,62 e 0,59 para Versicolor e Virginica). O F2 pesa
  mais a sensibilidade: na distância mínima, Versicolor (sens. = 1) sobe de
  F1 = 0,83 para F2 = 0,93, e Virginica (sens. = 0,60) cai de 0,75 para 0,65.

**Questão 2**
- Superfície 1×todos × perceptron (Kappa): Z = 0,36 e p = 0,72. Não se
  rejeita H0: a diferença de Kappa (0,867 × 0,833) é compatível com o acaso
  da amostra. Os dois classificadores não diferem significativamente a 5%.
- Regra delta × perceptron: o perceptron tem maior acurácia (A_g = 0,889
  contra 0,822), mas Z = -0,91 (p = 0,36) para Kappa e Z = -0,90 (p = 0,37)
  para Tau. Também não se rejeita H0: com 45 amostras não há evidência de
  que um seja melhor que o outro.
- O p-valor é a probabilidade de observar uma diferença pelo menos tão
  grande quanto a obtida se os classificadores fossem equivalentes. Valores
  de 0,36 a 0,72 dizem que diferenças desse tamanho são comuns por acaso.
  Nenhum par de classificadores difere a 5% (tabela do complemento).
  Se houvesse diferença significativa (|Z| > 1,96, p < 0,05), isso
  significaria que a concordância de um classificador com a referência é
  maior que a do outro além do que a variação amostral explica.
- O motivo principal é o tamanho da amostra: com m = 45, σ_K ≈ 0,07, e uma
  diferença de Kappa só seria significativa a partir de cerca de 0,2.
- Observação sobre o perceptron simples: Versicolor × resto e Virginica ×
  resto não são linearmente separáveis, então o perceptron não converge e
  para em 1000 épocas com os pesos da última correção. O resultado dele
  depende de onde o treinamento parou. A regra delta converge (o EQM
  estabiliza) nas três classes, mas minimiza o erro quadrático e não o
  número de erros, o que explica a acurácia menor neste teste.
"""

plt.show()
