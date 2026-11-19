# -*- coding: utf-8 -*-
r"""Laboratório 04 — Classificador de Bayes ótimo e Naive Bayes (aula PR5)

Como executar (a partir da raiz do repositório):

    pip install numpy pandas matplotlib xlrd
    python lab4/lab4_bayes.py

A verificação de normalidade no R (opcional, os mesmos testes também são
feitos em Python por normalidade.py):

    Rscript lab4/normalidade.R

Atividade (sem bibliotecas de aprendizado de máquina, base Iris):
0) Verificar, para cada classe, a aderência à normal multivariada pelos
   testes de Henze-Zirkler (p > 0,05 → normal) e de Mardia (assimetria e
   curtose). Verificada a normalidade, P(x | C_j) é normal de parâmetros
   m_j e Σ_j.
a) 70% dos dados para treinamento e 30% para teste.
b) Classificar as três classes pelo máximo a posteriori (MAP) com o
   classificador ótimo de Bayes e com o Naive Bayes, com probabilidades a
   priori iguais, P(C_1) = P(C_2) = P(C_3), que são suprimidas.
c) Determinar as superfícies de decisão de Setosa × Virginica,
   Versicolor × Virginica e Setosa × Versicolor.
d) Em todos os casos, matriz de confusão e todas as métricas de qualidade
   estudadas (aula PR51).
e) Qual classificador tem maior acurácia no item (a)? Teste de
   significância do Kappa.

Organização do código:
- dados.py: leitura da Iris e divisão estratificada 70/30;
- normalidade.py: testes de Henze-Zirkler e Mardia (NumPy);
- normalidade.R: os mesmos testes com o pacote MVN do R;
- bayes.py: Bayes ótimo, Naive Bayes, classificação MAP e superfícies;
- metricas.py: matriz de confusão e métricas da PR51 (igual à do lab3);
- visualizacao.py: impressões e gráficos;
- lab4_bayes.py (este arquivo): execução dos itens e conclusões.

O arquivo pr5_bayesotimo_naivebayes.py é a versão original do Colab, que
deu origem a esta organização em módulos.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import metricas as M
from bayes import (
    MODELOS,
    prever_modelo,
    superficie_decisao,
    avaliar_superficie,
    fatia_superficie,
    formatar_equacao
)
from dados import (
    ATRIBUTOS,
    CLASSES,
    NOMES_CLASSES,
    SEED,
    carregar_iris,
    divisao_treino_teste_estratificada
)
from normalidade import henze_zirkler, mardia
from visualizacao import (
    titulo,
    imprimir_normalidade,
    imprimir_metricas,
    imprimir_teste,
    plotar_qq,
    plotar_matrizes,
    plotar_superficie,
    salvar_figura
)

np.set_printoptions(precision=4, suppress=True)

PASTA = os.path.dirname(os.path.abspath(__file__))
NOMES = [NOMES_CLASSES[c] for c in CLASSES]

PARES_CLASSES = [
    (0, 2),  # Setosa × Virginica
    (1, 2),  # Versicolor × Virginica
    (0, 1)   # Setosa × Versicolor
]

"""## 1. Dados"""

X, y = carregar_iris()
print("Dimensão de X:", X.shape, "| dimensão de y:", y.shape)

r"""## 2. Verificação da normalidade multivariada (por classe, 50 amostras)

Henze-Zirkler: se p > 0,05, não se rejeita H0 e os dados são considerados
vindos de uma população normal. Mardia complementa o HZ: assimetria e
curtose devem ter p > 0,05. Os valores abaixo coincidem com os do pacote MVN
do R (ver resultados_normalidade_R.txt, gerado por normalidade.R).
"""

titulo("Normalidade multivariada por classe (Henze-Zirkler e Mardia)")
resultados_normalidade = {
    c: (henze_zirkler(X[y == c]), mardia(X[y == c]))
    for c in CLASSES
}
imprimir_normalidade(resultados_normalidade, NOMES_CLASSES)

arquivo_r = os.path.join(PASTA, "resultados_normalidade_R.txt")
if os.path.exists(arquivo_r):
    print(f"\nResultados do R: {os.path.relpath(arquivo_r)}")
else:
    print("\nPara conferir no R: Rscript lab4/normalidade.R")

plotar_qq(X, y, CLASSES, NOMES_CLASSES, "qq_quiquadrado")

r"""## 3. Item (a) — divisão 70% treinamento / 30% teste"""

X_treino, X_teste, y_treino, y_teste = divisao_treino_teste_estratificada(X, y, 0.70, SEED)

titulo("Item (a) — divisão estratificada 70/30")
print(f"Treinamento: {len(X_treino)} amostras | Teste: {len(X_teste)} amostras")
for c in CLASSES:
    print(f"  {NOMES_CLASSES[c]:<11} treino: {np.sum(y_treino == c)} | teste: {np.sum(y_teste == c)}")

r"""## 4. Item (b) — classificação MAP das três classes

Com priors iguais, a regra MAP escolhe a classe de maior g_j(x), isto é, de
maior verossimilhança P(x | C_j).
"""

parametros = {}
predicoes = {}
for nome, modelo in MODELOS.items():
    parametros[nome] = modelo["treinar"](X_treino, y_treino)
    predicoes[nome] = prever_modelo(X_teste, parametros[nome], modelo["pontuar"])

titulo("Item (b) — parâmetros estimados no treinamento")
for c in CLASSES:
    print(f"\n{NOMES_CLASSES[c]}: m_j = {parametros['Bayes ótimo'][c]['media']}")
    print("  Σ_j (Bayes ótimo):")
    print("  " + str(parametros["Bayes ótimo"][c]["covariancia"]).replace("\n", "\n  "))
    print(f"  diag(Σ_j) (Naive Bayes): {parametros['Naive Bayes'][c]['variancias']}")

r"""## 5. Item (c) — superfícies de decisão

d_ij(x) = g_i(x) - g_j(x) = x^T A x + b^T x + c = 0, com os quatro atributos
(x1 = comprimento da sépala, x2 = largura da sépala, x3 = comprimento da
pétala, x4 = largura da pétala). d_ij(x) > 0 → classe i; d_ij(x) < 0 →
classe j. No Naive Bayes, A é diagonal e não há termos cruzados x_k·x_l.

Para desenhar, o classificador continua com os quatro atributos, mas o
gráfico mostra uma fatia 2D no plano da pétala (x3, x4), com a sépala (x1,
x2) fixa na média do treinamento das duas classes do par. Nessa fatia, a
superfície é a cônica impressa como "fatia".
"""

INDICES_VISUAIS = (2, 3)  # comprimento e largura da pétala
INDICES_FIXOS = [k for k in range(4) if k not in INDICES_VISUAIS]
NOMES_X = ["x1", "x2", "x3", "x4"]

titulo("Item (c) — superfícies de decisão d_ij(x) = 0")

superficies = {}
fig, eixos = plt.subplots(len(PARES_CLASSES), len(MODELOS), figsize=(13, 16))

for linha, (ca, cb) in enumerate(PARES_CLASSES):
    mascara_treino = (y_treino == ca) | (y_treino == cb)
    X_treino_par = X_treino[mascara_treino]
    valores_fixos = X_treino_par[:, INDICES_FIXOS].mean(axis=0)

    for coluna, (nome, modelo) in enumerate(MODELOS.items()):
        A, b, c = superficie_decisao(parametros[nome], ca, cb)

        # Confere a expansão algébrica contra g_a(x) - g_b(x)
        diferenca = (
            modelo["pontuar"](X_teste, parametros[nome][ca])
            - modelo["pontuar"](X_teste, parametros[nome][cb])
        )
        assert np.allclose(avaliar_superficie(X_teste, A, b, c), diferenca)

        A2, b2, c2 = fatia_superficie(A, b, c, INDICES_VISUAIS, valores_fixos)
        superficies[(ca, cb, nome)] = {"A": A, "b": b, "c": c, "fatia": (A2, b2, c2)}

        print(f"\n{nome} — {NOMES_CLASSES[ca]} (d > 0) × {NOMES_CLASSES[cb]} (d < 0)")
        print(f"  4D:    {formatar_equacao(A, b, c, NOMES_X)}")
        print(f"  fatia: {formatar_equacao(A2, b2, c2, ['x3', 'x4'])}"
              f"   (x1 = {valores_fixos[0]:.3f}, x2 = {valores_fixos[1]:.3f})")

        plotar_superficie(
            eixos[linha, coluna],
            lambda G, A=A, b=b, c=c: avaliar_superficie(G, A, b, c),
            X_treino_par, X_teste, y_teste, ca, cb,
            INDICES_VISUAIS, valores_fixos, NOMES_CLASSES, ATRIBUTOS,
            f"{nome}: {NOMES_CLASSES[ca]} × {NOMES_CLASSES[cb]}\n"
            f"(sépala fixa em x1 = {valores_fixos[0]:.2f}, x2 = {valores_fixos[1]:.2f})"
        )

fig.suptitle("Superfícies de decisão (fatia no plano da pétala, sépala na média do par)", fontsize=14)
fig.tight_layout()
salvar_figura(fig, "superficies_decisao")

r"""## 6. Item (d) — matrizes de confusão e métricas

Convenção da aula: linhas = predição e colunas = referência.
- três classes, com o resultado do item (b);
- os três pares do item (c): cada classificador decide só entre as duas
  classes do par, com as amostras de teste dessas duas classes.
"""

titulo("Item (d) — três classes")
resultados_3 = {}
for nome in MODELOS:
    mc = M.matriz_confusao(y_teste, predicoes[nome], CLASSES)
    resultados_3[nome] = {"mc": mc, **imprimir_metricas(nome, mc, NOMES)}

plotar_matrizes({n: r["mc"] for n, r in resultados_3.items()}, NOMES, "matrizes_confusao")

resultados_pares = {}
for ca, cb in PARES_CLASSES:
    titulo(f"Item (d) — {NOMES_CLASSES[ca]} × {NOMES_CLASSES[cb]}")
    mascara = (y_teste == ca) | (y_teste == cb)
    for nome, modelo in MODELOS.items():
        pred = prever_modelo(X_teste[mascara], parametros[nome], modelo["pontuar"], [ca, cb])
        mc = M.matriz_confusao(y_teste[mascara], pred, [ca, cb])
        resultados_pares[(ca, cb, nome)] = {
            "mc": mc,
            **imprimir_metricas(nome, mc, [NOMES_CLASSES[ca], NOMES_CLASSES[cb]])
        }

r"""## 7. Item (e) — teste de significância (α = 5%)

H0: não há diferença entre os coeficientes; H1: há diferença.
Z = (k1 - k2) / sqrt(σ1² + σ2²) ~ N(0,1); rejeita-se H0 se |Z| > 1,96.
"""

titulo("Item (e) — Bayes ótimo × Naive Bayes (três classes)")
bayes, naive = resultados_3["Bayes ótimo"], resultados_3["Naive Bayes"]

print(f"Acerto global: Bayes ótimo = {bayes['A_g']:.4f} | Naive Bayes = {naive['A_g']:.4f}")
teste_kappa = M.comparar_kappa(bayes["mc"], naive["mc"])
teste_tau = M.comparar_tau(bayes["mc"], naive["mc"])
imprimir_teste("Kappa", bayes["kappa"], naive["kappa"], teste_kappa)
imprimir_teste("Tau", bayes["tau"], naive["tau"], teste_tau)
print(f"σ²_K: Bayes ótimo = {bayes['var_kappa']:.6f} | Naive Bayes = {naive['var_kappa']:.6f}")
print("Obs.: com K = 1 a variância do Kappa é 0, então o Z usa só a variância do Naive Bayes.")

r"""## 8. Complemento — sensibilidade à divisão 70/30

Com 45 amostras de teste, um único erro muda o acerto global em 2,2 pontos.
Repetir o experimento com outras sementes mostra quanto o resultado do item
(e) depende da divisão escolhida.
"""

titulo("Complemento — Bayes ótimo × Naive Bayes em 10 divisões 70/30")
print(f"{'Semente':>8}{'A_g Bayes':>11}{'A_g Naive':>11}{'K Bayes':>9}{'K Naive':>9}{'Z':>9}{'p':>8}")
vitorias = {"Bayes ótimo": 0, "Naive Bayes": 0, "empate": 0}
for semente in range(10):
    Xtr, Xte, ytr, yte = divisao_treino_teste_estratificada(X, y, 0.70, semente)
    mcs = {}
    for nome, modelo in MODELOS.items():
        p = modelo["treinar"](Xtr, ytr)
        mcs[nome] = M.matriz_confusao(yte, prever_modelo(Xte, p, modelo["pontuar"]), CLASSES)
    t = M.comparar_kappa(mcs["Bayes ótimo"], mcs["Naive Bayes"])
    ag_b, ag_n = M.acerto_global(mcs["Bayes ótimo"]), M.acerto_global(mcs["Naive Bayes"])
    vitorias["Bayes ótimo" if ag_b > ag_n else "Naive Bayes" if ag_n > ag_b else "empate"] += 1
    print(f"{semente:8d}{ag_b:11.4f}{ag_n:11.4f}{M.kappa(mcs['Bayes ótimo']):9.4f}"
          f"{M.kappa(mcs['Naive Bayes']):9.4f}{t['z']:9.3f}{t['p_valor']:8.3f}")
print(f"Maior acerto global: {vitorias}")

r"""## 9. Conclusões

**Normalidade (item 0).**
- Versicolor (HZ: p = 0,226) e Virginica (p = 0,497) são compatíveis com a
  normal multivariada, e os testes de Mardia confirmam (assimetria e
  curtose com p > 0,15 nas duas).
- Setosa: HZ = 0,9488 com p = 0,04995. O MVN do R exibe "p = 0,05" porque
  arredonda para três casas, mas o valor completo (conferido no R e na
  implementação em Python) é menor que 0,05: rejeita-se H0 pelo HZ, por
  margem mínima. Os testes de Mardia não acusam desvio (assimetria
  p = 0,177; curtose p = 0,195), e o gráfico Q-Q mostra só um afastamento
  leve na cauda superior. A Setosa é, portanto, um caso limítrofe.
- Consequência para o classificador: o modelo gaussiano é usado nas três
  classes. Para a Setosa a hipótese é só aproximada, mas isso não afeta o
  resultado na prática, porque a Setosa é separada sem erro pelos dois
  classificadores em todos os casos.

**Itens (b) e (d).**
- Três classes: o Bayes ótimo acertou as 45 amostras de teste (A_g = 1,
  K = τ = 1, MCC = 1). O Naive Bayes errou uma Virginica, classificada como
  Versicolor (A_g = 0,9778, K = τ = 0,9667, σ²_K = 0,0011).
- Pares: em Setosa × Virginica e Setosa × Versicolor os dois acertam tudo.
  Em Versicolor × Virginica o Bayes ótimo acerta tudo e o Naive Bayes
  repete o erro da Virginica (A_g = 0,9667, K = 0,9333).
- O Bayes ótimo leva vantagem em Versicolor × Virginica porque modela as
  correlações entre os atributos (por exemplo, comprimento e largura da
  pétala), que o Naive Bayes ignora ao supor independência.

**Item (c).** As superfícies do Bayes ótimo são quádricas com termos
cruzados x_k·x_l, porque Σ_i ≠ Σ_j; as do Naive Bayes também são
quadráticas, mas sem termos cruzados. As equações completas em 4D e as
cônicas da fatia no plano da pétala são impressas acima. A fatia é só uma
visualização: em Versicolor × Virginica (Bayes ótimo), a Virginica de teste
em (x3, x4) = (5,0; 1,5) aparece do lado da Versicolor no gráfico, mas é
classificada corretamente em 4D, porque a sépala dela não é a média usada
na fatia.

**Item (e).** O Bayes ótimo teve a maior acurácia (100% contra 97,78%), mas
a diferença não é significativa: Z = 1,012 e p = 0,312 para o Kappa (o
mesmo para o Tau), então não se rejeita H0 ao nível de 5%. Não há evidência
estatística de que um classificador seja melhor que o outro. O complemento
mostra a mesma coisa com outras divisões: em 10 sementes, o Bayes ótimo
teve acerto maior em 5, empatou em 4 e perdeu em 1 (semente 1), e nenhuma
diferença foi significativa (maior |Z| = 1,01, justamente na semente 42
usada no item (a)). A tendência favorece o Bayes ótimo, mas com 45
amostras de teste ela não chega a ser estatisticamente comprovada.
"""

plt.show()
