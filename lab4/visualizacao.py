# -*- coding: utf-8 -*-
r"""Exibição dos resultados do Laboratório 04

- impressões no terminal: normalidade, matrizes de confusão, métricas,
  equações das superfícies e testes de significância;
- gráficos: Q-Q de qui-quadrado, superfícies de decisão e matrizes de
  confusão, salvos em figuras/, na pasta deste arquivo.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import metricas as M
from normalidade import distancias_mahalanobis, qui_quadrado_quantil

PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")


def salvar_figura(fig, nome_arquivo):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    fig.savefig(os.path.join(PASTA_FIGURAS, f"{nome_arquivo}.png"), dpi=150, bbox_inches="tight")


def titulo(texto, caractere="="):
    print("\n" + caractere * 78)
    print(texto)
    print(caractere * 78)


# ---------------------------------------------------------------------------
# Normalidade
# ---------------------------------------------------------------------------

def imprimir_normalidade(resultados, nomes_classes):
    """resultados: {classe: (resultado_hz, resultado_mardia)}."""
    print(f"{'Classe':<12}{'HZ':>9}{'p (HZ)':>11}  {'Veredito HZ':<13}"
          f"{'Assim.':>9}{'p':>9}{'Curt.':>9}{'p':>9}  Veredito Mardia")
    print("-" * 100)
    for classe, (hz, md) in resultados.items():
        print(f"{nomes_classes[classe]:<12}{hz['estatistica']:9.4f}{hz['p_valor']:11.6f}  "
              f"{'normal' if hz['normal'] else 'NÃO normal':<13}"
              f"{md['assimetria']:9.3f}{md['p_assimetria']:9.4f}"
              f"{md['curtose']:9.3f}{md['p_curtose']:9.4f}  "
              f"{'normal' if md['normal'] else 'NÃO normal'}")


def plotar_qq(X, y, classes, nomes_classes, nome_arquivo):
    """Q-Q de qui-quadrado: D_i² ordenadas contra os quantis da χ²(p)."""
    fig, eixos = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.5))
    p = X.shape[1]
    for ax, classe in zip(eixos, classes):
        d2 = np.sort(distancias_mahalanobis(X[y == classe]))
        n = len(d2)
        quantis = [qui_quadrado_quantil((i - 0.5) / n, p) for i in range(1, n + 1)]
        ax.scatter(quantis, d2, s=18)
        limite = max(max(quantis), d2.max())
        ax.plot([0, limite], [0, limite], color="firebrick", lw=1.5)
        ax.set_title(f"{nomes_classes[classe]}")
        ax.set_xlabel(f"Quantis da $\\chi^2({p})$")
        ax.set_ylabel("Distância de Mahalanobis ao quadrado")
        ax.grid(alpha=0.3)
    fig.suptitle("Gráfico Q-Q de qui-quadrado por classe (Iris, 4 atributos)")
    fig.tight_layout()
    salvar_figura(fig, nome_arquivo)


# ---------------------------------------------------------------------------
# Matriz de confusão e métricas
# ---------------------------------------------------------------------------

def imprimir_matriz(mc, nomes):
    largura = max(len(n) for n in nomes) + 2
    print(" " * (largura + 6) + "Referência")
    print("Pred. ".ljust(largura + 6) + "".join(n.rjust(12) for n in nomes) + "Total".rjust(8))
    for i, nome in enumerate(nomes):
        print(f"      {nome:<{largura}}" + "".join(f"{v:12d}" for v in mc[i]) + f"{mc[i].sum():8d}")
    print(f"      {'Total':<{largura}}" + "".join(f"{v:12d}" for v in mc.sum(axis=0))
          + f"{mc.sum():8d}")


def imprimir_metricas(nome, mc, nomes):
    """Todas as métricas da aula PR51 para uma matriz de confusão."""
    r = M.metricas_multiclasse(mc)

    titulo(nome, "-")
    imprimir_matriz(mc, nomes)

    print(f"\nAcerto global A_g = {r['A_g']:.4f} | acerto casual A_a = {r['A_a']:.4f}")
    print(f"Kappa = {r['kappa']:.4f} ({M.interpretar_kappa(r['kappa'])}) | σ²_K = {r['var_kappa']:.6f}")
    print(f"Tau   = {r['tau']:.4f} | σ²_τ = {r['var_tau']:.6f}")

    print(f"\n{'Classe':<12}{'A_p':>8}{'A_u':>8}{'VP':>5}{'FP':>4}{'FN':>4}{'VN':>4}"
          f"{'Sens.':>8}{'Espec.':>8}{'Prec.':>8}{'F1':>8}{'F2':>8}{'MCC':>8}")
    binarias = []
    for i, classe in enumerate(nomes):
        b = M.metricas_binarias(*M.binaria_da_multiclasse(mc, i))
        binarias.append(b)
        print(f"{classe:<12}{r['A_p'][i]:8.4f}{r['A_u'][i]:8.4f}"
              f"{b['VP']:5d}{b['FP']:4d}{b['FN']:4d}{b['VN']:4d}"
              f"{b['sensibilidade']:8.4f}{b['especificidade']:8.4f}{b['precisao']:8.4f}"
              f"{b['F1']:8.4f}{b['F2']:8.4f}{b['MCC']:8.4f}")
    if len(nomes) > 2:
        medias = {k: np.mean([b[k] for b in binarias]) for k in ("F1", "F2", "MCC")}
        print(f"{'média':<12}{'':>44}{'':>24}{medias['F1']:8.4f}{medias['F2']:8.4f}{medias['MCC']:8.4f}")
    else:
        print("(duas classes: MCC e F_b usam a primeira classe do par como positiva;"
              " o MCC é o mesmo nas duas linhas)")

    r["binarias"] = binarias
    return r


def imprimir_teste(rotulo, coef_1, coef_2, teste):
    decisao = "rejeita H0 (diferem)" if teste["rejeita_h0"] else "não rejeita H0"
    print(f"{rotulo:<6} {coef_1:.4f} × {coef_2:.4f}  Z = {teste['z']:+.4f}  "
          f"p = {teste['p_valor']:.4f}  → {decisao}")


def plotar_matrizes(matrizes, nomes, nome_arquivo):
    """matrizes: {titulo: mc}."""
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(4.6 * len(matrizes), 4.4))
    for ax, (rotulo, mc) in zip(np.atleast_1d(eixos), matrizes.items()):
        ax.imshow(mc, cmap="Blues", vmin=0, vmax=mc.sum(axis=0).max())
        for i in range(len(mc)):
            for j in range(len(mc)):
                ax.text(j, i, mc[i, j], ha="center", va="center", fontsize=12,
                        color="white" if mc[i, j] > mc.max() / 2 else "black")
        ax.set_xticks(range(len(nomes)), nomes, fontsize=8)
        ax.set_yticks(range(len(nomes)), nomes, fontsize=8)
        ax.set_xlabel("Referência")
        ax.set_ylabel("Predição")
        ax.set_title(f"{rotulo}\nA_g = {M.acerto_global(mc):.3f} | K = {M.kappa(mc):.3f}", fontsize=10)
    fig.suptitle("Matrizes de confusão no conjunto de teste (3 classes)")
    fig.tight_layout()
    salvar_figura(fig, nome_arquivo)


# ---------------------------------------------------------------------------
# Superfícies de decisão
# ---------------------------------------------------------------------------

def plotar_superficie(ax, funcao_d, X_treino_par, X_teste, y_teste, classe_a, classe_b,
                      indices_visuais, valores_fixos, nomes_classes, nomes_atributos, titulo_grafico):
    """Fatia 2D de d_ab(x): os atributos fora de `indices_visuais` ficam fixos.

    `funcao_d(X)` devolve d_ab(x) = g_a(x) - g_b(x) para vetores de 4 atributos.
    """
    ix, iy = indices_visuais
    fixos = [k for k in range(X_treino_par.shape[1]) if k not in indices_visuais]

    xmin, xmax = X_treino_par[:, ix].min(), X_treino_par[:, ix].max()
    ymin, ymax = X_treino_par[:, iy].min(), X_treino_par[:, iy].max()
    mx, my = (xmax - xmin) * 0.15, (ymax - ymin) * 0.15
    xx, yy = np.meshgrid(np.linspace(xmin - mx, xmax + mx, 400),
                         np.linspace(ymin - my, ymax + my, 400))

    grade = np.zeros((xx.size, X_treino_par.shape[1]))
    grade[:, fixos] = valores_fixos
    grade[:, ix] = xx.ravel()
    grade[:, iy] = yy.ravel()
    d = funcao_d(grade).reshape(xx.shape)

    ax.contourf(xx, yy, (d > 0).astype(int), levels=[-0.5, 0.5, 1.5],
                colors=["C1", "C0"], alpha=0.15)
    if d.min() <= 0 <= d.max():
        ax.contour(xx, yy, d, levels=[0], colors="k", linewidths=2)

    for classe, cor in [(classe_a, "C0"), (classe_b, "C1")]:
        mascara = y_teste == classe
        ax.scatter(X_teste[mascara, ix], X_teste[mascara, iy], c=cor,
                   edgecolors="k", s=35, label=f"{nomes_classes[classe]} (teste)")

    ax.plot([], [], "k", lw=2, label="$d_{ij}(x) = 0$")
    ax.set_title(titulo_grafico, fontsize=10)
    ax.set_xlabel(nomes_atributos[ix])
    ax.set_ylabel(nomes_atributos[iy])
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
