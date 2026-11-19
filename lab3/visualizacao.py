# -*- coding: utf-8 -*-
r"""Perceptron com regra delta (aula PR41) — exibição dos resultados

Tudo o que mostra os resultados dos experimentos fica aqui:
- impressões no terminal (convergência, pesos, tabelas verdade e resumo);
- gráficos ÉPOCA × EQM, dispersão 2D da Iris e fronteiras das tabelas
  verdade, salvos em figuras/, na pasta deste arquivo.

Os resultados recebidos são os dicionários devolvidos por
`regra_delta.treinar_regra_delta`, completados pelo script principal com
"nome", "separavel" e, conforme o experimento, os dados de teste ou o EQM
mínimo.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from regra_delta import MAX_EPOCAS, aumentar, classificar, resultado_converge

PASTA_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")


def salvar_figura(nome_arquivo):
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    plt.savefig(
        os.path.join(PASTA_FIGURAS, f"{nome_arquivo}.png"),
        dpi=150,
        bbox_inches="tight"
    )


def sim_nao(valor):
    return "sim" if valor else "não"


def rotulo_convergencia(resultado):
    return "CONVERGE" if resultado_converge(resultado) else "NÃO CONVERGE"


# ---------------------------------------------------------------------------
# Impressões no terminal
# ---------------------------------------------------------------------------

def imprimir_cabecalho(titulo):
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)


def imprimir_convergencia(resultado):
    erros = resultado["erros"]
    epocas_sem_erro = np.where(erros == 0)[0]

    print(f"Épocas executadas: {resultado['epocas']}")
    print(
        f"EQM final: {resultado['eqm'][-1]:.6f}"
        f" (estabilizou: {sim_nao(resultado['eqm_estabilizou'])})"
    )
    print(f"Erros de classificação no treino (última época): {erros[-1]}")

    if resultado_converge(resultado):
        print(
            ">>> CONVERGE: o vetor de pesos separa as classes de treino"
            f" (primeira época sem erros: {epocas_sem_erro[0]})."
        )
    elif resultado["eqm_estabilizou"]:
        print(
            ">>> NÃO CONVERGE para uma separação: o EQM estabilizou, mas a"
            f" solução de EQM mínimo deixa {erros[-1]} erro(s) no treino."
        )
    else:
        print(
            ">>> NÃO CONVERGE: o EQM ainda variava ao atingir o limite de"
            f" {MAX_EPOCAS} épocas."
        )

    print(
        "Classes linearmente separáveis no treino (perceptron simples):"
        f" {sim_nao(resultado['separavel'])}"
    )


def imprimir_resultados_iris(resultados, descricao):
    for resultado in resultados:
        n_atributos = len(resultado["w"]) - 1
        rotulos = "[" + ", ".join(
            [f"w{j}" for j in range(1, n_atributos + 1)] + ["w0"]
        ) + "]"

        imprimir_cabecalho(f"{resultado['nome']} — {descricao}")
        print(f"Pesos (atributos padronizados) {rotulos}:")
        print(resultado["w"])
        print(f"Pesos (unidades originais)     {rotulos}:")
        print(resultado["w_original"])
        imprimir_convergencia(resultado)
        print(
            f"Teste: {len(resultado['r_teste'])} amostras,"
            f" {resultado['erros_teste']} erros,"
            f" acurácia = {resultado['acuracia_teste'] * 100:.2f}%"
        )


def imprimir_tabela_verdade(resultado, X_tabela):
    w = resultado["w"]
    X_aumentado = aumentar(X_tabela)
    g = X_aumentado @ w
    pred = classificar(X_aumentado, w)
    saidas = np.where(resultado["respostas"] == 1, 1, 0)

    imprimir_cabecalho(f"Tabela verdade {resultado['nome']}")
    print("Pesos [w1, w2, w0]:", w)
    imprimir_convergencia(resultado)
    print(
        "Menor EQM possível (mínimos quadrados):"
        f" {resultado['eqm_minimo']:.6f}"
    )
    print("\n x1 x2 | desejado | g(x)     | saída")
    for entrada, desejado, gx, previsto in zip(X_tabela, saidas, g, pred):
        print(
            f"  {int(entrada[0])}  {int(entrada[1])} |    {desejado}     |"
            f" {gx:+.4f}  |   {1 if previsto == 1 else 0}"
        )
    print(f"Acertos: {resultado['acertos']}/4")


def imprimir_resumo(experimentos_iris, resultados_tabelas):
    """`experimentos_iris` é uma lista de pares (sufixo, resultados)."""
    print("\n" + "=" * 108)
    print(
        f"{'Experimento':<34}{'Épocas':>7}{'EQM final':>11}"
        f"{'Erros treino':>14}{'Acurácia teste':>16}{'Separável':>11}   Veredito"
    )
    print("-" * 108)

    for sufixo, resultados in experimentos_iris:
        for resultado in resultados:
            print(
                f"{resultado['nome'] + ' (' + sufixo + ')':<34}"
                f"{resultado['epocas']:>7}{resultado['eqm'][-1]:>11.4f}"
                f"{resultado['erros'][-1]:>14}"
                f"{resultado['acuracia_teste'] * 100:>15.2f}%"
                f"{sim_nao(resultado['separavel']):>11}"
                f"   {rotulo_convergencia(resultado)}"
            )

    for resultado in resultados_tabelas:
        print(
            f"{resultado['nome']:<34}"
            f"{resultado['epocas']:>7}{resultado['eqm'][-1]:>11.4f}"
            f"{resultado['erros'][-1]:>14}{'—':>16}"
            f"{sim_nao(resultado['separavel']):>11}"
            f"   {rotulo_convergencia(resultado)}"
        )


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plotar_eqm(ax, resultado, **kwargs):
    eqm = resultado["eqm"]
    ax.plot(range(len(eqm)), eqm, lw=1.8, **kwargs)
    ax.set_xlabel("Época")
    ax.set_ylabel("EQM (MSE)")
    ax.grid(alpha=0.3)


def plotar_eqm_por_experimento(resultados, titulo, nome_arquivo):
    """Um gráfico ÉPOCA × EQM por experimento, lado a lado.

    Se o resultado tiver "eqm_minimo", ele é desenhado como referência.
    """
    fig, axs = plt.subplots(1, len(resultados), figsize=(15, 4))
    for ax, resultado in zip(axs, resultados):
        plotar_eqm(ax, resultado)
        if "eqm_minimo" in resultado:
            ax.axhline(resultado["eqm_minimo"], color="gray", ls="--", lw=1,
                       label=f"EQM mínimo = {resultado['eqm_minimo']:.4f}")
            ax.legend(fontsize=8)
        ax.set_title(
            f"{resultado['nome']} — {rotulo_convergencia(resultado)}\n"
            f"{resultado['epocas']} épocas, EQM final = {resultado['eqm'][-1]:.4f}"
        )
    fig.suptitle(titulo)
    plt.tight_layout()
    salvar_figura(nome_arquivo)


def plotar_eqm_comparacao(resultados, titulo, nome_arquivo):
    """Curvas ÉPOCA × EQM de vários experimentos no mesmo gráfico."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for resultado in resultados:
        plotar_eqm(ax, resultado, label=resultado["nome"])
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    salvar_figura(nome_arquivo)


def plotar_dispersao_2d(resultados, nomes_classes, nomes_eixos, nome_arquivo):
    """Dispersão de treino e teste com a reta g(x) = 0 de cada par.

    Para dois atributos: w_1 x_1 + w_2 x_2 + w_0 = 0
                         →  x_2 = -(w_1 x_1 + w_0) / w_2
    """
    fig, axs = plt.subplots(1, len(resultados), figsize=(16, 4.8))
    for ax, resultado in zip(axs, resultados):
        w1, w2, w0 = resultado["w_original"]
        Xtr, rtr = resultado["X_treino"], resultado["r_treino"]
        Xte, rte = resultado["X_teste"], resultado["r_teste"]
        nome_a = nomes_classes[resultado["classe_a"]]
        nome_b = nomes_classes[resultado["classe_b"]]

        for X_conj, r_conj, opaco, sufixo in (
            (Xtr, rtr, 0.3, "treino"),
            (Xte, rte, 1.0, "teste")
        ):
            ax.scatter(*X_conj[r_conj == 1].T, c="tab:blue", marker="o",
                       alpha=opaco, label=f"{nome_a} ({sufixo})")
            ax.scatter(*X_conj[r_conj == -1].T, c="tab:red", marker="s",
                       alpha=opaco, label=f"{nome_b} ({sufixo})")

        erros = classificar(aumentar(Xte), resultado["w_original"]) != rte
        ax.scatter(*Xte[erros].T, facecolors="none", edgecolors="k", s=160,
                   lw=1.5, label="erro no teste")

        todos = np.r_[Xtr, Xte]
        x1 = np.linspace(todos[:, 0].min() - 0.3, todos[:, 0].max() + 0.3, 100)
        if abs(w2) > 1e-12:
            ax.plot(x1, -(w1 * x1 + w0) / w2, "k-", lw=2, label="g(x) = 0")

        ax.set_xlim(x1[0], x1[-1])
        ax.set_ylim(todos[:, 1].min() - 0.3, todos[:, 1].max() + 0.3)
        ax.set_xlabel(nomes_eixos[0])
        ax.set_ylabel(nomes_eixos[1])
        ax.set_title(
            f"{resultado['nome']} — {rotulo_convergencia(resultado)}\n"
            f"acurácia no teste = {resultado['acuracia_teste'] * 100:.2f}%"
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    salvar_figura(nome_arquivo)


def plotar_fronteiras_tabelas(resultados, X_tabela, nome_arquivo):
    """Padrões de cada tabela verdade e a reta g(x) = 0 encontrada."""
    fig, axs = plt.subplots(1, len(resultados), figsize=(15, 4.5))
    for ax, resultado in zip(axs, resultados):
        w1, w2, w0 = resultado["w"]
        respostas = resultado["respostas"]

        ax.scatter(*X_tabela[respostas == 1].T, c="tab:blue", s=120,
                   label="saída 1")
        ax.scatter(*X_tabela[respostas == -1].T, c="tab:red", marker="s",
                   s=120, label="saída 0")

        x1 = np.linspace(-0.5, 1.5, 100)
        if abs(w2) > 1e-9:
            ax.plot(x1, -(w1 * x1 + w0) / w2, "k-", lw=2, label="g(x) = 0")

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"{resultado['nome']} — {resultado['acertos']}/4 acertos")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    salvar_figura(nome_arquivo)
