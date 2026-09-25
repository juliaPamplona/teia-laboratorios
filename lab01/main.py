import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import distancia_minima
import funcao_decisao
import superficie_decisao

PASTA = os.path.dirname(os.path.abspath(__file__))
PARES = [('setosa', 'versicolor'), ('setosa', 'virginica'), ('versicolor', 'virginica')]
TARGET = 'Species'


def dividir_amostra(df, classes, frac_treino=0.7, semente=42):
    """Divisão estratificada: 70% treino / 30% teste dentro de cada classe
    (35 / 15), preservando a proporção de 1/3 por classe nos dois conjuntos."""
    np.random.seed(semente)
    train_idx, test_idx = [], []
    for c in classes:
        idx_c = np.random.permutation(np.where(df[TARGET] == c)[0])
        n_train = int(round(frac_treino * len(idx_c)))
        train_idx.extend(idx_c[:n_train])
        test_idx.extend(idx_c[n_train:])
    return df.iloc[train_idx], df.iloc[test_idx]


def matriz_confusao(y_real, y_pred, classes):
    mc = pd.DataFrame(0, index=classes, columns=classes)
    for r, p in zip(y_real, y_pred):
        mc.loc[r, p] += 1
    return mc


def main():
    # 1. Carregamento dos dados
    df = pd.read_excel(os.path.join(PASTA, "Iris data.xls"))
    atributos = df.columns[2:4].tolist() # 3ª e 4ª colunas (Petal length, Petal width)
    classes = df[TARGET].unique()        # setosa, versicolor, virginica

    # 2. Divisão da amostra (70% Treinamento / 30% Teste)
    train_data, test_data = dividir_amostra(df, classes)
    X_test, y_test = test_data[atributos].values, test_data[TARGET].values

    print(f"Atributos utilizados: {atributos}")
    print(f"Treinamento: {len(train_data)} amostras | Teste: {len(test_data)} amostras")
    print("Amostras de teste por classe:", test_data[TARGET].value_counts().to_dict())

    # 3. Protótipos (vetor média) de cada classe, estimados no conjunto de treino
    prototipos = distancia_minima.calcular_prototipos(train_data, atributos, TARGET, classes)

    print("\nVetores protótipos:")
    for c in classes:
        print(f"  m({c}) = {np.round(prototipos[c], 3)}")

    print("\nFunções de decisão:")
    for c in classes:
        print(f"  d_{c}(x) = {funcao_decisao.formatar(prototipos[c])}")

    # 4. Avaliação no conjunto de teste: (i) e (ii)
    pred_dist_min = distancia_minima.classificar(X_test, prototipos)
    pred_max_dec = funcao_decisao.classificar(X_test, prototipos)

    acc_dist_min = np.mean(np.array(pred_dist_min) == y_test)
    acc_max_dec = np.mean(np.array(pred_max_dec) == y_test)

    print(f"\nAcurácia Distância Mínima: {acc_dist_min * 100:.2f}%")
    print(f"Acurácia Máximo Função Decisão: {acc_max_dec * 100:.2f}%")
    print("Predições idênticas nos dois classificadores:", pred_dist_min == pred_max_dec)

    print("\nMatriz de confusão (linhas = classe real, colunas = classe predita):")
    print(matriz_confusao(y_test, pred_dist_min, classes))

    # 5. (iii) Superfícies de decisão (retas) e regra d_ij(x)
    print("\nSuperfícies de decisão (retas) e regra d_ij(x):")
    for c1, c2 in PARES:
        w, w0 = superficie_decisao.coef_superficie(prototipos[c1], prototipos[c2])
        print(f"  {c1} x {c2}: d_ij(x) = {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w0:.2f} = 0")
        for nome, dados in [('teste', test_data), ('dataset completo', df)]:
            sub = dados[dados[TARGET].isin([c1, c2])]
            pred = superficie_decisao.classifica_par(sub[atributos].values, w, w0, c1, c2)
            acertos = np.sum(pred == sub[TARGET].values)
            print(f"    Acurácia ({nome}): {acertos}/{len(sub)} = {acertos / len(sub) * 100:.2f}%")

    # 6. Plotagem (Diagramas de dispersão - Variabilidade e Superfícies)
    pasta_figuras = os.path.join(PASTA, 'figuras')
    os.makedirs(pasta_figuras, exist_ok=True)

    fig_var, ax_var = plt.subplots(1, 3, figsize=(18, 5))
    fig_test, ax_test = plt.subplots(1, 3, figsize=(18, 5))
    fig_full, ax_full = plt.subplots(1, 3, figsize=(18, 5))

    for i, (c1, c2) in enumerate(PARES):
        df_pair_full = df[df[TARGET].isin([c1, c2])]
        df_pair_test = test_data[test_data[TARGET].isin([c1, c2])]

        # Plot 1: Variabilidade da nuvem (Sem reta)
        superficie_decisao.dispersao(df_pair_full, ax_var[i], c1, c2, atributos, TARGET)
        ax_var[i].set_title(f'Variabilidade: {c1} X {c2}')
        ax_var[i].legend()

        # Plot 2: Apenas Amostras de Teste + Superfície de decisão
        superficie_decisao.plot_reta(df_pair_test, ax_test[i], c1, c2, prototipos,
                                     atributos, TARGET, f'Teste: {c1} X {c2}')
        # Plot 3: Dataset Completo + Superfície de decisão
        superficie_decisao.plot_reta(df_pair_full, ax_full[i], c1, c2, prototipos,
                                     atributos, TARGET, f'Dataset Completo: {c1} X {c2}')

    for fig, nome in [(fig_var, 'variabilidade'), (fig_test, 'superficie_teste'), (fig_full, 'superficie_completo')]:
        fig.tight_layout()
        fig.savefig(os.path.join(pasta_figuras, f'{nome}.png'), dpi=150)

    plt.show()


if __name__ == '__main__':
    main()
