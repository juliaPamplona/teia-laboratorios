import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
df = pd.read_excel('Iris data.xls')
atributos_4d = df.columns[:4].tolist() # 4 atributos definidos no dataset
atributos_2d = df.columns[:2].tolist() # 2 primeiros atributos (Sepal length, Sepal width)
target = 'Species'
classes = df[target].unique() # setosa, versicolor, virginica

# 2. Divisão estratificada da amostra (70% Treinamento / 30% Teste)
# A divisão é feita dentro de cada classe (35 treino / 15 teste), preservando a
# proporção de 1/3 por classe nos dois conjuntos.
np.random.seed(42)
train_idx, test_idx = [], []
for c in classes:
    idx_c = np.random.permutation(np.where(df[target] == c)[0])
    n_train = int(round(0.7 * len(idx_c)))
    train_idx.extend(idx_c[:n_train])
    test_idx.extend(idx_c[n_train:])

train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]
X_test_4d, y_test = test_data[atributos_4d].values, test_data[target].values

print(f"Treinamento: {len(train_data)} amostras | Teste: {len(test_data)} amostras")
print("Amostras de teste por classe:", test_data[target].value_counts().to_dict())

# 3. Cálculo dos protótipos (vetor média) para cada classe no conjunto de treino
#    m_j = (1/N_j) * soma(x), x pertencente a w_j
means_4d = {c: train_data[train_data[target] == c][atributos_4d].mean().values for c in classes}
means_2d = {c: train_data[train_data[target] == c][atributos_2d].mean().values for c in classes}

print("\nVetores protótipos (4 atributos):")
for c in classes:
    print(f"  m({c}) = {np.round(means_4d[c], 3)}")

# i) Classificador de distância mínima (Euclidiana)
#    D(x, m_j) = sqrt((x - m_j)^t (x - m_j))
def dist_euclidiana(x, m):
    return np.sqrt(np.sum((x - m)**2))

# ii) Classificador de máximo da função de decisão
#     d_j(x) = x^t m_j - 1/2 m_j^t m_j
# Obs.: menor D(x, m_j) equivale a maior d_j(x), pois
# D^2 = x^t x - 2 d_j(x), e x^t x é igual para todas as classes.
# Logo, os dois classificadores devem produzir exatamente as mesmas predições.
def func_decisao(x, m):
    return np.dot(x, m) - 0.5 * np.dot(m, m)

print("\nFunções de decisão (4 atributos):")
for c in classes:
    m = means_4d[c]
    termos = " + ".join(f"{m[k]:.3f}*x{k+1}" for k in range(len(m)))
    print(f"  d_{c}(x) = {termos} - {0.5 * np.dot(m, m):.3f}")

# 4. Avaliação no conjunto de teste (Cenário com 4 atributos)
pred_dist_min, pred_max_dec = [], []

for x in X_test_4d:
    distancias = {c: dist_euclidiana(x, means_4d[c]) for c in classes}
    pred_dist_min.append(min(distancias, key=distancias.get)) # Classe com menor distância

    decisoes = {c: func_decisao(x, means_4d[c]) for c in classes}
    pred_max_dec.append(max(decisoes, key=decisoes.get))      # Classe com maior valor numérico

acc_dist_min = np.mean(np.array(pred_dist_min) == y_test)
acc_max_dec = np.mean(np.array(pred_max_dec) == y_test)

print(f"\nAcurácia Distância Mínima: {acc_dist_min * 100:.2f}%")
print(f"Acurácia Máximo Função Decisão: {acc_max_dec * 100:.2f}%")
print("Predições idênticas nos dois classificadores:", pred_dist_min == pred_max_dec)

def matriz_confusao(y_real, y_pred):
    mc = pd.DataFrame(0, index=classes, columns=classes)
    for r, p in zip(y_real, y_pred):
        mc.loc[r, p] += 1
    return mc

print("\nMatriz de confusão (linhas = classe real, colunas = classe predita):")
print(matriz_confusao(y_test, pred_dist_min))

# iii) Superfície de decisão d_ij(x) = (m_i - m_j)^t x - 1/2 (m_i - m_j)^t (m_i + m_j) = 0
def coef_superficie(mi, mj):
    w = mi - mj
    w0 = -0.5 * (np.dot(mi, mi) - np.dot(mj, mj))
    return w, w0

# Regra de decisão: d_ij(x) > 0 -> w_i ; d_ij(x) < 0 -> w_j
def classifica_par(X, w, w0, ci, cj):
    return np.where(X @ w + w0 > 0, ci, cj)

pares = [('setosa', 'versicolor'), ('setosa', 'virginica'), ('versicolor', 'virginica')]

print("\nSuperfícies de decisão (4 atributos - hiperplanos):")
for c1, c2 in pares:
    w, w0 = coef_superficie(means_4d[c1], means_4d[c2])
    print(f"  {c1} x {c2}: {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w[2]:.2f}*x3 + {w[3]:.2f}*x4 + {w0:.2f} = 0")

# Com os atributos da sépala, versicolor e virginica se sobrepõem bastante,
# por isso a reta desse par tende a errar mais que as retas que envolvem setosa.
print("\nSuperfícies de decisão (2 primeiros atributos - retas) e regra d_ij(x):")
for c1, c2 in pares:
    w, w0 = coef_superficie(means_2d[c1], means_2d[c2])
    print(f"  {c1} x {c2}: d_ij(x) = {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w0:.2f} = 0")
    for nome, dados in [('teste', test_data), ('dataset completo', df)]:
        sub = dados[dados[target].isin([c1, c2])]
        pred = classifica_par(sub[atributos_2d].values, w, w0, c1, c2)
        acertos = np.sum(pred == sub[target].values)
        print(f"    Acurácia ({nome}): {acertos}/{len(sub)} = {acertos / len(sub) * 100:.2f}%")

# 5. Plotagem (Diagramas de dispersão - Variabilidade e Superfícies)
os.makedirs('figuras', exist_ok=True)

def dispersao(data, ax, c1, c2):
    for c in [c1, c2]:
        sub = data[data[target] == c]
        ax.scatter(sub[atributos_2d[0]], sub[atributos_2d[1]], label=c, edgecolors='k')
    ax.set_xlabel(f'x1 = {atributos_2d[0]} (cm)')
    ax.set_ylabel(f'x2 = {atributos_2d[1]} (cm)')

def plot_reta(data, ax, c1, c2, title):
    dispersao(data, ax, c1, c2)
    mi, mj = means_2d[c1], means_2d[c2]

    # Protótipos e segmento que os liga: a reta é o bissetor perpendicular desse segmento
    ax.plot([mi[0], mj[0]], [mi[1], mj[1]], 'k:', label='Segmento $m_i m_j$')
    ax.scatter(*mi, marker='X', s=200, c='C0', edgecolors='k', label=f'$m$ {c1}')
    ax.scatter(*mj, marker='X', s=200, c='C1', edgecolors='k', label=f'$m$ {c2}')

    # Reta de decisão (2D), limitada à região da nuvem de pontos
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    w, w0 = coef_superficie(mi, mj)
    x_vals = np.array(xlim)
    if w[1] != 0:
        y_vals = (-w[0] * x_vals - w0) / w[1]
        ax.plot(x_vals, y_vals, '--', color='red', label='Superfície Decisão')
    else:
        ax.axvline(x=-w0/w[0], color='red', linestyle='--', label='Superfície Decisão')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal') # mesma escala nos eixos para evidenciar a perpendicularidade

    ax.set_title(title)
    ax.legend(fontsize=8)

fig_var, ax_var = plt.subplots(1, 3, figsize=(18, 5))
fig_test, ax_test = plt.subplots(1, 3, figsize=(18, 5))
fig_full, ax_full = plt.subplots(1, 3, figsize=(18, 5))

for i, (c1, c2) in enumerate(pares):
    df_pair_full = df[df[target].isin([c1, c2])]
    df_pair_test = test_data[test_data[target].isin([c1, c2])]

    # Plot 1: Variabilidade da nuvem (Sem reta)
    dispersao(df_pair_full, ax_var[i], c1, c2)
    ax_var[i].set_title(f'Variabilidade: {c1} X {c2}')
    ax_var[i].legend()

    # Plot 2: Apenas Amostras de Teste + Superfície 2D
    plot_reta(df_pair_test, ax_test[i], c1, c2, f'Teste: {c1} X {c2}')
    # Plot 3: Dataset Completo + Superfície 2D
    plot_reta(df_pair_full, ax_full[i], c1, c2, f'Dataset Completo: {c1} X {c2}')

for fig, nome in [(fig_var, 'variabilidade'), (fig_test, 'superficie_teste'), (fig_full, 'superficie_completo')]:
    fig.tight_layout()
    fig.savefig(os.path.join('figuras', f'{nome}.png'), dpi=150)

plt.show()
