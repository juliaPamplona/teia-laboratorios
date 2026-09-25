"""(iii) Superfície de decisão entre duas classes.

d_ij(x) = (m_i - m_j)^t x - 1/2 (m_i - m_j)^t (m_i + m_j) = 0

É o bissetor perpendicular ao segmento que liga m_i e m_j: uma reta com 2
atributos e um hiperplano com 4.
"""
import numpy as np


def coef_superficie(mi, mj):
    """Coeficientes de d_ij(x) = w^t x + w0."""
    w = mi - mj
    w0 = -0.5 * (np.dot(mi, mi) - np.dot(mj, mj))
    return w, w0


def classifica_par(X, w, w0, ci, cj):
    """Regra de decisão: d_ij(x) > 0 -> w_i ; d_ij(x) < 0 -> w_j"""
    return np.where(X @ w + w0 > 0, ci, cj)


def dispersao(data, ax, c1, c2, atributos, target):
    """Diagrama de dispersão das classes c1 e c2 nos dois atributos dados."""
    for c in [c1, c2]:
        sub = data[data[target] == c]
        ax.scatter(sub[atributos[0]], sub[atributos[1]], label=c, edgecolors='k')
    ax.set_xlabel(f'x1 = {atributos[0]} (cm)')
    ax.set_ylabel(f'x2 = {atributos[1]} (cm)')


def plot_reta(data, ax, c1, c2, prototipos, atributos, target, title):
    """Dispersão de c1 x c2 com os protótipos e a reta de decisão entre eles."""
    dispersao(data, ax, c1, c2, atributos, target)
    mi, mj = prototipos[c1], prototipos[c2]

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
