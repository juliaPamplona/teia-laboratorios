# -*- coding: utf-8 -*-
r"""Métricas de qualidade de classificadores (aula PR51)

Tudo é calculado à mão, só com NumPy, a partir da matriz de confusão.

Convenção da aula (slide 2): LINHAS = classificação (predição) e
COLUNAS = referência (dados reais). Assim, a_ij é o número de exemplos
atribuídos à classe w_i que pertenciam à classe w_j, a_i+ é o total da
linha i, a_+i é o total da coluna i e m é o total de exemplos.

Métricas multiclasse (slides 4 a 14):
- acerto global          A_g  = (1/m) Σ a_ii
- acurácia do produtor   A_pi = a_ii / a_+i   (sensibilidade da classe i)
- acurácia do usuário    A_ui = a_ii / a_i+   (precisão / VPP da classe i)
- acerto casual          A_a  = Σ a_i+ a_+i / m²
- Kappa                  K    = (A_g - A_a) / (1 - A_a)
- variância do Kappa (Congalton e Green, 2009), com
      φ1 = A_g, φ2 = A_a,
      φ3 = (1/m²) Σ a_ii (a_i+ + a_+i),
      φ4 = (1/m³) Σ_i Σ_j a_ij (a_j+ + a_+i)²
- Tau                    τ    = (A_g - 1/C) / (1 - 1/C)
- variância do Tau       σ²_τ = (1/m) A_g (1 - A_g) / (1 - 1/C)²
- teste de significância Z = (k1 - k2) / sqrt(σ1² + σ2²) ~ N(0,1)

Métricas binárias (slides 17 a 22), a partir de VP, FP, FN e VN:
- sensibilidade (revocação) R  = VP / (VP + FN)
- especificidade             E  = VN / (VN + FP)
- precisão                   P  = VP / (VP + FP)
- acurácia                   A  = (VP + VN) / (VP + FP + FN + VN)
- F_b = (1 + b²) P R / (b² P + R)
- MCC = (VP·VN - FP·FN) / sqrt((VP+FP)(VP+FN)(VN+FP)(VN+FN))

Duas observações sobre as fórmulas dos slides:
- no slide 20 o numerador do F_b aparece como (1 + b); a definição usual
  (van Rijsbergen) é (1 + b²). Para b = 1 as duas coincidem, mas para b = 2
  só (1 + b²) dá uma média harmônica ponderada que vale 1 quando P = R = 1;
- no slide 8 o termo (a_j+ + a_+i) de φ4 aparece sem o expoente 2, que
  existe na fórmula original de Congalton e Green (2009).
"""
import math

import numpy as np


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------

def matriz_confusao(y_real, y_pred, classes):
    """Linhas = predição, colunas = referência (convenção da aula)."""
    indice = {c: k for k, c in enumerate(classes)}
    mc = np.zeros((len(classes), len(classes)), dtype=int)
    for real, pred in zip(y_real, y_pred):
        mc[indice[pred], indice[real]] += 1
    return mc


def _dividir(num, den):
    return num / den if den != 0 else float("nan")


# ---------------------------------------------------------------------------
# Métricas multiclasse
# ---------------------------------------------------------------------------

def acerto_global(mc):
    mc = np.asarray(mc, dtype=float)
    return np.trace(mc) / mc.sum()


def acuracia_produtor(mc):
    """A_pi = a_ii / a_+i: acertos da classe i sobre o total real da classe."""
    mc = np.asarray(mc, dtype=float)
    return np.array([_dividir(mc[i, i], mc[:, i].sum()) for i in range(len(mc))])


def acuracia_usuario(mc):
    """A_ui = a_ii / a_i+: acertos da classe i sobre o total predito como i."""
    mc = np.asarray(mc, dtype=float)
    return np.array([_dividir(mc[i, i], mc[i, :].sum()) for i in range(len(mc))])


def acerto_casual(mc):
    mc = np.asarray(mc, dtype=float)
    m = mc.sum()
    return np.sum(mc.sum(axis=1) * mc.sum(axis=0)) / m ** 2


def kappa(mc):
    A_g = acerto_global(mc)
    A_a = acerto_casual(mc)
    return (A_g - A_a) / (1 - A_a)


def variancia_kappa(mc):
    mc = np.asarray(mc, dtype=float)
    m = mc.sum()
    linhas = mc.sum(axis=1)   # a_i+
    colunas = mc.sum(axis=0)  # a_+i

    phi1 = acerto_global(mc)
    phi2 = acerto_casual(mc)
    phi3 = np.sum(np.diag(mc) * (linhas + colunas)) / m ** 2

    # φ4 = (1/m³) Σ_i Σ_j a_ij (a_j+ + a_+i)²
    phi4 = sum(
        mc[i, j] * (linhas[j] + colunas[i]) ** 2
        for i in range(len(mc))
        for j in range(len(mc))
    ) / m ** 3

    termo1 = phi1 * (1 - phi1) / (1 - phi2) ** 2
    termo2 = 2 * (1 - phi1) * (2 * phi1 * phi2 - phi3) / (1 - phi2) ** 3
    termo3 = (1 - phi1) ** 2 * (phi4 - 4 * phi2 ** 2) / (1 - phi2) ** 4

    return (termo1 + termo2 + termo3) / m


def tau(mc):
    C = len(mc)
    return (acerto_global(mc) - 1 / C) / (1 - 1 / C)


def variancia_tau(mc):
    mc = np.asarray(mc, dtype=float)
    C = len(mc)
    m = mc.sum()
    A_g = acerto_global(mc)
    return A_g * (1 - A_g) / (m * (1 - 1 / C) ** 2)


def interpretar_kappa(k):
    """Escala de Landis e Koch (1977), usada no slide 10."""
    if k < 0:
        return "sem concordância"
    if k <= 0.20:
        return "fraca"
    if k <= 0.40:
        return "razoável"
    if k <= 0.60:
        return "moderada"
    if k <= 0.80:
        return "substancial"
    return "quase perfeita"


# ---------------------------------------------------------------------------
# Teste de significância (slides 9 e 14)
# ---------------------------------------------------------------------------

Z_CRITICO_5 = 1.959964  # Z_{1 - α/2} para α = 0,05


def p_valor_bilateral(z):
    """P(|Z| > |z|) para Z ~ N(0,1)."""
    return math.erfc(abs(z) / math.sqrt(2))


def teste_z(coef1, var1, coef2, var2, alpha=0.05):
    """H0: não há diferença entre os coeficientes; H1: há diferença."""
    z_critico = Z_CRITICO_5 if alpha == 0.05 else _quantil_normal(1 - alpha / 2)
    denominador = math.sqrt(var1 + var2)
    z = (coef1 - coef2) / denominador if denominador > 0 else 0.0
    p = p_valor_bilateral(z)
    return {
        "z": z,
        "z_critico": z_critico,
        "p_valor": p,
        "rejeita_h0": abs(z) > z_critico
    }


def _quantil_normal(prob):
    """Inversa da N(0,1) por bisseção (só usada se α ≠ 0,05)."""
    baixo, alto = -10.0, 10.0
    for _ in range(200):
        meio = (baixo + alto) / 2
        if 0.5 * math.erfc(-meio / math.sqrt(2)) < prob:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def comparar_kappa(mc1, mc2, alpha=0.05):
    return teste_z(kappa(mc1), variancia_kappa(mc1),
                   kappa(mc2), variancia_kappa(mc2), alpha)


def comparar_tau(mc1, mc2, alpha=0.05):
    return teste_z(tau(mc1), variancia_tau(mc1),
                   tau(mc2), variancia_tau(mc2), alpha)


# ---------------------------------------------------------------------------
# Problema de duas classes
# ---------------------------------------------------------------------------

def binaria_da_multiclasse(mc, i):
    """Reduz a matriz C×C ao problema de duas classes "classe i × resto".

    Devolve (VP, FP, FN, VN) da classe i.
    """
    mc = np.asarray(mc)
    vp = mc[i, i]
    fp = mc[i, :].sum() - vp   # preditos como i, mas eram de outra classe
    fn = mc[:, i].sum() - vp   # eram da classe i, mas foram preditos como outra
    vn = mc.sum() - vp - fp - fn
    return int(vp), int(fp), int(fn), int(vn)


def matriz_binaria(vp, fp, fn, vn):
    """Matriz 2×2 no formato da aula: linhas = predição (+, -)."""
    return np.array([[vp, fp], [fn, vn]])


def sensibilidade(vp, fp, fn, vn):
    return _dividir(vp, vp + fn)


def especificidade(vp, fp, fn, vn):
    return _dividir(vn, vn + fp)


def precisao(vp, fp, fn, vn):
    return _dividir(vp, vp + fp)


def acuracia(vp, fp, fn, vn):
    return _dividir(vp + vn, vp + fp + fn + vn)


def f_beta(vp, fp, fn, vn, b=1.0):
    P = precisao(vp, fp, fn, vn)
    R = sensibilidade(vp, fp, fn, vn)
    if math.isnan(P) or math.isnan(R) or (b ** 2 * P + R) == 0:
        return float("nan") if (math.isnan(P) or math.isnan(R)) else 0.0
    return (1 + b ** 2) * P * R / (b ** 2 * P + R)


def mcc(vp, fp, fn, vn):
    den = math.sqrt(float(vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    if den == 0:
        return 0.0
    return (vp * vn - fp * fn) / den


def metricas_binarias(vp, fp, fn, vn):
    c = (vp, fp, fn, vn)
    return {
        "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "acuracia": acuracia(*c),
        "sensibilidade": sensibilidade(*c),
        "especificidade": especificidade(*c),
        "precisao": precisao(*c),
        "F1": f_beta(*c, b=1),
        "F2": f_beta(*c, b=2),
        "MCC": mcc(*c)
    }


# ---------------------------------------------------------------------------
# Resumo completo
# ---------------------------------------------------------------------------

def metricas_multiclasse(mc):
    return {
        "A_g": acerto_global(mc),
        "A_p": acuracia_produtor(mc),
        "A_u": acuracia_usuario(mc),
        "A_a": acerto_casual(mc),
        "kappa": kappa(mc),
        "var_kappa": variancia_kappa(mc),
        "tau": tau(mc),
        "var_tau": variancia_tau(mc)
    }


# ---------------------------------------------------------------------------
# Autoteste com os exemplos dos slides
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Slide 5: COVID-19, acurácia global de 95,5%
    covid = matriz_binaria(185, 30, 15, 770)
    assert abs(acerto_global(covid) - 0.955) < 1e-12

    # Slide 10: radiologistas, Kappa = 0,78
    radiologistas = np.array([[88, 5, 2], [8, 52, 4], [3, 6, 32]])
    assert round(kappa(radiologistas), 2) == 0.78

    # Casos extremos: concordância perfeita
    perfeita = np.diag([10, 10, 10])
    assert kappa(perfeita) == 1 and tau(perfeita) == 1
    assert variancia_kappa(perfeita) == 0 and variancia_tau(perfeita) == 0
    assert mcc(10, 0, 0, 10) == 1 and f_beta(10, 0, 0, 10, b=2) == 1
    assert mcc(0, 10, 10, 0) == -1

    # Matriz de confusão: linhas = predição, colunas = referência
    mc = matriz_confusao(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert mc.tolist() == [[1, 0], [1, 1]]
    assert binaria_da_multiclasse(mc, 0) == (1, 0, 1, 1)

    # p-valor: Z = 1,96 → p ≈ 0,05
    assert abs(p_valor_bilateral(1.959964) - 0.05) < 1e-6

    print("Autoteste das métricas: OK")
    print(f"Slide 10 — Kappa dos radiologistas = {kappa(radiologistas):.4f}")
