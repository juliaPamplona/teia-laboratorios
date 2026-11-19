# -*- coding: utf-8 -*-
"""Perceptron com regra delta (aula PR41) — o algoritmo

Base teórica (slides 13 a 16):
- vetores aumentados x = (x_1, ..., x_n, 1) e w = (w_1, ..., w_n, w_0);
- função discriminante g(x) = w^T x; saída +1 (classe ω1) se g(x) > 0 e
  -1 (classe ω2) se g(x) < 0;
- resposta desejada r ∈ {+1, -1};
- erro quadrático total E(w) = 1/2 Σ (r_i - w^T x_i)^2 e EQM = E(w) / n;
- regra delta (gradiente descendente):
      w(k+1) = w(k) + α [r(k) - w^T(k) x(k)] x(k),   0,1 < α < 1,0.

Diferente do perceptron simples, que só corrige os pesos quando erra a
classificação, a regra delta corrige a cada padrão na proporção do erro
contínuo r - w^T x. Por isso o EQM sempre estabiliza num mínimo, mesmo
quando as classes não são linearmente separáveis.

Funções:
- `aumentar`: acrescenta a coluna de 1s (peso limiar w_0);
- `calcular_eqm`: EQM = E(w) / n, com E(w) = 1/2 Σ (r_i - w^T x_i)^2 (slide 15);
- `classificar`: +1 se g(x) > 0 e -1 caso contrário (slide 8);
- `treinar_regra_delta`: apresenta os padrões na ordem, como no exemplo do
  slide 11, e aplica a regra delta a cada padrão. Ao fim de cada época guarda
  o EQM e o número de erros de classificação no treinamento;
- `verificar_separabilidade`: perceptron simples do slide 9;
- `resultado_converge`: veredito de convergência (ver abaixo).

A época 0 do histórico corresponde aos pesos iniciais (w = 0), para que o
gráfico mostre a queda do EQM desde o começo do treinamento.

O algoritmo converge?
"Converge" pode significar duas coisas diferentes, e o código mede as duas:

1. O EQM converge? O gradiente descendente leva o EQM a um mínimo e ele
   para de variar (|ΔEQM| < tolerância). Com α adequado isso acontece em
   todos os casos, separáveis ou não.
2. O classificador separa as classes? Ao fim do treinamento não sobra
   nenhum erro de classificação no conjunto de treinamento.

O veredito "CONVERGE" exige as duas condições.

Atenção: sobrar erro não prova que as classes são não separáveis. A regra
delta minimiza o erro quadrático, não o número de erros de classificação.
Padrões distantes da fronteira também puxam a solução de EQM mínimo, que
pode deixar alguns padrões do lado errado mesmo quando existe uma reta que
separa tudo. Por isso a separabilidade linear (slide 5) é verificada à parte
com o perceptron simples do slide 9, que só termina sem erros quando as
classes são separáveis. Se ele não termina dentro do limite de épocas, as
classes são consideradas não separáveis.
"""
import numpy as np

# Parâmetros do treinamento
ALPHA = 0.2
MAX_EPOCAS = 1000
TOLERANCIA = 1e-8    # |EQM(k) - EQM(k-1)| < TOLERANCIA


def aumentar(X_amostras):
    return np.c_[X_amostras, np.ones(len(X_amostras))]


def calcular_eqm(X_aumentado, respostas, w):
    erros = respostas - X_aumentado @ w
    E = 0.5 * np.sum(erros ** 2)
    return E / len(X_aumentado)


def classificar(X_aumentado, w):
    return np.where(X_aumentado @ w > 0, 1, -1)


def treinar_regra_delta(
    X_amostras,
    respostas,
    alpha=ALPHA,
    max_epocas=MAX_EPOCAS,
    tolerancia=TOLERANCIA
):
    X_aumentado = aumentar(X_amostras)

    # Pesos iniciais arbitrários (slide 16): partimos do vetor nulo
    w = np.zeros(X_aumentado.shape[1])

    historico_eqm = [calcular_eqm(X_aumentado, respostas, w)]
    historico_erros = [int(np.sum(classificar(X_aumentado, w) != respostas))]
    eqm_estabilizou = False

    for epoca in range(1, max_epocas + 1):

        for x, r in zip(X_aumentado, respostas):
            # REGRA DELTA: w(k+1) = w(k) + α [r - w^T x] x
            erro = r - w @ x
            w = w + alpha * erro * x

        historico_eqm.append(calcular_eqm(X_aumentado, respostas, w))
        historico_erros.append(
            int(np.sum(classificar(X_aumentado, w) != respostas))
        )

        if abs(historico_eqm[-2] - historico_eqm[-1]) < tolerancia:
            eqm_estabilizou = True
            break

    return {
        "w": w,
        "epocas": epoca,
        "eqm": np.array(historico_eqm),
        "erros": np.array(historico_erros),
        "eqm_estabilizou": eqm_estabilizou
    }


def verificar_separabilidade(X_amostras, respostas, max_epocas=5000):
    X_aumentado = aumentar(X_amostras)
    w = np.zeros(X_aumentado.shape[1])

    for _ in range(max_epocas):
        houve_correcao = False

        for x, r in zip(X_aumentado, respostas):
            # Perceptron simples: corrige só quando r * w^T x <= 0
            if r * (w @ x) <= 0:
                w = w + r * x
                houve_correcao = True

        if not houve_correcao:
            return True

    return False


def resultado_converge(resultado):
    return resultado["eqm_estabilizou"] and resultado["erros"][-1] == 0
