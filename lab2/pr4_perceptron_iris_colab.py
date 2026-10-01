# -*- coding: utf-8 -*-
"""PR4 — Perceptron aplicado à base Iris e às tabelas verdade

Como executar (a partir da raiz do repositório):

    pip install numpy pandas matplotlib xlrd
    python lab2/pr4_perceptron_iris_colab.py

A base "Iris data.xls" deve estar na mesma pasta deste arquivo. As figuras
são salvas em lab2/figuras/ e exibidas ao final da execução.

Esta seção complementa o laboratório PR3. A implementação trata da função discriminante linear e do algoritmo perceptron.

A atividade será executada com:
- 70% das amostras para treinamento;
- 30% para teste;
- vetor de característica completo com 4 atributos para o classificador;
- os dois primeiros atributos somente para os diagramas de dispersão e para a reta de separação 2D;
- nenhum classificador de aprendizado de máquina pronto.

O perceptron simples é um classificador linear para duas classes. A saída é +1 ou -1 e os pesos são corrigidos quando o padrão é classificado incorretamente.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", None)

# Pasta deste arquivo: a base e as figuras ficam relativas a ela,
# então o script funciona de qualquer diretório de execução.
PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_FIGURAS = os.path.join(PASTA, "figuras")
os.makedirs(PASTA_FIGURAS, exist_ok=True)


def salvar_figura(nome_arquivo):
    plt.savefig(
        os.path.join(PASTA_FIGURAS, f"{nome_arquivo}.png"),
        dpi=150,
        bbox_inches="tight"
    )

"""## 0.1 Leitura dos dados da base Iris"""

# Neste trecho usamos o Pandas para ler a base de dados

iris = pd.read_excel(os.path.join(PASTA, "Iris data.xls"), engine="xlrd")

print(iris.shape)
print(iris.columns.tolist())

print(iris.head())

"""## 0.2 Tranformação dos dados no vetor de característica

Cada flor será representada pelo vetor de quatro atributos:

x = [x_1, x_2, x_3, x_4]


Na base Iris:
- `x1` = comprimento da sépala;
- `x2` = largura da sépala;
- `x3` = comprimento da pétala;
- `x4` = largura da pétala.

A coluna da espécie será transformada em rótulos numéricos:
- 0 = Setosa
- 1 = Versicolor
- 2 = Virginica
"""

# Transformando as classes em valores numéricos para facilitar os cálculos

nome_para_classe = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}

y = iris["Species"].map(nome_para_classe).to_numpy(dtype=int)

# Vetor de característica com os quatro atributos
X = iris[
    [
        "Sepal length",
        "Sepal width",
        "Petal length",
        "Petal width"
    ]
].to_numpy(dtype=float)

nomes_classes = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

print("Dimensão de X:", X.shape)
print("Dimensão de y:", y.shape)

print("\nPrimeiras amostras de X:")
print(X[:5])

print("\nPrimeiras classes:")
print(y[:5])

"""## 0.3 Separação 70% treinamento / 30% teste

A divisão é feita **sem `train_test_split`**. Para manter as proporções das três classes, fazemos uma divisão estratificada manualmente: 70% de cada classe ficam no treinamento e 30% no teste.

Não existe conjunto de validação nesta atividade.
"""

# Função que divide o conjunto com base nos parametros estabelecidos

def divisao_treino_teste_estratificada(X, y, percentual_treino=0.70, seed=41):
    rng = np.random.default_rng(seed)

    indices_treino = []
    indices_teste = []

    for classe in np.unique(y):
        indices = np.where(y == classe)[0].copy()
        rng.shuffle(indices)

        n_treino = int(round(len(indices) * percentual_treino))

        indices_treino.extend(indices[:n_treino])
        indices_teste.extend(indices[n_treino:])

    indices_treino = np.array(indices_treino)
    indices_teste = np.array(indices_teste)

    rng.shuffle(indices_treino)
    rng.shuffle(indices_teste)

    return (
        X[indices_treino],
        X[indices_teste],
        y[indices_treino],
        y[indices_teste]
    )

X_treino, X_teste, y_treino, y_teste = divisao_treino_teste_estratificada(
    X, y, percentual_treino=0.70, seed=41
)

print("Treinamento:", len(X_treino), "amostras")
print("Teste:", len(X_teste), "amostras")

print("\nDistribuição no treinamento:")
for c in np.unique(y):
    print(nomes_classes[c], np.sum(y_treino == c))

print("\nDistribuição no teste:")
for c in np.unique(y):
    print(nomes_classes[c], np.sum(y_teste == c))

r"""## 1. Perceptron binário

Usaremos um vetor aumentado:


$$\mathbf{x}_a = [x_1,x_2,\ldots,x_n,1]$$

e um vetor de pesos:

$$\mathbf{w} = [w_1,w_2,\ldots,w_n,w_0]$$

A função discriminante é:

$$g(x)=\mathbf{w}^T\mathbf{x}_a$$

A decisão será:
- `g(x) >= 0` → classe positiva (+1);
- `g(x) < 0` → classe negativa (-1).

Quando houver erro, aplicamos a regra de atualização:

$$\mathbf{w} \leftarrow \mathbf{w}+\eta d\mathbf{x}_a$$

onde `d` é a resposta desejada (+1 ou -1).

E η é a taxa de aprendizagem.

## 2. Implementação do Perceptron
"""

def treinar_perceptron(X_treino_binario, d_treino, taxa_aprendizado=0.01, max_epocas=1000):

    # Adiciona o termo de bias: 1
    X_aumentado = np.c_[X_treino_binario, np.ones(len(X_treino_binario))]

    # Pesos iniciais conforme a ideia do exemplo da aula:
    # vetor inicialmente nulo.
    w = np.zeros(X_aumentado.shape[1])

    erros_por_epoca = []

    for epoca in range(1, max_epocas + 1):
        erros = 0

        for x, d in zip(X_aumentado, d_treino):
            g = np.dot(w, x)

            # Se o padrão não foi corretamente classificado,
            # atualiza os pesos.
            if d * g <= 0:
                w = w + taxa_aprendizado * d * x
                erros += 1

        erros_por_epoca.append(erros)

        # Se não houve nenhum erro, houve convergência.
        if erros == 0:
            return w, epoca, True, erros_por_epoca

    return w, max_epocas, False, erros_por_epoca


def preparar_par(X, y, classe_positiva, classe_negativa):
    mascara = (y == classe_positiva) | (y == classe_negativa)

    X_par = X[mascara]
    d_par = np.where(y[mascara] == classe_positiva, 1, -1)

    return X_par, d_par


def prever_perceptron(X_amostras, w, classe_positiva, classe_negativa):
    X_aumentado = np.c_[X_amostras, np.ones(len(X_amostras))]
    g = X_aumentado @ w

    previsoes = np.where(
        g >= 0,
        classe_positiva,
        classe_negativa
    )

    return previsoes, g

"""## 3. Experimentos com a Iris usando os 4 atributos

Cada par de classes recebe um perceptron binário próprio. O treinamento ocorre somente com as amostras do conjunto de treinamento.

Pares:
- Setosa × Virginica;
- Setosa × Versicolor;
- Virginica × Versicolor.

"""

pares_classes_pr4 = [
    (0, 2),  # Setosa x Virginica
    (0, 1),  # Setosa x Versicolor
    (2, 1)   # Virginica x Versicolor
]

resultados_4d = {}

for classe_pos, classe_neg in pares_classes_pr4:

    Xtr_par, dtr_par = preparar_par(
        X_treino, y_treino,
        classe_pos, classe_neg
    )

    Xte_par = X_teste[
        (y_teste == classe_pos) | (y_teste == classe_neg)
    ]

    yte_par = y_teste[
        (y_teste == classe_pos) | (y_teste == classe_neg)
    ]

    w, epocas, convergiu, historico = treinar_perceptron(
        Xtr_par,
        dtr_par,
        taxa_aprendizado=0.01,
        max_epocas=1000
    )

    y_pred, _ = prever_perceptron(
        Xte_par,
        w,
        classe_pos,
        classe_neg
    )

    acuracia = np.mean(y_pred == yte_par)

    resultados_4d[(classe_pos, classe_neg)] = {
        "w": w,
        "epocas": epocas,
        "convergiu": convergiu,
        "historico": historico,
        "acuracia": acuracia
    }

    print("=" * 60)
    print(f"{nomes_classes[classe_pos]} x {nomes_classes[classe_neg]}")
    print(f"Convergiu: {convergiu}")
    print(f"Épocas: {epocas}")
    print(f"Acurácia no teste: {acuracia * 100:.2f}%")
    print("Vetor de pesos:")
    print(w)
    termos = " + ".join(f"({w[i]:.3f})x{i + 1}" for i in range(4))
    print(f"Superfície de separação: g(x) = {termos} + ({w[4]:.3f}) = 0")
    print()

r"""## 4. Visualização 2D da superfície de separação

Para a visualização solicitada na atividade, executamos o perceptron novamente usando somente os **dois primeiros atributos**.

Isso é necessário porque, com os quatro atributos, a superfície de decisão é um hiperplano em 4 dimensões e não pode ser representada como uma reta em um gráfico 2D.

Para dois atributos:

$$w_1x_1+w_2x_2+w_0=0$$

e, quando `w2` é diferente de zero:

$$x_2=-\frac{w_1x_1+w_0}{w_2}$$

"""

def equacao_reta(a, b, c):
    # Texto da reta a·x1 + b·x2 + c = 0, usado na legenda dos gráficos
    return (
        f"{'−' if a < 0 else ''}{abs(a):.2f}x₁ "
        f"{'−' if b < 0 else '+'} {abs(b):.2f}x₂ "
        f"{'−' if c < 0 else '+'} {abs(c):.2f} = 0"
    )


def plotar_perceptron_2d(
    classe_pos,
    classe_neg,
    X2_treino,
    y_treino,
    X2_plot,
    y_plot,
    titulo,
    nome_arquivo
):
    Xtr_par, dtr_par = preparar_par(
        X2_treino, y_treino,
        classe_pos, classe_neg
    )

    w, epocas, convergiu, historico = treinar_perceptron(
        Xtr_par,
        dtr_par,
        taxa_aprendizado=1.0,
        max_epocas=1000
    )

    mascara = (y_plot == classe_pos) | (y_plot == classe_neg)

    Xp = X2_plot[mascara]
    yp = y_plot[mascara]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        Xp[yp == classe_pos, 0],
        Xp[yp == classe_pos, 1],
        label=nomes_classes[classe_pos],
        alpha=0.75
    )

    plt.scatter(
        Xp[yp == classe_neg, 0],
        Xp[yp == classe_neg, 1],
        label=nomes_classes[classe_neg],
        alpha=0.75
    )

    xmin = Xp[:, 0].min() - 0.3
    xmax = Xp[:, 0].max() + 0.3
    xs = np.linspace(xmin, xmax, 300)

    estilo = "-" if convergiu else "--"
    rotulo = (
        "Superfície de separação"
        if convergiu
        else "Tentativa final (não convergiu)"
    )
    rotulo += "\n" + equacao_reta(w[0], w[1], w[2])

    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + w[2]) / w[1]

        plt.plot(xs, ys, linestyle=estilo, linewidth=2, label=rotulo)

    else:
        x_vertical = -w[2] / w[0]

        plt.axvline(
            x_vertical,
            linestyle=estilo,
            linewidth=2,
            label=rotulo
        )

    plt.ylim(Xp[:, 1].min() - 0.3, Xp[:, 1].max() + 0.3)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title(
        f"{titulo} — {('convergiu' if convergiu else 'não convergiu')}"
    )
    plt.legend()
    plt.grid(True, alpha=0.25)
    salvar_figura(nome_arquivo)

    print("Vetor de pesos 2D:", w)
    print("Épocas:", epocas)
    print("Convergiu:", convergiu)

    return w, epocas, convergiu, historico


# Dados 2D
X2_treino_pr4 = X_treino[:, :2]
X2_teste_pr4 = X_teste[:, :2]
X2_toda_pr4 = X[:, :2]

"""## 4.1. Setosa × Virginica

Primeiro usamos somente as amostras do conjunto de teste.

"""

# Setosa x Virginica — conjunto de teste

w_sv_teste, ep_sv_teste, conv_sv_teste, hist_sv_teste = plotar_perceptron_2d(
    0, 2,
    X2_treino_pr4, y_treino,
    X2_teste_pr4, y_teste,
    "Setosa × Virginica — conjunto de teste",
    "sv_teste"
)

# Setosa x Virginica — toda a base

w_sv_toda, ep_sv_toda, conv_sv_toda, hist_sv_toda = plotar_perceptron_2d(
    0, 2,
    X2_treino_pr4, y_treino,
    X2_toda_pr4, y,
    "Setosa × Virginica — toda a base",
    "sv_toda"
)

"""## 4.2. Setosa × Versicolor"""

# Setosa x Versicolor — conjunto de teste

w_sver_teste, ep_sver_teste, conv_sver_teste, hist_sver_teste = plotar_perceptron_2d(
    0, 1,
    X2_treino_pr4, y_treino,
    X2_teste_pr4, y_teste,
    "Setosa × Versicolor — conjunto de teste",
    "sver_teste"
)

# Setosa x Versicolor — toda a base

w_sver_toda, ep_sver_toda, conv_sver_toda, hist_sver_toda = plotar_perceptron_2d(
    0, 1,
    X2_treino_pr4, y_treino,
    X2_toda_pr4, y,
    "Setosa × Versicolor — toda a base",
    "sver_toda"
)

"""## 4.3. Virginica × Versicolor"""

# Virginica x Versicolor — conjunto de teste

w_vver_teste, ep_vver_teste, conv_vver_teste, hist_vver_teste = plotar_perceptron_2d(
    2, 1,
    X2_treino_pr4, y_treino,
    X2_teste_pr4, y_teste,
    "Virginica × Versicolor — conjunto de teste",
    "vver_teste"
)

# Virginica x Versicolor — toda a base

w_vver_toda, ep_vver_toda, conv_vver_toda, hist_vver_toda = plotar_perceptron_2d(
    2, 1,
    X2_treino_pr4, y_treino,
    X2_toda_pr4, y,
    "Virginica × Versicolor — toda a base",
    "vver_toda"
)

"""## 5. Histórico de erros e convergência

O gráfico abaixo ajuda a visualizar a diferença entre uma execução que converge e uma em que o perceptron continua realizando correções.

"""

def plotar_erros(historico, titulo, nome_arquivo):
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(historico) + 1),
        historico
    )
    plt.xlabel("Época")
    plt.ylabel("Quantidade de correções")
    plt.title(titulo)
    plt.grid(True, alpha=0.25)
    salvar_figura(nome_arquivo)

plotar_erros(
    hist_sv_teste,
    "Erros por época — Setosa × Virginica",
    "erros_sv"
)

plotar_erros(
    hist_sver_teste,
    "Erros por época — Setosa × Versicolor",
    "erros_sver"
)

plotar_erros(
    hist_vver_teste,
    "Erros por época — Virginica × Versicolor",
    "erros_vver"
)

"""# Parte II — Tabelas verdade

O perceptron também será usado para representar as funções lógicas OR, AND e XOR.

Vamos codificar:
- saída lógica 1 → resposta desejada +1;
- saída lógica 0 → resposta desejada -1.

O XOR é o caso clássico em que um perceptron simples não consegue encontrar uma única reta que separe corretamente as quatro combinações.

"""

X_logica = np.array([
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


def executar_tabela_verdade(nome, saidas_logicas):
    # Converte 1 -> +1 e 0 -> -1
    d = np.where(saidas_logicas == 1, 1, -1)

    w, epocas, convergiu, historico = treinar_perceptron(
        X_logica,
        d,
        taxa_aprendizado=1.0,
        max_epocas=1000
    )

    X_aumentado = np.c_[X_logica, np.ones(len(X_logica))]
    g = X_aumentado @ w

    saidas_preditas = np.where(g >= 0, 1, 0)

    tabela = pd.DataFrame({
        "x1": X_logica[:, 0].astype(int),
        "x2": X_logica[:, 1].astype(int),
        "Desejada": saidas_logicas,
        "g(x)": g,
        "Predita": saidas_preditas
    })

    print("=" * 60)
    print(nome)
    print(f"Convergiu: {convergiu}")
    print(f"Épocas: {epocas}")
    print("Pesos:", w)
    print()
    print(tabela)

    return {
        "w": w,
        "epocas": epocas,
        "convergiu": convergiu,
        "historico": historico,
        "tabela": tabela
    }


resultados_logica = {}

for nome, saidas in tabelas_verdade.items():
    resultados_logica[nome] = executar_tabela_verdade(
        nome,
        saidas
    )

"""## 6. Resumo das tabelas verdade

Para OR e AND, o perceptron encontra uma reta de separação e converge.

Para XOR, o perceptron simples não encontra uma única fronteira linear capaz de separar corretamente os quatro padrões. Nesse caso, atingir zero erros em uma época não acontece.

"""

print("===== RESUMO DAS TABELAS VERDADE =====")

for nome, resultado in resultados_logica.items():
    print(
        f"{nome:3s} | "
        f"Convergiu: {str(resultado['convergiu']):5s} | "
        f"Épocas: {resultado['epocas']}"
    )

r"""# Conclusão da implementação

O perceptron simples foi implementado sem biblioteca de aprendizado de máquina. O treinamento utiliza um vetor aumentado, pesos inicialmente nulos, função discriminante linear e correções iterativas.

Na base Iris, cada par de classes possui um perceptron binário. Para os gráficos, uma segunda execução usa apenas os dois primeiros atributos, permitindo representar a superfície de decisão como uma reta.

A pergunta de convergência deve ser respondida com base no experimento realizado, informando também a representação dos atributos, a divisão 70/30, a semente e o limite de épocas.

## Respostas às perguntas da atividade

**Virginica × Versicolor — o algoritmo converge?** Não. Tanto com os 4 atributos quanto com os 2 primeiros, o perceptron atingiu o limite de 1000 épocas sem nenhuma época livre de correções, o que indica que as duas classes não são linearmente separáveis: não foi encontrado vetor peso com wᵗx > 0 para x ∈ ω₁ e wᵗx < 0 para x ∈ ω₂. Já Setosa × Virginica e Setosa × Versicolor convergiram.

**XOR — o algoritmo converge?** Não. Nenhuma reta separa as saídas 1 das saídas 0, então o perceptron simples não converge. OR e AND convergiram.

# 7 Comparação entre os métodos Distância Mínima (PR3) e Perceptron (PR4)

PR3 (distância mínima): calcula a média de cada classe (protótipo) e classifica cada amostra pela classe cujo protótipo está mais próximo. É analítico, determinístico, multiclasse (3 classes de uma vez) e não tem hiperparâmetros. A fronteira entre duas classes é a mediatriz entre os protótipos, então não se adapta à forma da nuvem de dados.

PR4 (Perceptron): aprende um vetor de pesos w de forma iterativa, corrigindo-o a cada erro (w ← w + η·d·xₐ). É binário, então usa um perceptron por par de classes. A fronteira é qualquer hiperplano que separe o treino, e depende da ordem das amostras e de η.

## Diferenças principais

Obtenção da fronteira: cálculo direto (PR3) × tentativa e erro (PR4).
Convergência: o PR3 sempre produz resposta; o PR4 só converge se as classes forem linearmente separáveis.
Flexibilidade: o PR3 usa só as médias e ignora a dispersão; o PR4 usa os próprios erros e pode se ajustar melhor aos dados.

Na Iris: Setosa é separável das outras duas, então ambos devem ir bem. Em Versicolor × Virginica há sobreposição: a distância mínima comete alguns erros, e o perceptron pode não convergir (principalmente nos 2 atributos da sépala).

XOR: só aparece no PR4. OR e AND convergem, XOR não, por não ser linearmente separável.

Relação: ambos são classificadores lineares. A distância mínima é um caso particular com pesos fixados pelos protótipos, e o perceptron generaliza isso ao buscar os pesos por correção de erros.

## 7.1 Calculo dos prototipos

O protótipo de cada classe será o vetor médio das amostras de treinamento pertencentes à classe.

$$ m_j = {\frac{1}{Nj}}{\sum_{x∈w_j} x_j}$$

Cada protótipo possui os **quatro atributos**.
"""

def calcular_prototipos(X_treino, y_treino):
    prototipos = {}

    for classe in np.unique(y_treino):
        amostras_classe = X_treino[y_treino == classe]

        prototipos[classe] = np.mean(
            amostras_classe,
            axis=0
        )

        print(
            f"{nomes_classes[classe]}: "
            f"{len(amostras_classe)} amostras utilizadas"
        )

    return prototipos

prototipos = calcular_prototipos(X_treino, y_treino)

print("\nProtótipos calculados:")
for classe, vetor in prototipos.items():
    print(
        f"{nomes_classes[classe]:10s}: "
        f"{vetor[0]:.1f}, "
        f"{vetor[1]:.1f}, "
        f"{vetor[2]:.1f}, "
        f"{vetor[3]:.1f}"
    )

"""## 7.2 Calculo da distancia mínima

Devido a equivalencia entre o método da distancia mínima e do máximo  das funções de decisão, a comparação será realizada apenas com o primeiro método.
"""

def distancia_euclidiana(x, prototipo):
    return np.sqrt(np.sum((x - prototipo) ** 2))  # Fórmula da distância euclidiana


def classificar_distancia_minima(x, prototipos):
    distancias = {
        classe: distancia_euclidiana(x, prototipo)
        for classe, prototipo in prototipos.items()
    }

    # Classifica de acordo com a menor distância
    classe_predita = min(distancias, key=distancias.get)

    # Retorna a classe prevista e as distâncias calculadas
    return classe_predita, distancias


def prever_distancia_minima(X_amostras, prototipos):
    previsoes = []

    for x in X_amostras:
        classe, _ = classificar_distancia_minima(x, prototipos)
        previsoes.append(classe)

    return np.array(previsoes)


y_pred_dist = prever_distancia_minima(X_teste, prototipos)

# Mostra a classe real e a classe prevista
print("Primeiras 10 previsões:")
for real, pred in zip(y_teste[:10], y_pred_dist[:10]):
    print(
        f"Real: {nomes_classes[real]:10s} | "
        f"Prevista: {nomes_classes[pred]}"
    )

def prever_distancia_minima_par(
    X_amostras,
    prototipos,
    classe_a,
    classe_b
):
    # Utiliza somente os dois protótipos do par analisado
    prototipos_par = {
        classe_a: prototipos[classe_a],
        classe_b: prototipos[classe_b]
    }

    previsoes = []

    for x in X_amostras:

        classe, _ = classificar_distancia_minima(
            x,
            prototipos_par
        )

        previsoes.append(classe)

    return np.array(previsoes)

"""## 7.3 Comparação entre os modelos

Por fim temos a comparação entre o modelo classico da distancia mínima e o perceptron visto na ultima aula.
"""

def coeficientes_reta_decisao(mi, mj):

    a = 2 * (mj[0] - mi[0])
    b = 2 * (mj[1] - mi[1])
    c = (mi[0]**2 + mi[1]**2) - (mj[0]**2 + mj[1]**2)

    return a, b, c

def equacao_reta_texto(a, b, c):
    eq_str = ""
    if a != 0:
        eq_str += f"{a:.2f}x" if a != 1 else "x"
    if b != 0:
        if eq_str and b > 0: eq_str += " + "
        elif b < 0: eq_str += " - " if eq_str else "-"
        eq_str += f"{abs(b):.2f}y" if abs(b) != 1 else "y"
    if c != 0:
        if eq_str and c > 0: eq_str += " + "
        elif c < 0: eq_str += " - " if eq_str else "-"
        eq_str += f"{abs(c):.2f}"

    if not eq_str: # All coefficients are zero (shouldn't happen for a line)
        return "0 = 0"

    return f"{eq_str} = 0"

# ============================================================
# COMPARAÇÃO: MÉTODO DO CASAMENTO × PERCEPTRON
# ============================================================

resultados_comparacao = []

for classe_a, classe_b in pares_classes_pr4:

    # --------------------------------------------------------
    # Seleciona somente o par de classes
    # --------------------------------------------------------

    mascara_treino = (
        (y_treino == classe_a) |
        (y_treino == classe_b)
    )

    mascara_teste = (
        (y_teste == classe_a) |
        (y_teste == classe_b)
    )

    Xtr = X_treino[mascara_treino]
    ytr = y_treino[mascara_treino]

    Xte = X_teste[mascara_teste]
    yte = y_teste[mascara_teste]


    # ========================================================
    # MÉTODO DO CASAMENTO
    # ========================================================

    y_pred_casamento = prever_distancia_minima_par(
        Xte,
        prototipos,
        classe_a,
        classe_b
    )

    acuracia_casamento = np.mean(
        y_pred_casamento == yte
    )


    # ========================================================
    # PERCEPTRON
    # ========================================================

    # Classe A → +1
    # Classe B → -1

    d_treino = np.where(
        ytr == classe_a,
        1,
        -1
    )

    w, epocas, convergiu, historico = treinar_perceptron(
        Xtr,
        d_treino,
        taxa_aprendizado=1.0,
        max_epocas=1000
    )

    y_pred_perceptron, _ = prever_perceptron(
        Xte,
        w,
        classe_a,
        classe_b
    )

    acuracia_perceptron = np.mean(
        y_pred_perceptron == yte
    )


    # ========================================================
    # ARMAZENAR RESULTADOS
    # ========================================================

    resultados_comparacao.append({

        "Par":
            f"{nomes_classes[classe_a]} × "
            f"{nomes_classes[classe_b]}",

        "Casamento":
            acuracia_casamento * 100,

        "Perceptron":
            acuracia_perceptron * 100,

        "Épocas":
            epocas,

        "Convergiu":
            convergiu
    })


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

comparacao = pd.DataFrame(
    resultados_comparacao
)

comparacao["Casamento"] = comparacao["Casamento"].map(
    lambda x: f"{x:.2f}%"
)

comparacao["Perceptron"] = comparacao["Perceptron"].map(
    lambda x: f"{x:.2f}%"
)

print(comparacao)

"""## 7.4 Comparação gráfica entre os métodos apresentados

Para melhor visualização dos resultados abaixo os dados estão sendo comparados e plotados em um gráfico de dispersão.
"""

def comparar_graficamente(
    classe_a,
    classe_b,
    X2_treino,
    y_treino,
    X2_teste,
    y_teste,
    prototipos_2d,
    nome_arquivo
):

# Seleção das classes

    mascara_treino = (
        (y_treino == classe_a) |
        (y_treino == classe_b)
    )

    mascara_teste = (
        (y_teste == classe_a) |
        (y_teste == classe_b)
    )

    Xtr = X2_treino[mascara_treino]
    ytr = y_treino[mascara_treino]

    Xte = X2_teste[mascara_teste]
    yte = y_teste[mascara_teste]


# Menor distancia

    mi = prototipos_2d[classe_a]
    mj = prototipos_2d[classe_b]

    a_c, b_c, c_c = coeficientes_reta_decisao(mi, mj)


# Perceptron

    # classe_a -> +1
    # classe_b -> -1

    d_treino = np.where(
        ytr == classe_a,
        1,
        -1
    )

    w, epocas, convergiu, historico = treinar_perceptron(
        Xtr,
        d_treino,
        taxa_aprendizado=1.0,
        max_epocas=1000
    )


    xmin = Xte[:, 0].min() - 0.3
    xmax = Xte[:, 0].max() + 0.3

    xs = np.linspace(xmin, xmax, 300)

    plt.figure(figsize=(9, 6))


    plt.scatter(
        Xte[yte == classe_a, 0],
        Xte[yte == classe_a, 1],
        label=nomes_classes[classe_a],
        alpha=0.75
    )


    plt.scatter(
        Xte[yte == classe_b, 0],
        Xte[yte == classe_b, 1],
        label=nomes_classes[classe_b],
        alpha=0.75
    )


    if abs(b_c) > 1e-12:

        ys_casamento = -(a_c * xs + c_c) / b_c

        plt.plot(
            xs,
            ys_casamento,
            linewidth=2,
            label=f"Reta — Método da menor distancia\n{equacao_reta(a_c, b_c, c_c)}"
        )

    else:

        x_vertical = -c_c / a_c

        plt.axvline(
            x_vertical,
            linewidth=2,
            label=f"Reta — Método da menor distancia\n{equacao_reta(a_c, b_c, c_c)}"
        )

    if abs(w[1]) > 1e-12:

        ys_perceptron = -(w[0] * xs + w[2]) / w[1]

        estilo = "-" if convergiu else "--"

        plt.plot(
            xs,
            ys_perceptron,
            linestyle=estilo,
            linewidth=2,
            label=f"Reta — Perceptron\n{equacao_reta(w[0], w[1], w[2])}"
        )

    else:

        x_vertical = -w[2] / w[0]

        estilo = "-" if convergiu else "--"

        plt.axvline(
            x_vertical,
            linestyle=estilo,
            linewidth=2,
            label=f"Reta — Perceptron\n{equacao_reta(w[0], w[1], w[2])}"
        )


    plt.scatter(
        mi[0],
        mi[1],
        marker="X",
        s=180,
        label=f"Protótipo {nomes_classes[classe_a]}"
    )

    plt.scatter(
        mj[0],
        mj[1],
        marker="X",
        s=180,
        label=f"Protótipo {nomes_classes[classe_b]}"
    )


    plt.ylim(Xte[:, 1].min() - 0.3, Xte[:, 1].max() + 0.3)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.title(
        f"{nomes_classes[classe_a]} × "
        f"{nomes_classes[classe_b]}"
    )

    # Legenda fora da área do gráfico para não cobrir os pontos
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.25)

    salvar_figura(nome_arquivo)


    print("==========================================")
    print(
        f"{nomes_classes[classe_a]} × "
        f"{nomes_classes[classe_b]}"
    )
    print("==========================================")

    print("\nMétodo do casamento:")
    print(
        "Equação:",
        equacao_reta_texto(a_c, b_c, c_c)
    )

    print("\nPerceptron:")
    print("Pesos:", w)
    print("Épocas:", epocas)
    print("Convergiu:", convergiu)

comparar_graficamente(
    0,
    1,
    X2_treino_pr4,
    y_treino,
    X2_teste_pr4,
    y_teste,
    prototipos,
    "comp_sver"
)

comparar_graficamente(
    0,
    2,
    X2_treino_pr4,
    y_treino,
    X2_teste_pr4,
    y_teste,
    prototipos,
    "comp_sv"
)

comparar_graficamente(
    2,
    1,
    X2_treino_pr4,
    y_treino,
    X2_teste_pr4,
    y_teste,
    prototipos,
    "comp_vver"
)

print(f"\nFiguras salvas em: {PASTA_FIGURAS}")

# Exibe todas as figuras geradas
plt.show()
