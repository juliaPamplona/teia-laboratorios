# Laboratório 01 — Classificador de Distância Mínima

Implementação, sem bibliotecas de aprendizado de máquina, de um sistema de reconhecimento supervisionado de padrões baseado em decisão teórica, aplicado a base de dados Iris.

Disciplina: Tópicos Especiais em Inteligência Artificial — UEPB

Prof. Robson Pequeno de Sousa

## O que o código faz

- Divide a base em 70% treino / 30% teste, estratificado por classe (35/15 amostras por espécie).
- Usa em todos os sistemas apenas a 3ª e a 4ª colunas da base: `Petal length` (x1) e `Petal width` (x2).
- Calcula o vetor protótipo (média) de cada classe no conjunto de treino.
- **(i)** Classificador de distância mínima (distância euclidiana), para as três classes.
- **(ii)** Classificador pelo máximo das funções de decisão `d_j(x) = xᵗm_j − ½m_jᵗm_j`, para as três classes.
- **(iii)** Superfícies de decisão `d_ij(x) = (m_i − m_j)ᵗx − ½(m_i − m_j)ᵗ(m_i + m_j) = 0` para os pares setosa × versicolor, setosa × virginica e versicolor × virginica, que no plano da pétala são retas.
- Gera diagramas de dispersão (variabilidade, retas sobre o teste e retas sobre a base completa) em `figuras/`.

## Como executar

```bash
pip install pandas numpy matplotlib xlrd
python lab01/main.py
```

## Organização do código

| Arquivo | Responsabilidade |
|---|---|
| `distancia_minima.py` | (i) protótipos, distância euclidiana e classificação pela menor distância |
| `funcao_decisao.py` | (ii) funções de decisão `d_j(x)` e classificação pelo maior valor |
| `superficie_decisao.py` | (iii) superfícies de decisão `d_ij(x)`, regra de sinal e gráficos com a reta |
| `main.py` | leitura da base, divisão 70/30, avaliação, matriz de confusão e geração das figuras |

## Resultados (semente 42)

| Modelo | Acurácia no teste |
|---|---|
| Distância mínima | 93,33% |
| Máximo da função de decisão | 93,33% |

Os dois classificadores produzem predições idênticas, como previsto pela teoria.

| Reta de decisão | Teste | Base completa |
|---|---|---|
| setosa × versicolor | 100% | 100% |
| setosa × virginica | 100% | 100% |
| versicolor × virginica | 90% | 94% |
