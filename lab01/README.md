# Laboratório 01 — Classificador de Distância Mínima

Implementação, sem bibliotecas de aprendizado de máquina, de um sistema de reconhecimento supervisionado de padrões baseado em decisão teórica, aplicado a base de dados Iris.

Disciplina: Tópicos Especiais em Inteligência Artificial — UEPB

Prof. Robson Pequeno de Sousa

## O que o código faz

- Divide a base em 70% treino / 30% teste, estratificado por classe (35/15 amostras por espécie).
- Calcula o vetor protótipo (média) de cada classe no conjunto de treino.
- **(i)** Classificador de distância mínima (distância euclidiana), com os 4 atributos.
- **(ii)** Classificador pelo máximo das funções de decisão `d_j(x) = xᵗm_j − ½m_jᵗm_j`, com os 4 atributos.
- **(iii)** Superfícies de decisão `d_ij(x) = (m_i − m_j)ᵗx − ½(m_i − m_j)ᵗ(m_i + m_j) = 0` para os pares setosa × versicolor, setosa × virginica e versicolor × virginica: hiperplanos com 4 atributos e retas com os 2 primeiros atributos.
- Gera diagramas de dispersão (variabilidade, retas sobre o teste e retas sobre a base completa) em `figuras/`.

## Como executar

```bash
pip install pandas numpy matplotlib xlrd
cd lab01
python lab01.py
```

Execute o script de dentro da pasta `lab01/`, onde estão o arquivo `Iris data.xls` e a pasta `figuras/`.

## Resultados (semente 42)

| Modelo | Acurácia no teste |
|---|---|
| Distância mínima (4 atributos) | 91,11% |
| Máximo da função de decisão (4 atributos) | 91,11% |

Os dois classificadores produzem predições idênticas, como previsto pela teoria.
