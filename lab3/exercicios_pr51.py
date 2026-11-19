# -*- coding: utf-8 -*-
r"""Exercícios resolvidos da aula PR51 — Métricas de qualidade

Como executar (a partir da raiz do repositório):

    python lab3/exercicios_pr51.py

Todas as matrizes seguem a convenção da aula: linhas = predição e
colunas = referência. As métricas vêm de metricas.py.

Exercícios:
- slide 5:  teste RT-PCR para COVID-19;
- slide 10: concordância entre dois radiologistas (3 classes);
- slide 12: rastreamento de câncer de mama (prevalência de 1%);
- slides 15 e 16: desafio obrigatório, classificadores A e B (4 classes);
- slide 23: caso clínico de diabetes;
- slide 24: caso clínico de pneumonia em raio-X.
"""
import numpy as np

import metricas as M


def titulo(texto):
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


def imprimir_binarias(rotulo, vp, fp, fn, vn):
    b = M.metricas_binarias(vp, fp, fn, vn)
    mc = M.matriz_binaria(vp, fp, fn, vn)
    print(f"{rotulo}: VP={vp} FP={fp} FN={fn} VN={vn} (total {vp + fp + fn + vn})")
    print(f"  Acurácia       = {b['acuracia']:.4f}")
    print(f"  Sensibilidade  = {b['sensibilidade']:.4f}")
    print(f"  Especificidade = {b['especificidade']:.4f}")
    print(f"  Precisão (VPP) = {b['precisao']:.4f}")
    print(f"  F1 = {b['F1']:.4f} | F2 = {b['F2']:.4f} | MCC = {b['MCC']:.4f}")
    print(f"  Kappa = {M.kappa(mc):.4f} ({M.interpretar_kappa(M.kappa(mc))})")
    return b


def matriz_com_prevalencia(sens, espec, prevalencia, total):
    """VP, FP, FN, VN esperados para um teste com a sensibilidade e a
    especificidade dadas, aplicado a uma população com outra prevalência."""
    positivos = prevalencia * total
    negativos = total - positivos
    vp = sens * positivos
    fn = positivos - vp
    vn = espec * negativos
    fp = negativos - vn
    return vp, fp, fn, vn


def vpp_por_prevalencia(sens, espec, prevalencia):
    """Teorema de Bayes: VPP = Se·p / (Se·p + (1 - Es)(1 - p))."""
    return sens * prevalencia / (sens * prevalencia + (1 - espec) * (1 - prevalencia))


# ---------------------------------------------------------------------------
titulo("Slide 5 — Teste RT-PCR para COVID-19")
# ---------------------------------------------------------------------------
covid = imprimir_binarias("RT-PCR", 185, 30, 15, 770)
r"""
A acurácia global confere com o slide (95,5%). O teste detecta 92,5% dos
infectados (sensibilidade) e, dos positivos, 86% estão realmente infectados
(VPP). As 15 pessoas infectadas com resultado negativo são o erro mais grave
numa doença transmissível.
"""

# ---------------------------------------------------------------------------
titulo("Slide 10 — Concordância entre radiologistas (Normal, Benigno, Maligno)")
# ---------------------------------------------------------------------------
radiologistas = np.array([
    [88, 5, 2],
    [8, 52, 4],
    [3, 6, 32]
])
r10 = M.metricas_multiclasse(radiologistas)
print(f"m = {radiologistas.sum()} | A_g = {r10['A_g']:.4f} | A_a = {r10['A_a']:.4f}")
print(f"Kappa = {r10['kappa']:.4f} ({M.interpretar_kappa(r10['kappa'])}), "
      f"σ²_K = {r10['var_kappa']:.6f}")
print(f"Tau   = {r10['tau']:.4f}, σ²_τ = {r10['var_tau']:.6f}")
z = r10["kappa"] / np.sqrt(r10["var_kappa"])
print(f"Kappa > 0? Z = K/σ_K = {z:.2f}, p = {M.p_valor_bilateral(z):.2e}")
r"""
Kappa = 0,776 ≈ 0,78, como no slide: concordância substancial. O teste
Z = K/σ_K mostra que essa concordância é muito superior à esperada por
acaso. O Tau (0,79) é um pouco maior que o Kappa porque supõe acerto casual
uniforme (1/3), enquanto as marginais reais concentram casos em "Normal"
(A_a = 0,375 > 1/3).
"""

# ---------------------------------------------------------------------------
titulo("Slide 12 — Rastreamento de câncer de mama (prevalência de 1%)")
# ---------------------------------------------------------------------------
r"""
A tabela do slide é inconsistente: 90 + 495 + 10 + 9.415 = 10.010, e a
coluna "Saudável" soma 495 + 9.415 = 9.910, não 9.900. Mantendo os totais
do enunciado (100 casos de câncer, 9.900 saudáveis, 10.000 no total), o
valor correto é VN = 9.900 - 495 = 9.405, que é o usado aqui.
"""
mama = imprimir_binarias("Mamografia", 90, 495, 10, 9405)
r"""
A acurácia é de 95%, mas só 15,4% das mamografias positivas são de fato
câncer (VPP). Com prevalência de 1%, mesmo uma especificidade de 95% gera
cinco vezes mais falsos positivos que verdadeiros positivos. Um classificador
que dissesse "saudável" para todas teria 99% de acurácia e 0% de
sensibilidade: é por isso que a acurácia engana em doenças raras. O MCC
(0,36) e o F1 (0,26) mostram o problema. O F2 (0,46) é maior que o F1
porque valoriza a sensibilidade de 90%, que é o que importa no rastreamento.
"""

# ---------------------------------------------------------------------------
titulo("Slides 15 e 16 — Desafio obrigatório: classificadores A e B")
# ---------------------------------------------------------------------------
classificacao_A = np.array([
    [140, 20, 0, 0],
    [10, 130, 0, 0],
    [5, 0, 150, 10],
    [15, 10, 0, 120]
])
classificacao_B = np.array([
    [140, 30, 2, 0],
    [10, 110, 5, 0],
    [0, 10, 140, 0],
    [20, 10, 3, 140]
])

resultados_desafio = {}
for nome, mc in [("A", classificacao_A), ("B", classificacao_B)]:
    r = M.metricas_multiclasse(mc)
    resultados_desafio[nome] = r
    print(f"\nClassificação {nome} (m = {mc.sum()})")
    print(f"  A_g = {r['A_g']:.4f} | A_a = {r['A_a']:.4f}")
    print(f"  A_p (produtor) = {np.round(r['A_p'], 4)}")
    print(f"  A_u (usuário)  = {np.round(r['A_u'], 4)}")
    print(f"  Kappa = {r['kappa']:.4f} | σ²_K = {r['var_kappa']:.6f}")
    print(f"  Tau   = {r['tau']:.4f} | σ²_τ = {r['var_tau']:.6f}")

print("\nTeste de significância (α = 5%, Z crítico = ±1,96)")
for coeficiente, funcao in [("Kappa", M.comparar_kappa), ("Tau", M.comparar_tau)]:
    t = funcao(classificacao_A, classificacao_B)
    decisao = "rejeita H0" if t["rejeita_h0"] else "não rejeita H0"
    print(f"  {coeficiente:<5}: Z = {t['z']:.4f} | p-valor = {t['p_valor']:.4f} → {decisao}")
r"""
- Kappa: A = 0,8469 (σ²_K = 0,000295) e B = 0,8065 (σ²_K = 0,000353).
- Tau:   A = 0,8470 (σ²_τ = 0,000296) e B = 0,8065 (σ²_τ = 0,000356).
- Teste com Kappa: Z = 1,585 e p = 0,113. |Z| < 1,96, então não se rejeita
  H0 ao nível de 5%: não há evidência de que os dois modelos diferem.
- O p-valor de 0,113 diz que, se os dois modelos fossem equivalentes, uma
  diferença de Kappa pelo menos desse tamanho apareceria por acaso em cerca
  de 11% das amostras. É pouco provável, mas não o bastante para o nível de
  significância de 5% (só seria significativo com α > 0,113).
- Teste com Tau: Z = 1,588 e p = 0,112. A conclusão é a mesma. O Tau dá
  valores parecidos com o Kappa porque as marginais das referências são
  quase uniformes, então o acerto casual do Kappa (A_a ≈ 0,25) fica próximo
  de 1/C = 0,25.
"""

# ---------------------------------------------------------------------------
titulo("Slide 23 — Caso clínico: triagem de diabetes tipo 2")
# ---------------------------------------------------------------------------
print("1. Matriz de confusão (linhas = predição, colunas = real)")
print("                 Diabetes   Saudável   Total")
print("   Positivo         255        120       375")
print("   Negativo          45       1580      1625")
print("   Total            300       1700      2000\n")
diabetes = imprimir_binarias("Modelo de IA", 255, 120, 45, 1580)

vp5, fp5, fn5, vn5 = matriz_com_prevalencia(
    diabetes["sensibilidade"], diabetes["especificidade"], 0.05, 2000
)
b5 = M.metricas_binarias(vp5, fp5, fn5, vn5)
print(f"\n4. Com prevalência de 5% (2.000 pacientes, mesma sens. e espec.):")
print(f"   VP={vp5:.0f} FP={fp5:.0f} FN={fn5:.0f} VN={vn5:.0f}")
print(f"   Acurácia = {b5['acuracia']:.4f} | VPP = {b5['precisao']:.4f} | "
      f"F1 = {b5['F1']:.4f} | MCC = {b5['MCC']:.4f}")
print(f"   Acurácia de quem diz 'sem diabetes' para todos: {0.95:.4f}")
r"""
2. Na triagem inicial, a métrica prioritária é a SENSIBILIDADE (ou o F2):
   o objetivo é não deixar diabéticos sem diagnóstico. Um falso positivo só
   gera um exame confirmatório barato (glicemia de jejum, HbA1c), enquanto
   um falso negativo deixa a doença evoluir sem tratamento. A sensibilidade
   atual é de 85%.
3. Os 45 falsos negativos (15% dos diabéticos) saem da triagem achando que
   estão saudáveis. O risco clínico é a progressão silenciosa da doença e
   das complicações (retinopatia, nefropatia, neuropatia, doença
   cardiovascular) e o atraso no tratamento. A métrica afetada é a
   sensibilidade (revocação), e com ela o F1 e o F2.
4. Não. Com prevalência de 5%, a acurácia sobe para 92,5% mesmo sem nenhuma
   mudança no modelo, e um classificador trivial que nunca acusa diabetes
   teria 95%. O VPP cai de 68% para 39%. Para comparar populações com
   prevalências diferentes devem-se usar sensibilidade e especificidade
   (que não dependem da prevalência) e o MCC, que resume a matriz inteira e
   é robusto ao desbalanceamento.
5. Não. Kappa = 0,65 é concordância substancial, mas mede a concordância
   ENTRE OS DOIS ENDOCRINOLOGISTAS, não entre o modelo e a referência. Ele
   indica que o padrão-ouro tem alguma variabilidade (os próprios
   especialistas discordam em parte dos casos). Para validar o modelo é
   preciso calcular o Kappa do modelo contra essa referência (ou contra
   cada especialista) e verificar se é comparável ao Kappa entre
   especialistas. Além disso, 100 casos geram uma variância grande para o
   Kappa.
"""

# ---------------------------------------------------------------------------
titulo("Slide 24 — Caso clínico: pneumonia em raio-X")
# ---------------------------------------------------------------------------
avaliadores = {
    "Modelo de IA": (920, 280, 80, 3720),
    "Radiologista A": (940, 150, 60, 3850),
    "Radiologista B": (890, 100, 110, 3900)
}
print("1. Métricas de cada avaliador contra o laudo de referência\n")
metricas_pneumonia = {}
for nome, contagens in avaliadores.items():
    metricas_pneumonia[nome] = imprimir_binarias(nome, *contagens)
    print()

print(f"{'':<16}{'Sens.':>8}{'Espec.':>8}{'Prec.':>8}{'F1':>8}{'MCC':>8}{'Kappa':>8}")
for nome, b in metricas_pneumonia.items():
    k = M.kappa(M.matriz_binaria(b["VP"], b["FP"], b["FN"], b["VN"]))
    print(f"{nome:<16}{b['sensibilidade']:8.4f}{b['especificidade']:8.4f}"
          f"{b['precisao']:8.4f}{b['F1']:8.4f}{b['MCC']:8.4f}{k:8.4f}")

print("\n3. Kappa de cada avaliador contra a referência (a tabela cruzada")
print("   entre avaliadores não foi fornecida) e teste Z entre eles:")
for n1, n2 in [("Modelo de IA", "Radiologista A"), ("Radiologista A", "Radiologista B"),
               ("Modelo de IA", "Radiologista B")]:
    t = M.comparar_kappa(M.matriz_binaria(*avaliadores[n1]), M.matriz_binaria(*avaliadores[n2]))
    decisao = "diferem" if t["rejeita_h0"] else "não diferem"
    print(f"   {n1} × {n2}: Z = {t['z']:+.3f}, p = {t['p_valor']:.4f} → {decisao} a 5%")

print("\n5. Prevalência de 5% (5.000 imagens, mesma sens. e espec.):")
print(f"{'':<16}{'VPP 20%':>9}{'VPP 5%':>9}{'F1 20%':>9}{'F1 5%':>9}{'MCC 20%':>9}{'MCC 5%':>9}")
for nome, b in metricas_pneumonia.items():
    c5 = matriz_com_prevalencia(b["sensibilidade"], b["especificidade"], 0.05, 5000)
    b5 = M.metricas_binarias(*c5)
    print(f"{nome:<16}{b['precisao']:9.4f}{b5['precisao']:9.4f}{b['F1']:9.4f}{b5['F1']:9.4f}"
          f"{b['MCC']:9.4f}{b5['MCC']:9.4f}")
r"""
2. A IA não supera o Radiologista A em nenhuma métrica. Ela supera o
   Radiologista B em SENSIBILIDADE (0,920 contra 0,890): deixa de detectar
   80 pneumonias, contra 110 do Rad. B. Em compensação, tem mais falsos
   positivos (280 contra 100), então perde em especificidade, precisão, F1
   e MCC. A IA está calibrada para "não perder casos", o que é o
   comportamento desejado numa segunda leitura.
3. O enunciado só dá a matriz de cada avaliador contra a referência. Para o
   Kappa ENTRE dois avaliadores seria preciso a tabela cruzada (quantas
   imagens a IA e o Rad. A laudaram igual e diferente), e ela não pode ser
   reconstruída a partir das matrizes individuais. O que dá para calcular é
   o Kappa de cada um contra a referência: IA = 0,79, Rad. A = 0,87 e
   Rad. B = 0,87. Pelo teste Z, a IA concorda significativamente menos com a
   referência que os dois radiologistas, e os dois radiologistas não
   diferem entre si.
4. Um critério razoável para aprovação como segunda leitura é Kappa
   (IA × médico) ≥ 0,61 (concordância substancial, Landis e Koch) e, mais
   rigorosamente, que ele não seja significativamente menor que o Kappa
   entre os próprios radiologistas (teste Z, p > 0,05). Contra a
   referência a IA tem Kappa = 0,79, ou seja, substancial, o que atende ao
   primeiro critério, mas não ao segundo (Z = -6,1 contra o Rad. A).
5. Com prevalência de 5%, o VPP da IA cai de 0,77 para 0,41: a maioria dos
   positivos da IA passa a ser falso alarme, porque os negativos são muito
   mais numerosos e a especificidade de 93% gera muitos FP. O F1 (0,84 →
   0,57) e o MCC (0,80 → 0,59) caem juntos, então os dois registram a
   piora. O MCC é a métrica mais útil: o F1 ignora os VN e muda se a classe
   "positiva" for trocada, enquanto o MCC usa os quatro valores da matriz,
   é simétrico entre as classes e só se aproxima de 1 quando o
   classificador acerta bem as duas (Chicco e Jurman, 2020).
"""
