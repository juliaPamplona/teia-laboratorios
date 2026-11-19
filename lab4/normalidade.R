# Verificação da normalidade multivariada de cada classe da base Iris
#
# Como executar (a partir da raiz do repositório):
#
#     Rscript lab4/normalidade.R
#
# ou abrir no RStudio e executar tudo (Ctrl + Shift + Enter).
#
# - Henze-Zirkler: se p > 0,05, os dados vêm de uma população normal;
# - Mardia: complementa o HZ verificando assimetria e curtose;
# - gráfico Q-Q de qui-quadrado das distâncias de Mahalanobis.
#
# Os resultados são gravados em resultados_normalidade_R.txt e os gráficos
# em figuras/qq_<classe>_R.png, na pasta deste script.

# No Windows a biblioteca do sistema não é gravável: os pacotes vão para a
# biblioteca do usuário (a mesma que o RStudio sugere ao instalar).
lib_usuario <- Sys.getenv("R_LIBS_USER")
dir.create(lib_usuario, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_usuario, .libPaths()))

pacotes <- c("MVN", "readxl")
for (pacote in pacotes) {
  if (!require(pacote, character.only = TRUE)) {
    install.packages(pacote, repos = "https://cloud.r-project.org", type = "binary")
    library(pacote, character.only = TRUE)
  }
}

# Pasta deste script (funciona com Rscript e no RStudio)
argumentos <- commandArgs(trailingOnly = FALSE)
arquivo <- sub("--file=", "", argumentos[grep("--file=", argumentos)])
if (length(arquivo) > 0) {
  pasta <- dirname(normalizePath(arquivo))
} else if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  pasta <- dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  pasta <- getwd()
}

iris_dados <- read_excel(file.path(pasta, "Iris data.xls"))
names(iris_dados) <- trimws(names(iris_dados))
iris_dados$Species <- tolower(trimws(iris_dados$Species))

atributos <- c("Sepal length", "Sepal width", "Petal length", "Petal width")
classes <- c("setosa", "versicolor", "virginica")

# O MVN 6.x renomeou os argumentos (mvnTest -> mvn_test) e os resultados
# (multivariateNormality -> multivariate_normality).
versao_nova <- packageVersion("MVN") >= "6.0"

testar <- function(dados, teste) {
  if (versao_nova) {
    resultado <- mvn(data = dados, mvn_test = teste)
    resultado$multivariate_normality
  } else {
    resultado <- mvn(data = dados, mvnTest = teste)
    resultado$multivariateNormality
  }
}

# A tabela do mvn() arredonda o p-valor para 3 casas (a Setosa aparece com
# p = 0,05). As funções hz() e mardia() do MVN 6 devolvem o valor completo.
p_valores_completos <- function(dados) {
  if (!versao_nova) return(invisible(NULL))
  resultados <- rbind(MVN::hz(dados), MVN::mardia(dados))
  data.frame(
    Teste = resultados$Test,
    Estatistica = format(resultados$Statistic, digits = 8),
    p.valor = format(resultados$p.value, digits = 8)
  )
}

saida <- file.path(pasta, "resultados_normalidade_R.txt")
sink(saida, split = TRUE)

cat("R:", R.version.string, "\n")
cat("MVN:", as.character(packageVersion("MVN")), "\n")

pasta_figuras <- file.path(pasta, "figuras")
dir.create(pasta_figuras, showWarnings = FALSE)

for (classe in classes) {
  dados_classe <- as.data.frame(iris_dados[iris_dados$Species == classe, atributos])

  cat("\n==============================================================\n")
  cat("Classe:", classe, "(n =", nrow(dados_classe), ")\n")
  cat("==============================================================\n")

  cat("\n--- Teste de Henze-Zirkler ---\n")
  print(testar(dados_classe, "hz"), digits = 6)

  cat("\n--- Teste de Mardia (assimetria e curtose) ---\n")
  print(testar(dados_classe, "mardia"), digits = 6)

  cat("\n--- p-valores sem arredondamento ---\n")
  print(p_valores_completos(dados_classe), row.names = FALSE)

  # Gráfico Q-Q de qui-quadrado: D_i² ordenadas contra quantis da χ²(p)
  n <- nrow(dados_classe)
  p <- ncol(dados_classe)
  S <- cov(dados_classe) * (n - 1) / n
  d2 <- sort(mahalanobis(dados_classe, colMeans(dados_classe), S))
  quantis <- qchisq((seq_len(n) - 0.5) / n, df = p)

  png(file.path(pasta_figuras, paste0("qq_", classe, "_R.png")), width = 900, height = 800, res = 150)
  plot(quantis, d2,
       main = paste("Q-Q de qui-quadrado -", classe),
       xlab = expression(paste("Quantis da ", chi^2, "(4)")),
       ylab = "Distância de Mahalanobis ao quadrado",
       pch = 19, col = "steelblue")
  abline(0, 1, col = "firebrick", lwd = 2)
  dev.off()
}

sink()
cat("\nResultados gravados em", saida, "\n")
