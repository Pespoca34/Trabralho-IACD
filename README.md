# Trabralho-IACD

Este repositório contém os arquivos de código para um projeto de análise de dados de câncer de fígado(HCC). O objetivo principal é aplicar técnicas de aprendizado de máquina para prever os resultados dos pacientes baseando-se em diversas características clínicas.

## Descrição

O script principal (`__init__.py`) realiza várias operações de pré-processamento nos dados, constrói modelos de classificação usando Random Forest e K-Nearest Neighbors, e avalia esses modelos com base em várias métricas de desempenho.

## Funcionalidades

- Carrega e analisa dados de um conjunto de dados de HCC.
- Realiza limpeza de dados e pré-processamento, incluindo imputação de valores ausentes e normalização.
- Aplica codificação one-hot para variáveis categóricas.
- Divide os dados em conjuntos de treino e teste.
- Treina modelos de Random Forest e K-Nearest Neighbors.
- Avalia os modelos com métricas como precisão, matriz de confusão, ROC AUC, entre outras.
- Gera visualizações de histogramas e matriz de correlação para análise exploratória.

## Instalação

Para executar este projeto, você precisara instalar as seguintes bibliotecas de Python:

```bash
pip install pandas numpy scikit-learn matplotlib 