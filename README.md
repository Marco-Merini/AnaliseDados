**Apresentação dos Resultados dos Modelos**

Os resultados dos modelos de Machine Learning foram apresentados por meio de um dashboard interativo desenvolvido em Python utilizando a biblioteca **Streamlit**. Este dashboard contempla **múltiplas métricas de análise preditiva**, de acordo com as melhores práticas para regressão e classificação.

- **Modelos de Regressão:**  
  Para a tarefa de regressão (predição de valor total), foram exibidas as seguintes métricas:
  - **MSE (Erro Quadrático Médio):** Mede o erro médio entre os valores reais e preditos.
  - **RMSE (Raiz do Erro Quadrático Médio):** Representa o erro em unidades originais (R$).
  - **R² (Coeficiente de Determinação):** Mede o quanto o modelo explica a variabilidade dos dados.

- **Modelos de Classificação:**  
  Para a tarefa de classificação (predição de alta venda), foram utilizadas:
  - **Accuracy (Acurácia):** Percentual de acertos do modelo.
  - **Precision (Precisão):** Proporção de positivos preditos corretamente.
  - **Recall (Revocação):** Proporção de positivos reais identificados.
  - **F1-Score:** Média harmônica entre Precision e Recall.

Além disso, os resultados são acompanhados de **gráficos de barras** para comparação do **R² Score** nos modelos de regressão, e um **Radar Chart** que compara as métricas de classificação de forma multidimensional.  
Essas representações atendem ao requisito de apresentar **ao menos duas métricas de análise preditiva** de forma clara e visual.
