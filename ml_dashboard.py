# ml_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Resultados ML", layout="wide")

# Simulação dos dados
regression_results = pd.DataFrame([
    {"Modelo": "Regressão Linear", "MSE": 245678.50, "RMSE": 495.66, "R2": 0.7234, "Performance": "Boa"},
    {"Modelo": "Random Forest Reg", "MSE": 189432.20, "RMSE": 435.23, "R2": 0.7891, "Performance": "Excelente"},
])

classification_results = pd.DataFrame([
    {"Modelo": "Regressão Logística", "Accuracy": 0.8450, "Precision": 0.8123, "Recall": 0.8267, "F1Score": 0.8194, "Performance": "Boa"},
    {"Modelo": "Random Forest Clf", "Accuracy": 0.8950, "Precision": 0.8834, "Recall": 0.8912, "F1Score": 0.8873, "Performance": "Excelente"},
])

radar_data = pd.DataFrame([
    {"Metric": "Accuracy", "Reg. Logística": 0.845, "Random Forest": 0.895},
    {"Metric": "Precision", "Reg. Logística": 0.812, "Random Forest": 0.883},
    {"Metric": "Recall", "Reg. Logística": 0.827, "Random Forest": 0.891},
    {"Metric": "F1-Score", "Reg. Logística": 0.819, "Random Forest": 0.887},
])

performance_comparison = pd.DataFrame([
    {"Modelo": "Linear Reg", "Score": 72.34, "Tipo": "R²"},
    {"Modelo": "RF Reg", "Score": 78.91, "Tipo": "R²"},
    {"Modelo": "Log Reg", "Score": 84.50, "Tipo": "Accuracy"},
    {"Modelo": "RF Clf", "Score": 89.50, "Tipo": "Accuracy"},
])

# Layout principal
st.title("📊 Resultados dos Modelos de Machine Learning")

# Status do pipeline
col1, col2, col3, col4 = st.columns(4)
col1.metric("✔️ Pipeline Status", "Concluído")
col2.metric("🎯 Modelos Treinados", "4 Modelos")
col3.metric("📈 Dados Processados", "500 registros")
col4.metric("🏆 Melhor Performance", "RF: 89.5%")

# Tabs
tab = st.radio("Selecione o tipo de modelo:", ["📈 Regressão", "📊 Classificação", "🔄 Comparação"])

if tab == "📈 Regressão":
    st.header("📈 Resultados dos Modelos de Regressão")
    st.dataframe(regression_results)

    fig = px.bar(
        regression_results,
        x="Modelo",
        y="R2",
        text=regression_results["R2"].apply(lambda x: f"{x:.2%}"),
        title="Comparação do R² Score",
        range_y=[0,1]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretação**
    - MSE (Erro Quadrático Médio): Quanto menor, melhor.
    - RMSE: Erro médio em unidades reais (R$).
    - R²: Quanto mais próximo de 1, mais o modelo explica a variância.
    - 🏆 Melhor: Random Forest Regressor com R² = 78.91%
    """)

elif tab == "📊 Classificação":
    st.header("📊 Resultados dos Modelos de Classificação")
    st.dataframe(classification_results)

    fig = px.line_polar(
        radar_data.melt(id_vars=["Metric"], var_name="Modelo", value_name="Valor"),
        r="Valor",
        theta="Metric",
        color="Modelo",
        line_close=True,
        title="Radar Chart - Métricas de Classificação"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretação**
    - Accuracy: Proporção de acertos.
    - Precision: Acurácia dos positivos previstos.
    - Recall: Capacidade de capturar positivos reais.
    - F1-Score: Média harmônica entre Precision e Recall.
    - 🏆 Melhor: Random Forest Classifier com 89.5% de acurácia.
    """)

else:
    st.header("🔄 Comparação Geral dos Modelos")
    st.dataframe(performance_comparison)

    fig = px.bar(
        performance_comparison,
        x="Modelo",
        y="Score",
        text=performance_comparison["Score"].apply(lambda x: f"{x:.2f}%"),
        title="Performance Geral",
        range_y=[0,100]
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.success("🏆 Melhor Regressão: Random Forest (R² = 78.91%)")
    with col2:
        st.success("🏆 Melhor Classificação: Random Forest (Accuracy = 89.5%)")

    st.info("""
    **Recomendações**
    - Use Random Forest para predição de valores.
    - Use Random Forest para classificação de vendas.
    - RMSE de R$ 435 é aceitável.
    - Divisão de treino/teste 80/20.
    """)
