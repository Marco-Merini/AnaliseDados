import random
from datetime import datetime, timedelta
import pandas as pd
import psycopg2

# ETAPA DE EXTRACTION (GERAÇÃO DOS DADOS BRUTOS)
def extract_data():
    categorias = ['Eletrônicos', 'Vestuário', 'Alimentos', 'Brinquedos']
    regioes = ['Sul', 'Sudeste', 'Norte', 'Centro-Oeste']
    canais = ['Loja Física', 'Online', 'Telefone']

    dados_brutos = []
    data_base = datetime(2024, 1, 1)

    for i in range(500):
        # Geração dos dados brutos
        data_venda = data_base + timedelta(days=random.randint(0, 120))
        mes = data_venda.month
        quantidade = random.randint(5, 20) if mes <= 3 else random.randint(1, 10)
        valor_unitario = round(random.uniform(20.0, 300.0), 2)

        dados_brutos.append({
            'id_venda': i + 1,
            'data_venda': data_venda,
            'id_produto': random.randint(100, 199),
            'categoria': random.choice(categorias),
            'regiao': random.choice(regioes),
            'quantidade': quantidade,
            'valor_unitario': valor_unitario,
            'canal': random.choice(canais)
        })

    return dados_brutos

# ETAPA DE TRANSFORMATION
def transform_data(dados_brutos):
    dados_transformados = []

    for registro in dados_brutos:
        # Cálculo do valor total (transformação)
        valor_total = round(registro['quantidade'] * registro['valor_unitario'], 2)

        # Formatação da data (transformação)
        data_formatada = registro['data_venda'].strftime('%Y-%m-%d')

        # Criar registro transformado
        dados_transformados.append({
            'id_venda': registro['id_venda'],
            'data_venda': data_formatada,
            'id_produto': registro['id_produto'],
            'categoria': registro['categoria'],
            'regiao': registro['regiao'],
            'quantidade': registro['quantidade'],
            'valor_unitario': registro['valor_unitario'],
            'valor_total': valor_total,
            'canal': registro['canal']
        })

    return dados_transformados

# ETAPA DE LOAD
def load_data(dados_transformados):
    df = pd.DataFrame(dados_transformados)
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="vendas",
            user="postgres",
            password="123456",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO vendas
                (id_venda, data_venda, id_produto, categoria, regiao, quantidade, valor_unitario, valor_total, canal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))

        conn.commit()
        print("Dados carregados com sucesso no PostgreSQL!")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
    finally:
        if conn:
            conn.close()

# EXPORTAÇÃO PARA CSV
def export_to_csv(dados_transformados, caminho_csv="vendas_exportadas.csv"):
    df = pd.DataFrame(dados_transformados)
    df.to_csv(caminho_csv, index=False, encoding='utf-8')
    print(f"Arquivo CSV gerado com sucesso em: {caminho_csv}")

# Execução do pipeline ETL
if __name__ == "__main__":
    print("Iniciando processo ETL...")

    print("Extração de dados...")
    dados_brutos = extract_data()

    print("Transformação de dados...")
    dados_transformados = transform_data(dados_brutos)

    print("Carregamento de dados...")
    load_data(dados_transformados)

    print("Exportando para CSV...")
    export_to_csv(dados_transformados)

    print("Processo ETL concluído!")
