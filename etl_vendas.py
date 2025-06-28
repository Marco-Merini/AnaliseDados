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

# FUNÇÃO PARA CRIAR A TABELA SE NÃO EXISTIR
def create_table_if_not_exists(cursor):
    """Cria a tabela vendas se ela não existir"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS vendas (
        id_venda INTEGER PRIMARY KEY,
        data_venda DATE,
        id_produto INTEGER,
        categoria VARCHAR(50),
        regiao VARCHAR(50),
        quantidade INTEGER,
        valor_unitario DECIMAL(10,2),
        valor_total DECIMAL(10,2),
        canal VARCHAR(50)
    );
    """
    cursor.execute(create_table_sql)
    print("📋 Tabela 'vendas' verificada/criada com sucesso!")

# ETAPA DE LOAD
def load_data(dados_transformados):
    df = pd.DataFrame(dados_transformados)
    conn = None
    try:
        # CONFIGURE AQUI SUAS CREDENCIAIS DO POSTGRESQL
        conn = psycopg2.connect(
            dbname="vendas",           # Nome do seu banco
            user="postgres",           # Seu usuário (geralmente 'postgres')
            password="1234", # SUBSTITUA pela sua senha
            host="localhost",          # Se PostgreSQL estiver local
            port="5432"               # Porta padrão do PostgreSQL
        )
        cursor = conn.cursor()

        print(f"Conectado ao banco!")
        
        # Criar tabela se não existir
        create_table_if_not_exists(cursor)
        conn.commit()
        
        print(f"Inserindo {len(dados_transformados)} registros...")
        
        # Inserção com contador de progresso
        contador = 0
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO vendas
                (id_venda, data_venda, id_produto, categoria, regiao, quantidade, valor_unitario, valor_total, canal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
            
            contador += 1
            if contador % 100 == 0:  # Mostra progresso a cada 100 registros
                print(f"Inseridos {contador} registros...")

        conn.commit()
        print(f"✅ Todos os {contador} registros foram inseridos com sucesso no PostgreSQL!")
        
        # Verificar quantos registros existem na tabela
        cursor.execute("SELECT COUNT(*) FROM vendas")
        total_registros = cursor.fetchone()[0]
        print(f"📊 Total de registros na tabela vendas: {total_registros}")
        
    except psycopg2.Error as e:
        print(f"❌ Erro de PostgreSQL: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"❌ Erro geral: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("🔌 Conexão fechada.")

# EXPORTAÇÃO PARA CSV
def export_to_csv(dados_transformados, caminho_csv="vendas_exportadas.csv"):
    df = pd.DataFrame(dados_transformados)
    df.to_csv(caminho_csv, index=False, encoding='utf-8')
    print(f"📄 Arquivo CSV gerado com sucesso em: {caminho_csv}")

# Execução do pipeline ETL
if __name__ == "__main__":
    print("🚀 Iniciando processo ETL...")
    print("=" * 50)

    print("📥 1. Extração de dados...")
    dados_brutos = extract_data()
    print(f"   ✅ {len(dados_brutos)} registros extraídos")

    print("🔄 2. Transformação de dados...")
    dados_transformados = transform_data(dados_brutos)
    print(f"   ✅ {len(dados_transformados)} registros transformados")

    print("📤 3. Carregamento de dados no PostgreSQL...")
    load_data(dados_transformados)

    print("📊 4. Exportando para CSV...")
    export_to_csv(dados_transformados)

    print("=" * 50)
    print("✅ Processo ETL concluído!")