import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class PipelineMLVendas:
    def __init__(self):
        self.dados_brutos = None
        self.dados_processados = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelos = {}
        self.preprocessor = None
        
    def etapa_1_extrair_dados_brutos(self):
        """ETAPA 1: EXTRAÇÃO DOS DADOS BRUTOS DO POSTGRESQL"""
        print("🔍 ETAPA 1: EXTRAINDO DADOS BRUTOS DO POSTGRESQL")
        print("=" * 60)
        
        try:
            # Conectar ao PostgreSQL
            conn = psycopg2.connect(
                dbname="vendas",
                user="postgres", 
                password="1234",
                host="localhost",
                port="5432"
            )
            
            # Extrair dados brutos
            query = "SELECT * FROM vendas ORDER BY data_venda"
            self.dados_brutos = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"✅ Dados extraídos: {len(self.dados_brutos)} registros")
            print(f"📊 Período: {self.dados_brutos['data_venda'].min()} a {self.dados_brutos['data_venda'].max()}")
            print(f"🏛️ Colunas: {list(self.dados_brutos.columns)}")
            print()
            
            # Mostrar amostra dos dados brutos
            print("📋 AMOSTRA DOS DADOS BRUTOS:")
            print(self.dados_brutos.head())
            print()
            
            return self.dados_brutos
            
        except Exception as e:
            print(f"❌ Erro na extração: {e}")
            return None
    
    def etapa_2_preprocessamento(self):
        """ETAPA 2: PRÉ-PROCESSAMENTO E FEATURE ENGINEERING"""
        print("🔧 ETAPA 2: PRÉ-PROCESSAMENTO E FEATURE ENGINEERING")
        print("=" * 60)
        
        # Fazer cópia dos dados brutos
        self.dados_processados = self.dados_brutos.copy()
        
        # 1. Conversão de tipos e tratamento de datas
        print("📅 1. Processando datas...")
        self.dados_processados['data_venda'] = pd.to_datetime(self.dados_processados['data_venda'])
        
        # 2. Feature Engineering - extrair informações da data
        print("🛠️ 2. Criando features derivadas...")
        self.dados_processados['ano'] = self.dados_processados['data_venda'].dt.year
        self.dados_processados['mes'] = self.dados_processados['data_venda'].dt.month
        self.dados_processados['dia_semana'] = self.dados_processados['data_venda'].dt.dayofweek
        self.dados_processados['trimestre'] = self.dados_processados['data_venda'].dt.quarter
        
        # 3. Criar variáveis categóricas de alta/baixa venda
        print("📈 3. Criando targets para classificação...")
        mediana_vendas = self.dados_processados['valor_total'].median()
        self.dados_processados['alta_venda'] = (self.dados_processados['valor_total'] > mediana_vendas).astype(int)
        
        # 4. Criar faixas de quantidade
        self.dados_processados['faixa_quantidade'] = pd.cut(
            self.dados_processados['quantidade'], 
            bins=[0, 5, 10, 15, float('inf')], 
            labels=['Baixa', 'Média', 'Alta', 'Muito Alta']
        )
        
        # 5. Análise de estatísticas descritivas
        print("📊 4. Estatísticas dos dados processados:")
        print(f"   Total de registros: {len(self.dados_processados)}")
        print(f"   Valor total médio: R$ {self.dados_processados['valor_total'].mean():.2f}")
        print(f"   Valor total mediano: R$ {self.dados_processados['valor_total'].median():.2f}")
        print(f"   Distribuição de alta venda: {self.dados_processados['alta_venda'].value_counts().to_dict()}")
        print()
        
        # 6. Verificar dados faltantes
        print("🔍 5. Verificando dados faltantes:")
        dados_faltantes = self.dados_processados.isnull().sum()
        if dados_faltantes.sum() == 0:
            print("   ✅ Nenhum dado faltante encontrado!")
        else:
            print(f"   ⚠️ Dados faltantes encontrados:\n{dados_faltantes[dados_faltantes > 0]}")
        print()
        
        # Mostrar amostra dos dados processados
        print("📋 AMOSTRA DOS DADOS PROCESSADOS:")
        print(self.dados_processados[['data_venda', 'categoria', 'valor_total', 'ano', 'mes', 'alta_venda']].head())
        print()
        
        return self.dados_processados
    
    def etapa_3_preparar_features(self):
        """ETAPA 3: PREPARAÇÃO DAS FEATURES PARA MODELAGEM"""
        print("🎯 ETAPA 3: PREPARAÇÃO DAS FEATURES PARA MODELAGEM")
        print("=" * 60)
        
        # Definir features para os modelos
        features_numericas = ['quantidade', 'valor_unitario', 'ano', 'mes', 'dia_semana', 'trimestre']
        features_categoricas = ['categoria', 'regiao', 'canal']
        
        print(f"📊 Features numéricas: {features_numericas}")
        print(f"🏷️ Features categóricas: {features_categoricas}")
        print()
        
        # Preparar dados para REGRESSÃO (predizer valor_total)
        print("📈 Preparando dados para REGRESSÃO (predizer valor_total):")
        X_regressao = self.dados_processados[features_numericas + features_categoricas].copy()
        y_regressao = self.dados_processados['valor_total'].copy()
        
        # Preparar dados para CLASSIFICAÇÃO (predizer alta_venda)
        print("📊 Preparando dados para CLASSIFICAÇÃO (predizer alta_venda):")
        X_classificacao = self.dados_processados[features_numericas + features_categoricas].copy()
        y_classificacao = self.dados_processados['alta_venda'].copy()
        
        # Criar preprocessador para features categóricas e numéricas
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features_numericas),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), features_categoricas)
            ]
        )
        
        print(f"   Formato dos dados: {X_regressao.shape}")
        print(f"   Target regressão - min: {y_regressao.min():.2f}, max: {y_regressao.max():.2f}")
        print(f"   Target classificação - distribuição: {y_classificacao.value_counts().to_dict()}")
        print()
        
        return X_regressao, y_regressao, X_classificacao, y_classificacao
    
    def etapa_4_dividir_treino_teste(self, X_reg, y_reg, X_clf, y_clf):
        """ETAPA 4: DIVISÃO EM TREINO E TESTE"""
        print("🔀 ETAPA 4: DIVISÃO DOS DADOS EM TREINO E TESTE")
        print("=" * 60)
        
        # Divisão para regressão
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Divisão para classificação
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        print("📊 DIVISÃO DOS DADOS:")
        print(f"   📈 REGRESSÃO:")
        print(f"      Treino: {X_train_reg.shape[0]} amostras")
        print(f"      Teste:  {X_test_reg.shape[0]} amostras")
        print(f"   📊 CLASSIFICAÇÃO:")
        print(f"      Treino: {X_train_clf.shape[0]} amostras")
        print(f"      Teste:  {X_test_clf.shape[0]} amostras")
        print(f"   🎯 Proporção treino/teste: 80/20")
        print()
        
        return X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf
    
    def etapa_5_treinar_modelos(self, X_train_reg, y_train_reg, X_train_clf, y_train_clf):
        """ETAPA 5: TREINAMENTO DOS MODELOS"""
        print("🤖 ETAPA 5: TREINAMENTO DOS MODELOS DE MACHINE LEARNING")
        print("=" * 60)
        
        # MODELOS DE REGRESSÃO
        print("📈 Treinando modelos de REGRESSÃO:")
        
        # Modelo 1: Regressão Linear
        print("   🔹 Regressão Linear...")
        pipeline_lr = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
        ])
        pipeline_lr.fit(X_train_reg, y_train_reg)
        self.modelos['regressao_linear'] = pipeline_lr
        
        # Modelo 2: Random Forest Regressor
        print("   🌳 Random Forest Regressor...")
        pipeline_rf_reg = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline_rf_reg.fit(X_train_reg, y_train_reg)
        self.modelos['random_forest_reg'] = pipeline_rf_reg
        
        # MODELOS DE CLASSIFICAÇÃO
        print("📊 Treinando modelos de CLASSIFICAÇÃO:")
        
        # Modelo 3: Regressão Logística
        print("   🔸 Regressão Logística...")
        pipeline_log = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        pipeline_log.fit(X_train_clf, y_train_clf)
        self.modelos['regressao_logistica'] = pipeline_log
        
        # Modelo 4: Random Forest Classifier
        print("   🌲 Random Forest Classifier...")
        pipeline_rf_clf = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline_rf_clf.fit(X_train_clf, y_train_clf)
        self.modelos['random_forest_clf'] = pipeline_rf_clf
        
        print(f"✅ {len(self.modelos)} modelos treinados com sucesso!")
        print()
        
        return self.modelos
    
    def etapa_6_avaliar_modelos(self, X_test_reg, y_test_reg, X_test_clf, y_test_clf):
        """ETAPA 6: AVALIAÇÃO DOS MODELOS"""
        print("📊 ETAPA 6: AVALIAÇÃO DOS MODELOS")
        print("=" * 60)
        
        resultados = {}
        
        # Avaliar modelos de regressão
        print("📈 AVALIAÇÃO DOS MODELOS DE REGRESSÃO:")
        print("-" * 40)
        
        for nome in ['regressao_linear', 'random_forest_reg']:
            modelo = self.modelos[nome]
            y_pred = modelo.predict(X_test_reg)
            
            mse = mean_squared_error(y_test_reg, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_reg, y_pred)
            
            resultados[nome] = {
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
            
            print(f"🔹 {nome.upper()}:")
            print(f"   MSE:  {mse:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R²:   {r2:.4f}")
            print()
        
        # Avaliar modelos de classificação
        print("📊 AVALIAÇÃO DOS MODELOS DE CLASSIFICAÇÃO:")
        print("-" * 40)
        
        for nome in ['regressao_logistica', 'random_forest_clf']:
            modelo = self.modelos[nome]
            y_pred = modelo.predict(X_test_clf)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test_clf, y_pred)
            precision = precision_score(y_test_clf, y_pred)
            recall = recall_score(y_test_clf, y_pred)
            f1 = f1_score(y_test_clf, y_pred)
            
            resultados[nome] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            print(f"🔸 {nome.upper()}:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            print()
        
        return resultados
    
    def etapa_7_salvar_modelos(self):
        """ETAPA 7: SALVAMENTO DOS MODELOS TREINADOS"""
        print("💾 ETAPA 7: SALVAMENTO DOS MODELOS")
        print("=" * 60)
        
        for nome, modelo in self.modelos.items():
            arquivo = f"modelo_{nome}.pkl"
            joblib.dump(modelo, arquivo)
            print(f"✅ Modelo '{nome}' salvo em: {arquivo}")
        
        print()
        print("📁 ARQUIVOS GERADOS:")
        for nome in self.modelos.keys():
            print(f"   📄 modelo_{nome}.pkl")
        print()
    
    def executar_pipeline_completo(self):
        """EXECUÇÃO DO PIPELINE COMPLETO DE ML"""
        print("🚀 PIPELINE COMPLETO DE MACHINE LEARNING PARA VENDAS")
        print("=" * 70)
        print()
        
        # Etapa 1: Extração
        dados_brutos = self.etapa_1_extrair_dados_brutos()
        if dados_brutos is None:
            return
        
        # Etapa 2: Pré-processamento
        dados_processados = self.etapa_2_preprocessamento()
        
        # Etapa 3: Preparação das features
        X_reg, y_reg, X_clf, y_clf = self.etapa_3_preparar_features()
        
        # Etapa 4: Divisão treino/teste
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf = self.etapa_4_dividir_treino_teste(X_reg, y_reg, X_clf, y_clf)
        
        # Etapa 5: Treinamento
        modelos = self.etapa_5_treinar_modelos(X_train_reg, y_train_reg, X_train_clf, y_train_clf)
        
        # Etapa 6: Avaliação
        resultados = self.etapa_6_avaliar_modelos(X_test_reg, y_test_reg, X_test_clf, y_test_clf)
        
        # Etapa 7: Salvamento
        self.etapa_7_salvar_modelos()
        
        print("🎉 PIPELINE DE MACHINE LEARNING CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        
        return self.dados_processados, self.modelos, resultados

# EXECUÇÃO DO PIPELINE
if __name__ == "__main__":
    # Criar instância do pipeline
    pipeline = PipelineMLVendas()
    
    # Executar pipeline completo
    dados_finais, modelos_treinados, metricas = pipeline.executar_pipeline_completo()
    
    print("\n🎯 RESUMO FINAL:")
    print("=" * 50)
    print(f"📊 Dados processados: {len(dados_finais) if dados_finais is not None else 0} registros")
    print(f"🤖 Modelos treinados: {len(modelos_treinados)} modelos")
    print("📁 Arquivos salvos: 4 modelos em formato .pkl")
    print("\n✅ Pipeline pronto para produção!")