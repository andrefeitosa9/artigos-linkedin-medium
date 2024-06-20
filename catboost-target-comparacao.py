import pandas as pd
import time
from memory_profiler import memory_usage
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

lances = pd.read_csv('lances.csv')
df_treino = pd.read_csv('train.csv')

df_treino = df_treino.drop(columns=['conta_pagamento', 'endereco'])
df = df_treino.merge(lances, on='id_participante', how='left')

df = df.drop(columns=['id_lance', 'tempo'], axis =1)

df['pais'] = df['pais'].fillna('Desconhecido')
df = df.dropna()

import pandas as pd
import time
from memory_profiler import memory_usage
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

colunas_categoricas = ['leilao', 'mercadoria', 'dispositivo', 'pais', 'ip', 'url']
y = df['resultado']

def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    mem_usage = memory_usage((func, args, kwargs))
    end_time = time.time()
    duration = end_time - start_time
    max_mem = max(mem_usage)
    return duration, max_mem

def cross_val_score_with_encoding(encoder, df, colunas_categoricas, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    model = GradientBoostingClassifier()
    scores = []

    # DataFrames para armazenar as versões codificadas
    all_df_train_encoded = pd.DataFrame()
    all_df_test_encoded = pd.DataFrame()

    for train_index, test_index in kf.split(df):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        encoder.fit(df_train[colunas_categoricas], y_train)
        df_train_encoded = encoder.transform(df_train[colunas_categoricas])
        df_test_encoded = encoder.transform(df_test[colunas_categoricas])

        df_train_encoded['id_participante'] = df_train['id_participante']
        df_test_encoded['id_participante'] = df_test['id_participante']
        df_train_encoded['resultado'] = y_train
        df_test_encoded['resultado'] = y_test

        # Calcular a média das colunas codificadas para cada id_participante
        for col in colunas_categoricas:
            df_train_encoded[f'media_{col}'] = df_train_encoded.groupby('id_participante')[col].transform('mean')
            df_test_encoded[f'media_{col}'] = df_test_encoded.groupby('id_participante')[col].transform('mean')

        # Remover as colunas originais codificadas
        df_train_encoded = df_train_encoded.drop(columns=colunas_categoricas).drop_duplicates('id_participante')
        df_test_encoded = df_test_encoded.drop(columns=colunas_categoricas).drop_duplicates('id_participante')

        # Armazenar as colunas codificadas após a agregação
        all_df_train_encoded = pd.concat([all_df_train_encoded, df_train_encoded], axis=0)
        all_df_test_encoded = pd.concat([all_df_test_encoded, df_test_encoded], axis=0)

        X_train = df_train_encoded.drop(columns=['id_participante', 'resultado'])
        y_train_grouped = df_train_encoded['resultado']
        X_test = df_test_encoded.drop(columns=['id_participante', 'resultado'])
        y_test_grouped = df_test_encoded['resultado']

        model.fit(X_train, y_train_grouped)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test_grouped, y_pred))

    return scores, all_df_train_encoded, all_df_test_encoded

# Funções para avaliação
def evaluate_target_encoding():
    return cross_val_score_with_encoding(encoder_target, df, colunas_categoricas, y)

def evaluate_catboost_encoding():
    return cross_val_score_with_encoding(encoder_catboost, df, colunas_categoricas, y)

if __name__ == '__main__':
    # Preparando os encoders
    encoder_target = TargetEncoder(cols=colunas_categoricas, min_samples_leaf=2, smoothing=2.0)
    encoder_catboost = CatBoostEncoder(cols=colunas_categoricas)

    # Medindo desempenho para Target Encoding
    duration_target, mem_target = measure_performance(evaluate_target_encoding)
    scores_target, df_train_encoded_target, df_test_encoded_target = evaluate_target_encoding()

    # Medindo desempenho para CatBoost Encoding
    duration_catboost, mem_catboost = measure_performance(evaluate_catboost_encoding)
    scores_catboost, df_train_encoded_catboost, df_test_encoded_catboost = evaluate_catboost_encoding()

    # Gerando o DataFrame com a média de score de cada modelo e as medições de desempenho
    resultados = pd.DataFrame({
        'Modelo': ['Target Encoding', 'CatBoost Encoding'],
        'Acurácia Média': [pd.Series(scores_target).mean(), pd.Series(scores_catboost).mean()],
        'Tempo (s)': [duration_target, duration_catboost],
        'Memória (MiB)': [mem_target, mem_catboost]
    })

    # Exibindo os primeiros 5 registros dos DataFrames codificados e agregados
    print("\nPrimeiras 5 linhas do DataFrame codificado com Target Encoding:")
    print(df_train_encoded_target.head())

    print("\nPrimeiras 5 linhas do DataFrame codificado com CatBoost Encoding:")
    print(df_train_encoded_catboost.head())

    print("\nResultados de desempenho:")
    print(resultados)