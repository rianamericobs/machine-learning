#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preenche células vazias (NaN) de um CSV usando MICE (Multiple Imputation
by Chained Equations), implementado pelo IterativeImputer do sklearn.

Como funciona:
  - Para cada coluna com NaN, treina um modelo de regressão usando
    todas as outras colunas como variáveis preditoras.
  - Repete o processo várias vezes (iterações) até os valores
    convergírem. É o método mais preciso para dados médicos.
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

ARQUIVO_ENTRADA = 'diabetes_dataset_comCelulasVazias.csv'
ARQUIVO_SAIDA   = 'diabetes_dataset.csv'

# Leitura
print(f'\n - Lendo "{ARQUIVO_ENTRADA}"...')
df = pd.read_csv(ARQUIVO_ENTRADA)

print(f' - Shape: {df.shape}')
print('\n - Células vazias por coluna (antes):')
print(df.isnull().sum())

# Separar a coluna alvo (Outcome) para não interferir na imputação
outcome = df['Outcome']
features = df.drop(columns=['Outcome'])

# Aplicar MICE / IterativeImputer
print('\n - Aplicando MICE (IterativeImputer)...')
print('   (pode demorar alguns segundos)')

imputer = IterativeImputer(
    max_iter=10,    # número de iterações até convergir
    random_state=42 # garante reprodutibilidade
)

features_tratado = imputer.fit_transform(features)

# Reconstruir o DataFrame
df_tratado = pd.DataFrame(features_tratado, columns=features.columns)
df_tratado['Outcome'] = outcome.values

# Verificação
nan_restantes = df_tratado.isnull().sum().sum()
print('\n - Células vazias por coluna (depois):')
print(df_tratado.isnull().sum())

if nan_restantes == 0:
    print('\n ✓ Nenhuma célula vazia restante!')
else:
    print(f'\n ⚠ Ainda restam {nan_restantes} células vazias.')

# Salvar novo arquivo
df_tratado.to_csv(ARQUIVO_SAIDA, index=False)
print(f'\n - Arquivo salvo como "{ARQUIVO_SAIDA}"')