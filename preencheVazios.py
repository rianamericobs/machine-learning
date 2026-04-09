#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preenche células vazias (NaN) de um CSV usando Forward Fill e Backward Fill,
e salva o resultado em um novo arquivo.

Forward Fill (ffill): preenche o NaN com o valor da linha ANTERIOR.
Backward Fill (bfill): preenche o NaN com o valor da linha SEGUINTE.
"""

import pandas as pd

ARQUIVO_ENTRADA = 'diabetes_dataset_comCelulasVazias.csv'
ARQUIVO_SAIDA   = 'diabetes_dataset.csv'

# Leitura
print(f'\n - Lendo "{ARQUIVO_ENTRADA}"...')
df = pd.read_csv(ARQUIVO_ENTRADA)

print(f' - Shape: {df.shape}')
print('\n - Células vazias por coluna (antes):')
print(df.isnull().sum())

# Preenchimento
df = df.ffill().bfill()

# Verificação
nan_restantes = df.isnull().sum().sum()
print('\n - Células vazias por coluna (depois):')
print(df.isnull().sum())

if nan_restantes == 0:
    print('\n ✓ Nenhuma célula vazia restante!')
else:
    print(f'\n ⚠ Ainda restam {nan_restantes} células vazias.')

# Salvar novo arquivo
df.to_csv(ARQUIVO_SAIDA, index=False)
print(f'\n - Arquivo salvo como "{ARQUIVO_SAIDA}"')