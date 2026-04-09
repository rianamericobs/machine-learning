#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preenche células vazias (NaN) de um CSV usando interpolação linear,
e salva o resultado em um novo arquivo.

Interpolação linear: estima o valor vazio com base nos valores
anterior e posterior da coluna, traçando uma linha reta entre eles.
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

# Interpolação linear coluna por coluna
# limit_direction='both' garante que preenche também nas bordas (início/fim)
df = df.interpolate(method='linear', limit_direction='both')

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