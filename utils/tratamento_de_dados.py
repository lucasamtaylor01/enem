from sklearn.preprocessing import StandardScaler

import pandas as pd

from utils.carregamento_dados import (
    ARQUIVOS_MICRODADOS,
    INDIR,
    OUTDIR_TRATAMENTO_BASE,
    preparar_diretorios,
)

preparar_diretorios()

def carregar_ou_tratar_dados(ano: int):
	outdir_tratamento = OUTDIR_TRATAMENTO_BASE / str(ano)
	outdir_tratamento.mkdir(parents=True, exist_ok=True)

	caminho_tratado = outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_TRATADO_{ano}.csv"
	caminho_modelo = outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_MODELO_{ano}.csv"

	if caminho_tratado.exists() and caminho_modelo.exists():
		print(f"Dados tratados de {ano} já existem. Pulando etapa de tratamento...\n")
		df_pre_clustering = pd.read_csv(caminho_tratado)
		x_scaled = pd.read_csv(caminho_modelo)
		return df_pre_clustering, x_scaled

	print(f"Carregando dados brutos de {ano}...\n")

	if ano == 2024:
		arquivo_participantes = INDIR / "PARTICIPANTES_2024.csv"
		df_participantes_raw = pd.read_csv(arquivo_participantes, sep=';', encoding='latin-1')

		arquivo_resultados = INDIR / "RESULTADOS_2024.csv"
		df_resultados_raw = pd.read_csv(arquivo_resultados, sep=';', encoding='latin-1')

	elif ano in ARQUIVOS_MICRODADOS:
		arquivo_microdados = INDIR / ARQUIVOS_MICRODADOS[ano]
		df_microdados = pd.read_csv(arquivo_microdados, sep=';', encoding='latin-1')
		df_participantes_raw, df_resultados_raw = separar_dados_participantes_resultados(df_microdados)

	else:
		raise ValueError("Ano inválido. Por favor, escolha um ano entre 2015 e 2024.")

	print(f"Dados de {ano} carregados com sucesso.\n")

	print(f"Tratando dados de {ano}...\n")
	df_pre_clustering, x_scaled = tratamento_de_dados(df_participantes_raw, df_resultados_raw, ano)
	print(f"Dados de {ano} tratados com sucesso.\n")

	print(f"Salvando dados tratados de {ano}...\n")
	df_pre_clustering.to_csv(caminho_tratado, index=False)
	print(f"Dados tratados e salvos com sucesso em {caminho_tratado}\n")

	print(f"Salvando dados para modelo de clustering de {ano}...\n")
	x_scaled.to_csv(caminho_modelo, index=False)
	print(f"Dados para modelo de clustering salvos com sucesso em {caminho_modelo}\n")

	return df_pre_clustering, x_scaled


def tratamento_participantes(df, ano):

    if ano == 2024:
        df_participantes = df[['IN_TREINEIRO', 'SG_UF_PROVA', 'Q007', 'NO_MUNICIPIO_PROVA']]
    else:
        df_participantes = df[['IN_TREINEIRO', 'SG_UF_PROVA', 'Q006', 'NO_MUNICIPIO_PROVA']]

    df_participantes = df_participantes[df_participantes['IN_TREINEIRO'] != 1]
    df_participantes = df_participantes.drop(columns=['IN_TREINEIRO'])

    cor_renda_map_sm = {
        "A": 0.0,
        "B": 1.0,
        "C": 1.5,
        "D": 2.0,
        "E": 2.5,
        "F": 3.0,
        "G": 4.0,
        "H": 5.0,
        "I": 6.0,
        "J": 7.0,
        "K": 8.0,
        "L": 9.0,
        "M": 10.0,
        "N": 12.0,
        "O": 15.0,
        "P": 20.0,
        "Q": 20.0
    }

    if ano == 2024:
        df_participantes['Q007'] = df_participantes['Q007'].map(cor_renda_map_sm)
        df_participantes['Q007'] = df_participantes['Q007'].astype("Float64")
        df_participantes = df_participantes.rename(columns={'Q007': 'RENDA_FAMILIAR_SM'})
    else:
        df_participantes['Q006'] = df_participantes['Q006'].map(cor_renda_map_sm)
        df_participantes['Q006'] = df_participantes['Q006'].astype("Float64")
        df_participantes = df_participantes.rename(columns={'Q006': 'RENDA_FAMILIAR_SM'})
    

    df_participantes['NO_MUNICIPIO_PROVA'] = df_participantes['NO_MUNICIPIO_PROVA'].str.upper()

    df_municipio = df_participantes.groupby('NO_MUNICIPIO_PROVA').agg(
    RENDA_FAMILIAR_SM_MEDIA=('RENDA_FAMILIAR_SM', 'mean')).reset_index()

    return df_municipio

def tratamento_resultado(df):
    df_resultado = df[['SG_UF_PROVA', 'NO_MUNICIPIO_PROVA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO' ]]
    
    df_resultado = df_resultado[df_resultado['TP_PRESENCA_CN'] == 1]
    df_resultado = df_resultado[df_resultado['TP_PRESENCA_CH'] == 1]
    df_resultado = df_resultado[df_resultado['TP_PRESENCA_LC'] == 1]
    df_resultado = df_resultado[df_resultado['TP_PRESENCA_MT'] == 1]

    df_resultado = df_resultado.drop(columns=['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT'])

    df_resultado['NO_MUNICIPIO_PROVA'] = df_resultado['NO_MUNICIPIO_PROVA'].str.upper()

    df_resultado = df_resultado.groupby('NO_MUNICIPIO_PROVA').agg(
    UF=('SG_UF_PROVA', 'first'),
    QTD_PARTICIPANTES=('NO_MUNICIPIO_PROVA', 'size'),
    NU_NOTA_CN_MEDIA=('NU_NOTA_CN', 'mean'),
    NU_NOTA_CH_MEDIA=('NU_NOTA_CH', 'mean'),
    NU_NOTA_LC_MEDIA=('NU_NOTA_LC', 'mean'),
    NU_NOTA_MT_MEDIA=('NU_NOTA_MT', 'mean'),
    NU_NOTA_REDACAO_MEDIA=('NU_NOTA_REDACAO', 'mean')
    ).reset_index()
    
    df_resultado['MEDIA_GERAL'] = df_resultado[['NU_NOTA_CN_MEDIA', 'NU_NOTA_CH_MEDIA', 'NU_NOTA_LC_MEDIA', 'NU_NOTA_MT_MEDIA', 'NU_NOTA_REDACAO_MEDIA']].mean(axis=1)  

    return df_resultado

def tratamento_clustering(df_municipio, df_resultado):
    df_pre_clustering = df_resultado.merge(
    df_municipio,
    on='NO_MUNICIPIO_PROVA',
    how='left'
    )

    X_scaled = df_pre_clustering.copy()

    X_scaled = X_scaled.drop(columns=['NO_MUNICIPIO_PROVA', 'RENDA_FAMILIAR_SM_MEDIA', 'UF', 'QTD_PARTICIPANTES'])

    col_scatter = ['NU_NOTA_CN_MEDIA', 'NU_NOTA_CH_MEDIA', 'NU_NOTA_LC_MEDIA', 'NU_NOTA_MT_MEDIA', 'NU_NOTA_REDACAO_MEDIA', 'MEDIA_GERAL']
    scaler = StandardScaler()
    X_scaled[col_scatter] = scaler.fit_transform(X_scaled[col_scatter])

    return df_pre_clustering, X_scaled


def tratamento_de_dados(df_participantes_raw, df_resultado_raw, ano):
    df_participantes = tratamento_participantes(df_participantes_raw, ano)
    df_resultado = tratamento_resultado(df_resultado_raw)
    df_pre_clustering, X_scaled = tratamento_clustering(df_participantes, df_resultado)

    return df_pre_clustering, X_scaled

def separar_dados_participantes_resultados(df_microdados):
    df_participantes = df_microdados[['IN_TREINEIRO', 'SG_UF_PROVA', 'Q006', 'NO_MUNICIPIO_PROVA']]
    df_resultado = df_microdados[['SG_UF_PROVA', 'NO_MUNICIPIO_PROVA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO' ]]

    return df_participantes, df_resultado

