from pathlib import Path

import pandas as pd

INDIR = Path("data/data_raw")
OUTDIR_TRATAMENTO_BASE = Path("data/data_processed")
OUTDIR_MODELO = Path("data/data_model")
OUTDIR_REPORT = Path("report/csv")

ARQUIVOS_MICRODADOS = {
	2023: "MICRODADOS_ENEM_2023.csv",
	2022: "MICRODADOS_ENEM_2022.csv",
	2021: "MICRODADOS_ENEM_2021.csv",
	2020: "MICRODADOS_ENEM_2020.csv",
	2019: "MICRODADOS_ENEM_2019.csv",
	2018: "MICRODADOS_ENEM_2018.csv",
	2017: "MICRODADOS_ENEM_2017.csv",
	2016: "MICRODADOS_ENEM_2016.csv",
	2015: "MICRODADOS_ENEM_2015.csv",
}

ANOS_DISPONIVEIS = list(range(2015, 2025))


def separar_dados_participantes_resultados(df_microdados):
	df_participantes = df_microdados[['IN_TREINEIRO', 'SG_UF_PROVA', 'Q006', 'NO_MUNICIPIO_PROVA']]
	df_resultado = df_microdados[['SG_UF_PROVA', 'NO_MUNICIPIO_PROVA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO' ]]

	return df_participantes, df_resultado


def preparar_diretorios() -> None:
	OUTDIR_TRATAMENTO_BASE.mkdir(parents=True, exist_ok=True)
	OUTDIR_MODELO.mkdir(parents=True, exist_ok=True)
	OUTDIR_REPORT.mkdir(parents=True, exist_ok=True)


def caminhos_processados(ano: int):
	outdir_tratamento = OUTDIR_TRATAMENTO_BASE / str(ano)
	outdir_tratamento.mkdir(parents=True, exist_ok=True)

	caminho_tratado = outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_TRATADO_{ano}.csv"
	caminho_modelo = outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_MODELO_{ano}.csv"

	return caminho_tratado, caminho_modelo


def arquivos_processados_existem(ano: int) -> bool:
	caminho_tratado, caminho_modelo = caminhos_processados(ano)
	return caminho_tratado.exists() and caminho_modelo.exists()


def carregar_dados_brutos(ano: int):
	if ano == 2024:
		arquivo_participantes = INDIR / "PARTICIPANTES_2024.csv"
		df_participantes_raw = pd.read_csv(arquivo_participantes, sep=';', encoding='latin-1')

		arquivo_resultados = INDIR / "RESULTADOS_2024.csv"
		df_resultados_raw = pd.read_csv(arquivo_resultados, sep=';', encoding='latin-1')
		return df_participantes_raw, df_resultados_raw

	if ano in ARQUIVOS_MICRODADOS:
		arquivo_microdados = INDIR / ARQUIVOS_MICRODADOS[ano]
		df_microdados = pd.read_csv(arquivo_microdados, sep=';', encoding='latin-1')
		return separar_dados_participantes_resultados(df_microdados)

	raise ValueError("Ano inválido. Por favor, escolha um ano entre 2015 e 2024.")
