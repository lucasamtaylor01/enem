from pathlib import Path
import pandas as pd

from utils.tratamento_de_dados import (
	tratamento_de_dados, separar_dados_participantes_resultados
)


data_dir = Path("data")
INDIR = Path("data/data_raw")
OUTDIR = Path("data/data_processed")

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Digite o ano para análise (2023, 2024):")
ano = int(input())

OUTDIR = OUTDIR / str(ano)
OUTDIR.mkdir(parents=True, exist_ok=True)

if ano == 2024:
	arquivo_participantes = INDIR / "PARTICIPANTES_2024.csv"
	df_participantes_raw = pd.read_csv(arquivo_participantes, sep=';', encoding='latin-1')

	arquivo_resultados = INDIR / "RESULTADOS_2024.csv"
	df_resultados_raw = pd.read_csv(arquivo_resultados, sep=';', encoding='latin-1')

else:
	if ano == 2023:
		arquivo_participantes = INDIR / "MICRODADOS_ENEM_2023.csv"

	if ano == 2022:
		arquivo_participantes = INDIR / "MICRODADOS_ENEM_2022.csv"

	if ano == 2021:
		arquivo_participantes = INDIR / "MICRODADOS_ENEM_2021.csv"

	df_microdados = pd.read_csv(arquivo_participantes, sep=';', encoding='latin-1')
	df_participantes_raw, df_resultados_raw = separar_dados_participantes_resultados(df_microdados, ano)


df_clustering, X_scaled = tratamento_de_dados(df_participantes_raw, df_resultados_raw, ano)

caminho_tratado = OUTDIR / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_TRATADO_{ano}.csv"
df_clustering.to_csv(caminho_tratado, index=False)

caminho_modelo = OUTDIR / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_MODELO_{ano}.csv"
X_scaled.to_csv(caminho_modelo, index=False)
print(f"Dados tratados e salvos com sucesso para o ano de {ano}.")
