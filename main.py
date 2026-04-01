from pathlib import Path
import pandas as pd

from utils.tratamento_de_dados import (
	tratamento_de_dados, separar_dados_participantes_resultados
)

from utils.clustering import clustering_de_dados


# -----------------------------------------
# DIRETÓRIOS E CARREGAMENTO DE DADOS
# -----------------------------------------
data_dir = Path("data")
INDIR = Path("data/data_raw")
OUTDIR_TRATAMENTO = Path("data/data_processed")
OUTDIR_TRATAMENTO = Path("data/data_processed")
OUTDIR_TRATAMENTO.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# SELEÇÃO E TRATAMENTO DE DADOS
# -----------------------------------------
print("Digite o ano para análise (2015-2024):")
ano = int(input())

print(f"Criando diretório para o ano {ano}...\n")
OUTDIR_TRATAMENTO = OUTDIR_TRATAMENTO / str(ano)
OUTDIR_TRATAMENTO.mkdir(parents=True, exist_ok=True)
print(f"Diretório criado com sucesso: {OUTDIR_TRATAMENTO}\n")

print(f"Carregando dados de {ano}...\n")

arquivos_microdados = {
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

if ano == 2024:
	arquivo_participantes = INDIR / "PARTICIPANTES_2024.csv"
	df_participantes_raw = pd.read_csv(arquivo_participantes, sep=';', encoding='latin-1')

	arquivo_resultados = INDIR / "RESULTADOS_2024.csv"
	df_resultados_raw = pd.read_csv(arquivo_resultados, sep=';', encoding='latin-1')

elif ano in arquivos_microdados:
	arquivo_microdados = INDIR / arquivos_microdados[ano]
	df_microdados = pd.read_csv(arquivo_microdados, sep=';', encoding='latin-1')
	df_participantes_raw, df_resultados_raw = separar_dados_participantes_resultados(df_microdados)

else:
	print("Ano inválido. Por favor, escolha um ano entre 2015 e 2024.")
	exit(1)

print(f"Dados de {ano} carregados com sucesso.\n")

print(f"Tratando dados de {ano}...\n")
df_pre_clustering, X_scaled = tratamento_de_dados(df_participantes_raw, df_resultados_raw, ano)

print(f"Dados de {ano} tratados com sucesso.\n")

print(f"Salvando dados tratados de {ano}...\n")
caminho_tratado = OUTDIR_TRATAMENTO / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_TRATADO_{ano}.csv"
df_pre_clustering.to_csv(caminho_tratado, index=False)
print(f"Dados tratados e salvos com sucesso em {caminho_tratado}\n")

print(f"Salvando dados para modelo de clustering do {ano}...\n")
caminho_modelo = OUTDIR_TRATAMENTO / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_MODELO_{ano}.csv"
X_scaled.to_csv(caminho_modelo, index=False)
print(f"Dados para modelo de clustering salvos com sucesso em {caminho_modelo}\n")


# -----------------------------------------
# CLUSTERING
# -----------------------------------------

INDIR_MODELO = Path(f"data/data_processed/{ano}")
OUTDIR_MODELO = Path(f"data/data_model/")
OUTDIR_REPORT = Path(f"report/csv/")

OUTDIR_MODELO.mkdir(parents=True, exist_ok=True)
OUTDIR_REPORT.mkdir(parents=True, exist_ok=True)

print(f"Realizando clustering para o ano {ano}...\n")
df_pos_clustering, resumo_clusters = clustering_de_dados(df_pre_clustering, X_scaled)
print(f"Clustering realizado com sucesso para o ano {ano}.\n")


caminho_cluster = OUTDIR_MODELO / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_CLUSTERS_{ano}.csv"
df_pos_clustering.to_csv(caminho_cluster, index=False)
print(f"Dados de clustering salvos com sucesso em {caminho_cluster}\n")

caminho_resumo = OUTDIR_REPORT / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_RESUMO_CLUSTERS_{ano}.csv"
resumo_clusters.to_csv(caminho_resumo, index=False)
print(f"Resumo dos clusters salvo com sucesso em {caminho_resumo}\n")
