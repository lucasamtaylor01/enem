from pathlib import Path
from typing import Tuple

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

ANOS_DISPONIVEIS = list(range(2015, 2024))

COLUNAS_MICRODADOS_NECESSARIAS = [
    "NO_MUNICIPIO_PROVA",
    "CO_MUNICIPIO_PROVA",
    "IN_TREINEIRO",
    "SG_UF_PROVA",
    "Q006",
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]


def separar_dados_participantes_resultados(
    df_microdados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa microdados anuais em DataFrames de participantes e resultados.

    Args:
        df_microdados: DataFrame bruto anual com colunas de participantes e notas.

    Returns:
        Tupla contendo:
        1) DataFrame de participantes (perfil socioeconomico e local da prova),
        2) DataFrame de resultados (presenca e notas).
    """

    df_participantes = df_microdados[
        [
            "NO_MUNICIPIO_PROVA",
            "CO_MUNICIPIO_PROVA",
            "IN_TREINEIRO",
            "SG_UF_PROVA",
            "Q006",
        ]
    ]
    df_resultado = df_microdados[
        [
            "SG_UF_PROVA",
            "CO_MUNICIPIO_PROVA",
            "NO_MUNICIPIO_PROVA",
            "TP_PRESENCA_CN",
            "TP_PRESENCA_CH",
            "TP_PRESENCA_LC",
            "TP_PRESENCA_MT",
            "NU_NOTA_CN",
            "NU_NOTA_CH",
            "NU_NOTA_LC",
            "NU_NOTA_MT",
            "NU_NOTA_REDACAO",
        ]
    ]

    return df_participantes, df_resultado


def preparar_diretorios() -> None:
    """Cria os diretorios de saida usados no pipeline, se necessario."""

    OUTDIR_TRATAMENTO_BASE.mkdir(parents=True, exist_ok=True)
    OUTDIR_MODELO.mkdir(parents=True, exist_ok=True)
    OUTDIR_REPORT.mkdir(parents=True, exist_ok=True)


def caminhos_processados(ano: int) -> Tuple[Path, Path]:
    """Monta os caminhos de saida por municipio para um ano.

    Args:
        ano: Ano de referencia do processamento.

    Returns:
        Tupla com caminho do csv tratado e caminho do csv do modelo por municipio.
    """

    outdir_tratamento = OUTDIR_TRATAMENTO_BASE / str(ano)
    outdir_tratamento.mkdir(parents=True, exist_ok=True)

    caminho_tratado = (
        outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_TRATADO_{ano}.csv"
    )
    caminho_modelo = (
        outdir_tratamento / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_MODELO_{ano}.csv"
    )

    return caminho_tratado, caminho_modelo


def caminhos_processados_tratamento(ano: int) -> Tuple[Path, Path, Path, Path]:
    """Monta todos os caminhos de saida do tratamento para um ano.

    Args:
        ano: Ano de referencia do processamento.

    Returns:
        Tupla com os caminhos, nesta ordem:
        1) tratado por municipio,
        2) modelo por municipio,
        3) tratado por UF,
        4) modelo por UF.
    """

    caminho_tratado_municipio, caminho_modelo_municipio = caminhos_processados(ano)
    outdir_tratamento = caminho_tratado_municipio.parent
    caminho_tratado_uf = (
        outdir_tratamento / f"ANALISE_NOTAS_ENEM_UF_BRASIL_TRATADO_{ano}.csv"
    )
    caminho_modelo_uf = (
        outdir_tratamento / f"ANALISE_NOTAS_ENEM_UF_BRASIL_MODELO_{ano}.csv"
    )

    return (
        caminho_tratado_municipio,
        caminho_modelo_municipio,
        caminho_tratado_uf,
        caminho_modelo_uf,
    )


def arquivos_processados_existem(ano: int) -> bool:
    """Verifica se os arquivos de saida do ano ja foram gerados.

    Args:
        ano: Ano de referencia do processamento.

    Returns:
        True quando ambos os arquivos esperados existem; caso contrario, False.
    """

    caminho_tratado, caminho_modelo = caminhos_processados(ano)
    return caminho_tratado.exists() and caminho_modelo.exists()


def arquivos_processados_tratamento_existem(ano: int) -> bool:
    """Verifica se todos os arquivos do tratamento do ano ja foram gerados.

    Args:
        ano: Ano de referencia do processamento.

    Returns:
        True quando todos os arquivos esperados existem; caso contrario, False.
    """

    (
        caminho_tratado_municipio,
        caminho_modelo_municipio,
        caminho_tratado_uf,
        caminho_modelo_uf,
    ) = caminhos_processados_tratamento(ano)

    return all(
        caminho.exists()
        for caminho in (
            caminho_tratado_municipio,
            caminho_modelo_municipio,
            caminho_tratado_uf,
            caminho_modelo_uf,
        )
    )


def carregar_dados_brutos(ano: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados brutos do ENEM e retorna participantes e resultados.

    Apenas as colunas necessarias ao pipeline sao carregadas para reduzir
    custo de memoria e tempo de leitura.

    Args:
        ano: Ano de referencia entre 2015 e 2023.

    Returns:
        Tupla com DataFrames de participantes e resultados brutos.

    Raises:
        ValueError: Quando o ano informado nao e suportado.
    """

    if ano in ARQUIVOS_MICRODADOS:
        arquivo_microdados = INDIR / ARQUIVOS_MICRODADOS[ano]
        df_microdados = pd.read_csv(
            arquivo_microdados,
            sep=";",
            encoding="latin-1",
            usecols=COLUNAS_MICRODADOS_NECESSARIAS,
        )
        return separar_dados_participantes_resultados(df_microdados)

    raise ValueError("Ano inválido. Por favor, escolha um ano entre 2015 e 2023.")
