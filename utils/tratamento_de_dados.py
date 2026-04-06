import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Tuple

import pandas as pd
from utils.carregamento_dados import (
    separar_dados_participantes_resultados as separar_dados_participantes_resultados_base,
)

COLUNAS_NOTAS_CLUSTERING = [
    "NOTA_CN_MEDIA",
    "NOTA_CH_MEDIA",
    "NOTA_LC_MEDIA",
    "NOTA_MT_MEDIA",
    "NOTA_REDACAO_MEDIA",
]

def tratamento_participantes(df_participantes_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra participantes validos e calcula renda media familiar por municipio.

    Args:
        df_participantes_raw: DataFrame bruto com dados de participantes.

    Returns:
        DataFrame agregado por municipio com as colunas:
        COD_MUNICIPIO, MUNICIPIO e RENDA_FAMILIAR_SM_MEDIA.
    """


    df_participantes = df_participantes_raw.rename(columns={'NO_MUNICIPIO_PROVA': 'MUNICIPIO', 
                                                    'CO_MUNICIPIO_PROVA': 'COD_MUNICIPIO'})


    cor_renda_map_sm = {
        "A": 0.0,
        "B": 0.5,
        "C": 1.25,
        "D": 1.75,
        "E": 2.25,
        "F": 2.75,
        "G": 3.5,
        "H": 4.5,
        "I": 5.5,
        "J": 6.5,
        "K": 7.5,
        "L": 8.5,
        "M": 9.5,
        "N": 11.0,
        "O": 13.5,
        "P": 17.5,
        "Q": 20.0,
    }

    df_participantes["Q006"] = (
        df_participantes["Q006"].squeeze().map(cor_renda_map_sm).astype("float64")
    )
    df_participantes = df_participantes.rename(columns={"Q006": "RENDA_FAMILIAR_SM"})

    if "RENDA_FAMILIAR_SM" in df_participantes.columns:
        if not np.issubdtype(
            df_participantes["RENDA_FAMILIAR_SM"].dropna().dtype, np.number
        ):
            df_participantes["RENDA_FAMILIAR_SM"] = (
                df_participantes["RENDA_FAMILIAR_SM"].squeeze().map(cor_renda_map_sm)
            )

        media_q006_por_uf = df_participantes.groupby("SG_UF_PROVA")[
            "RENDA_FAMILIAR_SM"
        ].transform("mean")
        df_participantes["RENDA_FAMILIAR_SM"] = df_participantes[
            "RENDA_FAMILIAR_SM"
        ].fillna(media_q006_por_uf)

        df_participantes["RENDA_FAMILIAR_SM"] = df_participantes[
            "RENDA_FAMILIAR_SM"
        ].fillna(df_participantes["RENDA_FAMILIAR_SM"].mean())

    df_participantes = df_participantes[df_participantes["IN_TREINEIRO"] != 1]
    df_participantes = df_participantes.drop(columns=["IN_TREINEIRO"])

    df_participantes["MUNICIPIO"] = df_participantes["MUNICIPIO"].str.upper()

    df_participantes["RENDA_FAMILIAR_SM_LOG"] = np.log1p(
        df_participantes["RENDA_FAMILIAR_SM"]
    )

    df_tratamento_outlier = df_participantes.copy()

    valores_q1 = [0.25, 0.2, 0.15, 0.1, 0.05]
    valores_q3 = [0.75, 0.8, 0.85, 0.9, 0.95]

    col = "RENDA_FAMILIAR_SM_LOG"

    for q1_val, q3_val in zip(valores_q1, valores_q3):
        df_col_filtrado = df_tratamento_outlier.copy()

        Q1 = df_tratamento_outlier[col].quantile(q1_val)
        Q3 = df_tratamento_outlier[col].quantile(q3_val)
        IQR = Q3 - Q1

        df_col_filtrado = df_col_filtrado[
            (df_col_filtrado[col] >= Q1 - 1.5 * IQR)
            & (df_col_filtrado[col] <= Q3 + 1.5 * IQR)
        ]

        proporcao_removida = (
            (df_tratamento_outlier.shape[0] - df_col_filtrado.shape[0])
            / df_tratamento_outlier.shape[0]
        ) * 100

        if proporcao_removida <= 5:
            print(
                f"Tratamento de outliers de renda | Q1={q1_val}, Q3={q3_val} {proporcao_removida:.2f}% removidos"
            )
            df_tratamento_outlier = df_col_filtrado.copy()
            break
    else:
        print(f"Nenhum corte adequado para {col} — mantendo dados originais\n")

    df_participantes = df_tratamento_outlier.copy()
    df_participantes = df_participantes.drop(columns=["RENDA_FAMILIAR_SM_LOG"])

    df_municipio = (
        df_participantes.groupby("COD_MUNICIPIO")
        .agg(
            MUNICIPIO=("MUNICIPIO", "first"),
            RENDA_FAMILIAR_SM_MEDIA=("RENDA_FAMILIAR_SM", "mean"),
        )
        .reset_index()
    )

    return df_municipio


def tratamento_resultado(df_resultado_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra presenca valida e calcula medias de notas por municipio.

    Args:
        df_resultado_raw: DataFrame bruto com resultados individuais dos participantes.

    Returns:
        DataFrame agregado por municipio com quantidade de participantes,
        medias por area e media geral.
    """

    df_resultado = df_resultado_raw.rename(columns={'CO_MUNICIPIO_PROVA': 'COD_MUNICIPIO'})

    df_resultado = df_resultado[df_resultado["TP_PRESENCA_CN"] == 1]
    df_resultado = df_resultado[df_resultado["TP_PRESENCA_CH"] == 1]
    df_resultado = df_resultado[df_resultado["TP_PRESENCA_LC"] == 1]
    df_resultado = df_resultado[df_resultado["TP_PRESENCA_MT"] == 1]

    df_resultado = df_resultado.drop(
        columns=["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
    )
    
    analise_notas = [
        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_MT",
        "NU_NOTA_REDACAO",
    ]

    df_tratamento_outlier = df_resultado.copy()

    valores_q1 = [0.25, 0.2, 0.15, 0.1, 0.05]
    valores_q3 = [0.75, 0.8, 0.85, 0.9, 0.95]

    n_original = df_resultado.shape[0]

    for q1_val, q3_val in zip(valores_q1, valores_q3):

        df_temp = df_resultado.copy()

        for col in analise_notas:
            Q1 = df_resultado[col].quantile(q1_val)
            Q3 = df_resultado[col].quantile(q3_val)
            IQR = Q3 - Q1

            df_temp = df_temp[
                (df_temp[col] >= Q1 - 1.5 * IQR) & (df_temp[col] <= Q3 + 1.5 * IQR)
            ]

        proporcao_total = ((n_original - df_temp.shape[0]) / n_original) * 100

        if proporcao_total <= 5:
            print(
                f"Tratamento de outliers de notas | Q1={q1_val}, Q3={q3_val} {proporcao_total:.2f}% removidos"
            )
            df_tratamento_outlier = df_temp.copy()
            break
    else:
        print("Nenhum corte global válido — mantendo dados")

    df_resultado = df_tratamento_outlier.copy()

    df_resultado = df_resultado.groupby('COD_MUNICIPIO').agg(
        UF=('SG_UF_PROVA', 'first'),
        QTD_PARTICIPANTES=('COD_MUNICIPIO', 'size'),
        NOTA_CN_MEDIA=('NU_NOTA_CN', 'mean'),
        NOTA_CH_MEDIA=('NU_NOTA_CH', 'mean'),
        NOTA_LC_MEDIA=('NU_NOTA_LC', 'mean'),
        NOTA_MT_MEDIA=('NU_NOTA_MT', 'mean'),
        NOTA_REDACAO_MEDIA=('NU_NOTA_REDACAO', 'mean')
    ).reset_index()

    df_resultado["NOTA_GERAL_MEDIA"] = df_resultado[
        [
            "NOTA_CN_MEDIA",
            "NOTA_CH_MEDIA",
            "NOTA_LC_MEDIA",
            "NOTA_MT_MEDIA",
            "NOTA_REDACAO_MEDIA",
        ]
    ].mean(axis=1)

    return df_resultado


def tratamento_clustering(
    df_municipio: pd.DataFrame,
    df_resultado: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combina dados por municipio e padroniza variaveis para clustering.

    Args:
        df_municipio: DataFrame com indicadores de renda por municipio.
        df_resultado: DataFrame com indicadores de desempenho por municipio.

    Returns:
        Tupla com dataframe pre-clustering por municipio e features
        padronizadas por municipio.
    """

    df_pre_clustering_municipio = df_resultado.merge(
        df_municipio,
        on='COD_MUNICIPIO',
        how='left'
    )

    x_scaled_municipio = escalar_features_clustering(
        df_pre_clustering_municipio,
        [
            "MUNICIPIO",
            "COD_MUNICIPIO",
            "RENDA_FAMILIAR_SM_MEDIA",
            "UF",
            "QTD_PARTICIPANTES",
            "NOTA_GERAL_MEDIA",
        ],
    )

    return df_pre_clustering_municipio, x_scaled_municipio


def agrupar_por_uf(df_pre_clustering_municipio: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os dados municipais por UF usando medias ponderadas por participantes.

    Args:
        df_pre_clustering_municipio: DataFrame pre-clustering no nivel municipal.

    Returns:
        DataFrame agregado por UF com total de participantes, renda media e
        medias ponderadas de notas.
    """

    colunas_notas_media = [*COLUNAS_NOTAS_CLUSTERING, "NOTA_GERAL_MEDIA"]

    return (
        df_pre_clustering_municipio.groupby("UF")
        .apply(
            lambda grupo: pd.Series(
                {
                    "QTD_PARTICIPANTES": grupo["QTD_PARTICIPANTES"].sum(),
                    "RENDA_FAMILIAR_SM_MEDIA": grupo["RENDA_FAMILIAR_SM_MEDIA"].mean(),
                    **{col: media_ponderada(grupo, col) for col in colunas_notas_media},
                }
            )
        )
        .reset_index()
        .sort_values("UF")
        .reset_index(drop=True)
    )


def escalar_features_clustering(
    df_pre_clustering: pd.DataFrame,
    colunas_excluir: Sequence[str],
) -> pd.DataFrame:
    """Prepara features para clustering removendo metadados e escalando notas.

    Args:
        df_pre_clustering: DataFrame com variaveis numericas e metadados.
        colunas_excluir: Colunas removidas antes da etapa de escalonamento.

    Returns:
        DataFrame com as features de notas padronizadas por StandardScaler.
    """

    x_scaled = df_pre_clustering.drop(columns=colunas_excluir).copy()
    scaler = StandardScaler()
    x_scaled[COLUNAS_NOTAS_CLUSTERING] = scaler.fit_transform(
        x_scaled[COLUNAS_NOTAS_CLUSTERING]
    )
    return x_scaled


def media_ponderada(
    grupo: pd.DataFrame,
    coluna: str,
    peso: str = "QTD_PARTICIPANTES",
) -> float | object:
    """Calcula a media ponderada de uma coluna usando o peso informado.

    Args:
        grupo: DataFrame com valores e pesos.
        coluna: Nome da coluna numerica a ser agregada.
        peso: Nome da coluna de pesos usada no calculo.

    Returns:
        Valor da media ponderada ou pd.NA quando a soma dos pesos e zero.
    """

    d = grupo[[coluna, peso]].dropna()
    soma_pesos = d[peso].sum()
    return (d[coluna] * d[peso]).sum() / soma_pesos if soma_pesos != 0 else pd.NA


def tratamento_de_dados(
    df_participantes_raw: pd.DataFrame,
    df_resultado_raw: pd.DataFrame,
    ano: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa o fluxo completo de tratamento para um ano especifico.

    Args:
        df_participantes_raw: DataFrame bruto de participantes.
        df_resultado_raw: DataFrame bruto de resultados.
        ano: Ano de referencia do processamento.

    Returns:
        Tupla com 4 saidas, nesta ordem:
        1) dados pre-clustering por municipio,
        2) features escalonadas por municipio,
        3) dados pre-clustering por UF,
        4) features escalonadas por UF.
    """

    _ = ano

    df_participantes = tratamento_participantes(df_participantes_raw)
    df_resultado = tratamento_resultado(df_resultado_raw)
    df_pre_clustering_municipio, x_scaled_municipio = tratamento_clustering(
        df_participantes, df_resultado
    )
    df_pre_clustering_uf = agrupar_por_uf(df_pre_clustering_municipio)
    x_scaled_uf = escalar_features_clustering(
        df_pre_clustering_uf,
        ["UF", "RENDA_FAMILIAR_SM_MEDIA", "QTD_PARTICIPANTES", "NOTA_GERAL_MEDIA"],
    )

    return (
        df_pre_clustering_municipio,
        x_scaled_municipio,
        df_pre_clustering_uf,
        x_scaled_uf,
    )


def separar_dados_participantes_resultados(
    df_microdados: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encaminha a separacao de microdados para a implementacao canonica.

    Args:
        df_microdados: DataFrame bruto anual com todas as colunas relevantes.

    Returns:
        Tupla contendo DataFrame de participantes e DataFrame de resultados.

    Notes:
        Mantida por compatibilidade para chamadas antigas deste modulo.
        A logica efetiva esta centralizada em utils.carregamento_dados.
    """

    return separar_dados_participantes_resultados_base(df_microdados)