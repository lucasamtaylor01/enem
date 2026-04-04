import pandas as pd

from utils.carregamento_dados import (
    ARQUIVOS_MICRODADOS,
    INDIR,
    arquivos_processados_tratamento_existem,
    caminhos_processados_tratamento,
    preparar_diretorios,
)
from utils.tratamento_de_dados import (
    separar_dados_participantes_resultados,
    tratamento_de_dados,
)

preparar_diretorios()


def carregar_ou_tratar_dados(ano: int):
    """Orquestra o fluxo entre arquivos, processamento e persistencia anual.

    Args:
        ano: Ano de referencia entre 2015 e 2023.

    Returns:
        Tupla contendo, nesta ordem:
        1) dataframe pre-clustering por municipio,
        2) dataframe escalonado por municipio,
        3) dataframe pre-clustering por UF,
        4) dataframe escalonado por UF.

    Raises:
        ValueError: Quando o ano informado nao e suportado.
    """

    (
        caminho_tratado_municipio,
        caminho_modelo_municipio,
        caminho_tratado_uf,
        caminho_modelo_uf,
    ) = caminhos_processados_tratamento(ano)

    if arquivos_processados_tratamento_existem(ano):
        print(f"Dados tratados de {ano} ja existem. Pulando etapa de tratamento...\n")
        df_pre_clustering_municipio = pd.read_csv(caminho_tratado_municipio)
        x_scaled_municipio = pd.read_csv(caminho_modelo_municipio)
        df_pre_clustering_uf = pd.read_csv(caminho_tratado_uf)
        x_scaled_uf = pd.read_csv(caminho_modelo_uf)
        return (
            df_pre_clustering_municipio,
            x_scaled_municipio,
            df_pre_clustering_uf,
            x_scaled_uf,
        )

    print(f"Carregando dados brutos de {ano}...\n")

    if ano in ARQUIVOS_MICRODADOS:
        arquivo_microdados = INDIR / ARQUIVOS_MICRODADOS[ano]
        df_microdados = pd.read_csv(arquivo_microdados, sep=";", encoding="latin-1")
        (
            df_participantes_raw,
            df_resultados_raw,
        ) = separar_dados_participantes_resultados(df_microdados)
    else:
        raise ValueError("Ano invalido. Por favor, escolha um ano entre 2015 e 2023.")

    print(f"Dados de {ano} carregados com sucesso.\n")

    print(f"Tratando dados de {ano}...\n")
    (
        df_pre_clustering_municipio,
        x_scaled_municipio,
        df_pre_clustering_uf,
        x_scaled_uf,
    ) = tratamento_de_dados(df_participantes_raw, df_resultados_raw, ano)
    print(f"Dados de {ano} tratados com sucesso.\n")

    print(f"Salvando dados tratados por municipio de {ano}...\n")
    df_pre_clustering_municipio.to_csv(caminho_tratado_municipio, index=False)
    print(f"Dados tratados e salvos com sucesso em {caminho_tratado_municipio}\n")

    print(f"Salvando dados de modelo por municipio de {ano}...\n")
    x_scaled_municipio.to_csv(caminho_modelo_municipio, index=False)
    print(
        f"Dados para modelo de clustering salvos com sucesso em {caminho_modelo_municipio}\n"
    )

    print(f"Salvando dados tratados por UF de {ano}...\n")
    df_pre_clustering_uf.to_csv(caminho_tratado_uf, index=False)
    print(f"Dados tratados e salvos com sucesso em {caminho_tratado_uf}\n")

    print(f"Salvando dados de modelo por UF de {ano}...\n")
    x_scaled_uf.to_csv(caminho_modelo_uf, index=False)
    print(
        f"Dados para modelo de clustering salvos com sucesso em {caminho_modelo_uf}\n"
    )

    return (
        df_pre_clustering_municipio,
        x_scaled_municipio,
        df_pre_clustering_uf,
        x_scaled_uf,
    )
