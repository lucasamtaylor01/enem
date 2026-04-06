import pandas as pd
from sklearn.cluster import KMeans

from utils.carregamento_dados import ANOS_DISPONIVEIS, OUTDIR_MODELO
from utils.fluxo_dados import carregar_ou_tratar_dados


def clustering_de_dados(
    df_pre_clustering: pd.DataFrame,
    x_scaled: pd.DataFrame,
) -> pd.DataFrame:
    """Executa KMeans e organiza os clusters por desempenho medio.

    Args:
        df_pre_clustering: DataFrame agregado para o nivel analisado
            (municipio ou UF), contendo as metricas de notas e renda.
        x_scaled: DataFrame com as features usadas no KMeans.

    Returns:
        DataFrame pos-clustering com a coluna CLUSTER reordenada por
        desempenho medio (0 = menor desempenho, 2 = maior desempenho).
    """

    kmeans = KMeans(n_clusters=3, n_init=200, random_state=0)
    kmeans.fit(x_scaled)

    labels = kmeans.labels_
    df_pos_clustering = df_pre_clustering.copy()
    df_pos_clustering["CLUSTER_ORIGINAL"] = labels

    desempenho_por_cluster = (
        df_pos_clustering.groupby("CLUSTER_ORIGINAL")["NOTA_GERAL_MEDIA"]
        .mean()
        .sort_values(ascending=True)
    )

    mapa_clusters = {
        cluster_original: novo_cluster
        for novo_cluster, cluster_original in enumerate(desempenho_por_cluster.index)
    }

    df_pos_clustering["CLUSTER"] = df_pos_clustering["CLUSTER_ORIGINAL"].map(
        mapa_clusters
    )
    df_pos_clustering = df_pos_clustering.drop(columns=["CLUSTER_ORIGINAL"])

    return df_pos_clustering


def processar_ano(ano: int) -> None:
    """Processa um ano completo: tratamento, clustering e exportacao.

    Args:
        ano: Ano de referencia a ser processado.
    """

    print(f"Iniciando processamento do ano {ano}...\n")
    (
        df_pre_clustering_municipio,
        x_scaled_municipio,
        df_pre_clustering_uf,
        x_scaled_uf,
    ) = carregar_ou_tratar_dados(ano)

    print(f"Realizando clustering por municipio para o ano {ano}...\n")
    df_pos_clustering_municipio = clustering_de_dados(
        df_pre_clustering_municipio,
        x_scaled_municipio,
    )
    print(f"Clustering por municipio realizado com sucesso para o ano {ano}.\n")

    (OUTDIR_MODELO / str(ano)).mkdir(parents=True, exist_ok=True)

    caminho_cluster_municipio = (
        OUTDIR_MODELO
        / str(ano)
        / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_CLUSTERS_{ano}.csv"
    )
    df_pos_clustering_municipio.to_csv(caminho_cluster_municipio, index=False)
    print(
        f"Dados de clustering por municipio salvos com sucesso em {caminho_cluster_municipio}\n"
    )

    print(f"Realizando clustering por UF para o ano {ano}...\n")
    df_pos_clustering_uf = clustering_de_dados(
        df_pre_clustering_uf,
        x_scaled_uf,
    )
    print(f"Clustering por UF realizado com sucesso para o ano {ano}.\n")

    caminho_cluster_uf = (
        OUTDIR_MODELO / str(ano) / f"ANALISE_NOTAS_ENEM_UF_BRASIL_CLUSTERS_{ano}.csv"
    )
    df_pos_clustering_uf.to_csv(caminho_cluster_uf, index=False)
    print(f"Dados de clustering por UF salvos com sucesso em {caminho_cluster_uf}\n")


def rodar_todos_os_anos() -> None:
    """Executa o pipeline de clustering para todos os anos disponiveis."""

    for ano in ANOS_DISPONIVEIS:
        processar_ano(ano)
