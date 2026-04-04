from sklearn.cluster import KMeans

from utils.carregamento_dados import (
    ANOS_DISPONIVEIS,
    OUTDIR_MODELO,
    OUTDIR_REPORT,
)
from utils.fluxo_dados import carregar_ou_tratar_dados


def clustering_de_dados(df_pre_clustering, x_scaled, coluna_identificacao="MUNICIPIO"):
    """Executa KMeans e organiza os clusters por desempenho medio.

    Args:
        df_pre_clustering: DataFrame com dados agregados por municipio.
        x_scaled: DataFrame com features padronizadas para o modelo.
        coluna_identificacao: Coluna usada para contar os registros no resumo.

    Returns:
        Tupla com dataframe pos-clustering por municipio e resumo agregado
        das metricas por cluster.
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

    nome_coluna_quantidade = (
        "QTD_MUNICIPIOS" if coluna_identificacao == "MUNICIPIO" else "QTD_UFS"
    )

    resumo_clusters = (
        df_pos_clustering.groupby("CLUSTER")
        .agg(
            **{
                nome_coluna_quantidade: (coluna_identificacao, "count"),
                "NOTA_CN_MEDIA": ("NOTA_CN_MEDIA", "mean"),
                "NOTA_CH_MEDIA": ("NOTA_CH_MEDIA", "mean"),
                "NOTA_LC_MEDIA": ("NOTA_LC_MEDIA", "mean"),
                "NOTA_MT_MEDIA": ("NOTA_MT_MEDIA", "mean"),
                "NOTA_REDACAO_MEDIA": ("NOTA_REDACAO_MEDIA", "mean"),
            },
            RENDA_FAMILIAR_SM_MEDIA=("RENDA_FAMILIAR_SM_MEDIA", "mean"),
        )
        .reset_index()
        .sort_values("CLUSTER")
        .reset_index(drop=True)
    )

    return df_pos_clustering, resumo_clusters


def processar_ano(ano: int):
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
    df_pos_clustering_municipio, resumo_clusters_municipio = clustering_de_dados(
        df_pre_clustering_municipio,
        x_scaled_municipio,
    )
    print(f"Clustering por municipio realizado com sucesso para o ano {ano}.\n")

    (OUTDIR_MODELO / str(ano)).mkdir(parents=True, exist_ok=True)
    (OUTDIR_REPORT / str(ano)).mkdir(parents=True, exist_ok=True)

    caminho_cluster_municipio = (
        OUTDIR_MODELO
        / str(ano)
        / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_CLUSTERS_{ano}.csv"
    )
    df_pos_clustering_municipio.to_csv(caminho_cluster_municipio, index=False)
    print(
        f"Dados de clustering por municipio salvos com sucesso em {caminho_cluster_municipio}\n"
    )

    caminho_resumo_municipio = (
        OUTDIR_REPORT
        / str(ano)
        / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_RESUMO_CLUSTERS_{ano}.csv"
    )
    resumo_clusters_municipio.to_csv(caminho_resumo_municipio, index=False)
    print(
        f"Resumo dos clusters por municipio salvo com sucesso em {caminho_resumo_municipio}\n"
    )

    print(f"Realizando clustering por UF para o ano {ano}...\n")
    df_pos_clustering_uf, resumo_clusters_uf = clustering_de_dados(
        df_pre_clustering_uf,
        x_scaled_uf,
        coluna_identificacao="UF",
    )
    print(f"Clustering por UF realizado com sucesso para o ano {ano}.\n")

    caminho_cluster_uf = (
        OUTDIR_MODELO / str(ano) / f"ANALISE_NOTAS_ENEM_UF_BRASIL_CLUSTERS_{ano}.csv"
    )
    df_pos_clustering_uf.to_csv(caminho_cluster_uf, index=False)
    print(f"Dados de clustering por UF salvos com sucesso em {caminho_cluster_uf}\n")

    caminho_resumo_uf = (
        OUTDIR_REPORT
        / str(ano)
        / f"ANALISE_NOTAS_ENEM_UF_BRASIL_RESUMO_CLUSTERS_{ano}.csv"
    )
    resumo_clusters_uf.to_csv(caminho_resumo_uf, index=False)
    print(f"Resumo dos clusters por UF salvo com sucesso em {caminho_resumo_uf}\n")


def rodar_todos_os_anos():
    """Executa o pipeline de clustering para todos os anos disponiveis."""

    for ano in ANOS_DISPONIVEIS:
        processar_ano(ano)
