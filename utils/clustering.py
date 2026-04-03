from sklearn.cluster import KMeans

from utils.carregamento_dados import (
    ANOS_DISPONIVEIS,
    OUTDIR_MODELO,
    OUTDIR_REPORT,
)
from utils.tratamento_de_dados import carregar_ou_tratar_dados


def clustering_de_dados(df_pre_clustering, X_scaled):
    """Executa KMeans e organiza os clusters por desempenho medio.

    Args:
        df_pre_clustering: DataFrame com dados agregados por municipio.
        X_scaled: DataFrame com features padronizadas para o modelo.

    Returns:
        Tupla com dataframe pos-clustering por municipio e resumo agregado
        das metricas por cluster.
    """

    kmeans = KMeans(n_clusters=3, n_init=200, random_state=0)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    df_pos_clustering = df_pre_clustering.copy()
    df_pos_clustering['CLUSTER_ORIGINAL'] = labels

    desempenho_por_cluster = (
        df_pos_clustering
        .groupby('CLUSTER_ORIGINAL')['MEDIA_GERAL']
        .mean()
        .sort_values(ascending=True)
    )

    mapa_clusters = {
        cluster_original: novo_cluster
        for novo_cluster, cluster_original in enumerate(desempenho_por_cluster.index)
    }

    df_pos_clustering['CLUSTER'] = df_pos_clustering['CLUSTER_ORIGINAL'].map(mapa_clusters)
    df_pos_clustering = df_pos_clustering.drop(columns=['CLUSTER_ORIGINAL'])

    resumo_clusters = (
        df_pos_clustering
        .groupby('CLUSTER')
        .agg(
            QTD_MUNICIPIOS=('NO_MUNICIPIO_PROVA', 'count'),
            NU_NOTA_CN_MEDIA=('NU_NOTA_CN_MEDIA', 'mean'),
            NU_NOTA_CH_MEDIA=('NU_NOTA_CH_MEDIA', 'mean'),
            NU_NOTA_LC_MEDIA=('NU_NOTA_LC_MEDIA', 'mean'),
            NU_NOTA_MT_MEDIA=('NU_NOTA_MT_MEDIA', 'mean'),
            NU_NOTA_REDACAO_MEDIA=('NU_NOTA_REDACAO_MEDIA', 'mean'),
            RENDA_FAMILIAR_SM_MEDIA=('RENDA_FAMILIAR_SM_MEDIA', 'mean'),
        )
        .reset_index()
        .sort_values('CLUSTER')
        .reset_index(drop=True)
    )
    
    return df_pos_clustering, resumo_clusters


def processar_ano(ano: int):
    """Processa um ano completo: tratamento, clustering e exportacao.

    Args:
        ano: Ano de referencia a ser processado.
    """

    print(f"Iniciando processamento do ano {ano}...\n")
    df_pre_clustering, x_scaled = carregar_ou_tratar_dados(ano)

    print(f"Realizando clustering para o ano {ano}...\n")
    df_pos_clustering, resumo_clusters = clustering_de_dados(df_pre_clustering, x_scaled)
    print(f"Clustering realizado com sucesso para o ano {ano}.\n")

    caminho_cluster = OUTDIR_MODELO / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_CLUSTERS_{ano}.csv"
    df_pos_clustering.to_csv(caminho_cluster, index=False)
    print(f"Dados de clustering salvos com sucesso em {caminho_cluster}\n")

    caminho_resumo = OUTDIR_REPORT / f"ANALISE_NOTAS_ENEM_MUNICIPIOS_BRASIL_RESUMO_CLUSTERS_{ano}.csv"
    resumo_clusters.to_csv(caminho_resumo, index=False)
    print(f"Resumo dos clusters salvo com sucesso em {caminho_resumo}\n")


def rodar_todos_os_anos():
    """Executa o pipeline de clustering para todos os anos disponiveis."""

    for ano in ANOS_DISPONIVEIS:
        processar_ano(ano)