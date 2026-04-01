from sklearn.cluster import KMeans


def clustering_de_dados(df_pre_clustering, X_scaled):
    kmeans = KMeans(n_clusters=3, n_init=200, random_state=0)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    df_pos_clustering = df_pre_clustering.copy()
    df_pos_clustering['CLUSTER'] = labels

    resumo_clusters = (
    df_pos_clustering
    .groupby("CLUSTER")
    .agg(
        QTD_MUNICIPIOS=("NO_MUNICIPIO_PROVA", "count"),
        NU_NOTA_CN_MEDIA=("NU_NOTA_CN_MEDIA", "mean"),
        NU_NOTA_CH_MEDIA=("NU_NOTA_CH_MEDIA", "mean"),
        NU_NOTA_LC_MEDIA=("NU_NOTA_LC_MEDIA", "mean"),
        NU_NOTA_MT_MEDIA=("NU_NOTA_MT_MEDIA", "mean"),
        NU_NOTA_REDACAO_MEDIA=("NU_NOTA_REDACAO_MEDIA", "mean"),
        RENDA_FAMILIAR_SM_MEDIA=("RENDA_FAMILIAR_SM_MEDIA", "mean"),
    )
    .reset_index()
    .sort_values("CLUSTER")
    .reset_index(drop=True)
    )
    
    return df_pos_clustering, resumo_clusters