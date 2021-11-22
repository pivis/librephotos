def spawnable_server(conn):
    import os
    os.environ["OMP_NUM_THREADS"] = ""
    from api.semantic_search.semantic_search import semantic_search_instance
    while True:
        imgs = conn.recv()
        if imgs is None:
            break
        imgs_emb, magnitudes = semantic_search_instance.calculate_clip_embeddings(imgs)
        conn.send((imgs_emb, magnitudes))
