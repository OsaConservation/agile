"""Hoplite embedding adapter; research dependencies load only when processing."""


def embed_folder(audio_dir, db_path, dataset_name, model_config):
    from ml_collections import config_dict
    from perch_hoplite.agile import embed, source_info
    from perch_hoplite.db import sqlite_usearch_impl

    config = embed.ModelConfig(**config_dict.ConfigDict(model_config))
    sources = source_info.AudioSources((source_info.AudioSourceConfig(
        dataset_name=dataset_name,
        base_path=audio_dir.as_posix(),
        file_glob='**/*.[wW][aA][vV]',
        min_audio_len_s=1.0,
        target_sample_rate_hz=-2,
        shard_len_s=60.0,
    ),))
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        str(db_path),
        usearch_cfg=sqlite_usearch_impl.get_default_usearch_config(config.embedding_dim),
    )
    try:
        worker = embed.EmbedWorker(sources, config, db)
        worker.process_all(target_dataset_name=dataset_name, handle_duplicates='skip')
        return db
    except BaseException:
        db.db.close()
        raise
