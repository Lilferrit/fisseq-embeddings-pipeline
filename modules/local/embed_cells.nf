// EMBED_CELLS, the pipeline's only GPU-bound stage. container
// "${params.container_image}" and a trailing random_seed=${params.random_seed}
// are the two additions every module picks up versus fisseq-data-pipeline's
// modules. The `channels=[...]`/`channel_apply_mask=[...]` list
// interpolation below mirrors aggregate_embeddings.nf's `aggregators=[...]`
// precedent.

process EMBED_CELLS {
    errorStrategy 'ignore'
    label 'process_gpu'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/embeddings/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(shards)   // dataset-*.tar, collected as a real path list from BUILD_DATASET

    output:
    tuple val(batch_stem), path("embeddings.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.embed \\
        output_dir=. \\
        'shard_pattern=./*.tar' \\
        checkpoint_path=${params.cell_dino_checkpoint} \\
        arch=${params.cell_dino_arch} \\
        patch_size=${params.cell_dino_patch_size} \\
        crop_size=${params.cell_dino_crop_size} \\
        'channels=[${params.cell_dino_channels.join(",")}]' \\
        'channel_apply_mask=[${params.cell_dino_channel_apply_mask.join(",")}]' \\
        channel_pool=${params.cell_dino_channel_pool} \\
        device=${params.cell_dino_device} \\
        batch_size=${params.cell_dino_batch_size} \\
        num_workers=${params.cell_dino_num_workers} \\
        random_seed=${params.random_seed}
    """
}
