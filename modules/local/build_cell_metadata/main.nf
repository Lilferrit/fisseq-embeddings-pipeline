// BUILD_CELL_METADATA. Projects BUILD_CELL_IMAGES' cell_table.parquet down
// to the seven meta_* columns QC_FILTER reads (metadata.parquet) -- the
// stage that makes QC_FILTER the pipeline's shared fan-out point rather
// than the image-reading cellDINO track. EMBED_CELLS also joins its meta_*
// columns back onto each embedded cell (a shard's meta.json carries only the
// cell's location), so both tracks see identical meta_* values.
//
// Before this stage existed, QC_FILTER was fed the (since removed)
// BUILD_DATASET stage's own metadata.parquet, which made the WebDataset build
// a hard dependency of the CellProfiler track too. See cell_metadata.py's own module docstring
// and docs/architecture.md for the full rationale, including why QC can't
// just read cell_table.parquet directly (qcfilter.py's filter_columns
// drops the unprefixed well/tile/tile_cell_index columns that become
// filter.py's JOIN_KEYS).

include { threadEnv } from '../functions'

process BUILD_CELL_METADATA {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cell_metadata/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(cell_table_args), path(cell_table)

    output:
    tuple val(batch_stem), path("metadata.parquet"), emit: metadata

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.cell_metadata \\
        output_dir=. \\
        cell_table=${cell_table} \\
        batch_stem=${batch_stem} \\
        ${cell_table_args} \\
        random_seed=${params.random_seed}
    """
}
