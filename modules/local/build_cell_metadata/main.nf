// BUILD_CELL_METADATA. Projects BUILD_CELL_IMAGES' cell_table.parquet down
// to the seven meta_* columns QC_FILTER reads (metadata.parquet) -- the
// stage that makes QC_FILTER the pipeline's shared fan-out point rather
// than BUILD_DATASET.
//
// Before this stage existed, QC_FILTER was fed BUILD_DATASET's own
// metadata.parquet, which made the expensive, image-reading WebDataset
// build a hard dependency of the CellProfiler track too (FILTER_CP_FEATURES
// consumes the same QC output). Now BUILD_DATASET/EMBED_CELLS and
// BUILD_CP_FEATURES hang off QC independently: either track can fail
// without stopping the other. See cell_metadata.py's own module docstring
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
    tuple val(batch_stem), path(cell_table)

    output:
    tuple val(batch_stem), path("metadata.parquet"), emit: metadata

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.cell_metadata \\
        output_dir=. \\
        cell_table=${cell_table} \\
        batch_stem=${batch_stem} \\
        random_seed=${params.random_seed}
    """
}
