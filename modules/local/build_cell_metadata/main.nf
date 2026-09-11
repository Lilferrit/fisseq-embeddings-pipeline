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
//
// Unlike BUILD_DATASET/BUILD_CP_FEATURES -- which take cell_images_dir as
// a plain Hydra-override string and therefore need nextflow.config's
// containerOptions bind mounts to see it at all -- this process takes
// cell_table.parquet as a real Nextflow `path` input, so Nextflow stages
// it into the task's own workDir like any other channel file and there is
// no arbitrary host path to bind. That works here and not there because
// this stage only ever reads the parquet itself; it never dereferences
// the per-tile crop-stack symlinks that live alongside it (see
// nextflow.config's BUILD_DATASET|BUILD_CP_FEATURES comment).

process BUILD_CELL_METADATA {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cell_metadata/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(cell_table)

    output:
    tuple val(batch_stem), path("metadata.parquet"), emit: metadata

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python -m fisseq_embeddings_pipeline.cell_metadata \\
        output_dir=. \\
        cell_table=${cell_table} \\
        batch_stem=${batch_stem} \\
        random_seed=${params.random_seed}
    """
}
