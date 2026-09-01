// BUILD_CELL_IMAGES. The ONLY thing in this pipeline that touches
// starcall-workflow's phenotyping_dir/segmentation_dir/sequencing_dir tree
// or invokes Snakemake. Reads a per-experiment config
// (starcall_workflow_dir, phenotyping_dir, segmentation_dir, sequencing_dir,
// wells, grid_size, segmentation_type, use_corrected,
// sequencing_reads_params, cp_features, cellprofiler_cycle,
// cellprofiler_pipeline, cell_images_hard_copy) from one entry of
// params.yaml's `experiments:` list.
//
// starcall_workflow_dir is per-experiment, not a single shared global path
// (unlike container_image/cell_dino_checkpoint): Snakemake keeps
// invocation-scoped state (.snakemake/ locks, log dirs) in its working
// directory, and Nextflow may schedule multiple experiments' BUILD_CELL_IMAGES
// tasks concurrently -- pointing two concurrent invocations at the same
// starcall-workflow checkout risks lock contention / metadata races. Each
// experiment must have (or be given) its own checkout/working directory.
//
// Does NOT force rule make_cell_images's pre-cropped output to exist --
// dataset.py's own _crop_cell already ports that rule's crop algorithm and
// deliberately avoids depending on make_cell_images itself, because that
// rule reads xpos/ypos columns that don't exist in the real cell table
// schema (confirmed against a real starcall-workflow origin/devel
// checkout). Instead this module forces the whole-tile phenotype image and
// segmentation mask that dataset.py already successfully reads today, plus
// (the actual fix this stage exists for) the sequencing-side genotype
// columns dataset.py currently reads from the wrong file. See
// docs/architecture.md's Data contracts section for the full rationale.
//
// Three-phase script, delegating everything but the Snakemake invocation
// and the plain-file collection loop to build_cell_images_glue.py (a
// standalone script alongside this module -- deliberately not importing
// fisseq_embeddings_pipeline, since this is the one stage whose Python
// runs inside params.starcall_container_image, not this repo's own image):
//   1. `glue.py enumerate` -- resolves each well's grid size, enumerates
//      existing tiles (mirrors dataset.py's discover_tiles glob -- tile
//      existence can't be known before Snakemake runs, but the *grid*
//      already exists on disk today, the same precondition discover_tiles
//      already assumed before this stage existed), and writes targets.txt
//      (Snakemake target paths), tiles_manifest.csv (drives step 3), and
//      symlinks.txt (drives step 2's collection loop).
//   2. A single `snakemake <targets>` invocation against the REAL,
//      unredirected phenotyping_dir/segmentation_dir/sequencing_dir (so
//      Snakemake's own mtime caching reuses whatever's already computed --
//      see decision 3 in the implementation plan for why this stage
//      doesn't instead redirect phenotyping_dir to force a from-scratch
//      rebuild every run), then a plain symlink loop over symlinks.txt to
//      collect just the two per-tile image files (pt_tif, mask_tif -- not
//      the CSVs, which step 3 reads directly from their real locations)
//      into this task's own working directory, preserving the
//      {well}_grid{N}/tile{x}x{y}y/ substructure. publishDir's own `mode:`
//      (below) then decides whether these become real copies or another
//      layer of symlinks when published.
//   3. `glue.py build-table` -- joins each tile's segmentation-side
//      {segtype}.csv to sequencing_dir's {segtype}_reads{params}.csv (by
//      index value -- both are provably the same RangeIndex per tile, see
//      the module docstring on combine_cell_reads/merge_final_tables) and,
//      if cp_features, the tile's CellProfiler CSV (by row position,
//      renamed cp_<name>), into one cell_table.parquet covering the whole
//      experiment -- the ONE complete, self-sufficient cell table
//      BUILD_DATASET/BUILD_CP_FEATURES need; neither reads starcall-
//      workflow's tree directly any more.
//
// An alternative was considered and rejected for step 2: overriding
// `--config phenotyping_dir=<this task's own directory>` would make
// Snakemake regenerate the whole chain (including make_cell_images'
// upstream temp intermediates) fresh, directly into this task's own
// directory -- verified against source to actually work, but it throws
// away Snakemake's own incremental caching against the real tree on every
// single run, and would force expensive CellProfiler analysis (normally
// run once) to look "missing" and recompute whenever cp_features is
// enabled. Not wired up; mentioned here only as a documented, available
// alternative for anyone who later wants a fully self-contained, zero-
// external-path output and is fine paying that recompute cost.

process BUILD_CELL_IMAGES {
    errorStrategy 'ignore'
    container "${params.starcall_container_image}"
    // symlink, not copy -- the one deliberate default deviation from every
    // other module's `mode: 'copy'` convention. Governed by the GLOBAL
    // params.cell_images_hard_copy only (params.yaml's "Shared per-
    // experiment defaults" section), not per-experiment: confirmed against
    // a real Nextflow 26.04.6 run that publishDir's `mode:` must be a
    // static value at process-definition time -- unlike `path:`, it does
    // NOT accept a per-task closure (`setMode()` rejects one with "No
    // signature of method... applicable for argument types: (Closure)").
    // A genuinely per-experiment override would need two separate
    // processes (one per mode) with experiments routed between them by
    // their own cell_images_hard_copy value -- not implemented here; flag
    // to revisit if per-experiment granularity is ever actually needed.
    publishDir(
        path: { "${params.pipeline_dir}/cell_images/${batch_stem}" },
        mode: (params.cell_images_hard_copy ? 'copy' : 'symlink'),
    )

    input:
    tuple val(batch_stem), val(batch_config)

    output:
    tuple val(batch_stem), path("cell_table.parquet"), path("*_grid*", type: 'dir')

    script:
    def wells = batch_config.wells.join(',')
    def phenotyping_dir = batch_config.phenotyping_dir
    def segmentation_dir = batch_config.segmentation_dir
    def sequencing_dir = batch_config.sequencing_dir
    def starcall_workflow_dir = batch_config.starcall_workflow_dir
    def grid_size_arg = (batch_config.grid_size != null) ? "--grid-size ${batch_config.grid_size}" : ''
    def segmentation_type = batch_config.segmentation_type ?: 'cells'
    def use_corrected_arg = batch_config.use_corrected ? '--use-corrected' : ''
    def reads_params = batch_config.sequencing_reads_params ?: ''
    def cp_features = batch_config.cp_features ?: false
    def cp_features_arg = cp_features ? '--cp-features' : ''
    def cellprofiler_cycle = batch_config.cellprofiler_cycle ?: ''
    def cellprofiler_pipeline = batch_config.cellprofiler_pipeline ?: ''
    """
    set -euo pipefail

    python3 "${moduleDir}/build_cell_images_glue.py" enumerate \\
        --phenotyping-dir "${phenotyping_dir}" \\
        --sequencing-dir "${sequencing_dir}" \\
        --wells "${wells}" \\
        ${grid_size_arg} \\
        --segmentation-type "${segmentation_type}" \\
        ${use_corrected_arg} \\
        --sequencing-reads-params "${reads_params}" \\
        ${cp_features_arg} \\
        --cellprofiler-cycle "${cellprofiler_cycle}" \\
        --cellprofiler-pipeline "${cellprofiler_pipeline}" \\
        --targets-out targets.txt \\
        --manifest-out tiles_manifest.csv \\
        --symlinks-out symlinks.txt

    # Snakemake resolves stitch_tile_pt/stitch_tile_segmentation's temp-
    # wrapped intermediates within this one invocation; only the requested,
    # non-temp final targets persist under phenotyping_dir/sequencing_dir.
    snakemake \\
        --snakefile "${starcall_workflow_dir}/workflow/Snakefile" \\
        --directory "${starcall_workflow_dir}" \\
        --cores ${params.snakemake_cores} \\
        --use-conda --conda-frontend conda \\
        --rerun-triggers mtime \\
        --config phenotyping_dir="${phenotyping_dir}" segmentation_dir="${segmentation_dir}" sequencing_dir="${sequencing_dir}" \\
        \$(cat targets.txt)

    while IFS=\$'\\t' read -r rel_path abs_path; do
        mkdir -p "\$(dirname "\$rel_path")"
        ln -s "\$abs_path" "\$rel_path"
    done < symlinks.txt

    python3 "${moduleDir}/build_cell_images_glue.py" build-table \\
        --manifest tiles_manifest.csv \\
        --output cell_table.parquet
    """
}
