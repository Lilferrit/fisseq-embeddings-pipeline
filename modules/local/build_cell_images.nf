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
// phenotyping_dir/segmentation_dir/sequencing_dir are all OPTIONAL, and
// their resolution is entirely phase 1's job now (see
// build_cell_images_enumerate.py's resolve_data_dir): an explicit value
// always wins; otherwise starcall_workflow_dir's own project config
// (config.yaml, or default-config.yaml if that's absent -- the same file
// workflow/Snakefile itself would load) is consulted for that key, so a
// project that remaps these paths still resolves correctly; only then does
// it fall back to a subdirectory of starcall_workflow_dir
// ('phenotyping'/'segmentation'/'sequencing', starcall-workflow's own
// documented default). This script: block does none of that resolution
// itself any more -- it reads phase 1's resolved_dirs.env (below) instead,
// so there's exactly one place (Python, unit-tested) that knows how to
// find these three directories.
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
// Three-phase script. Phases 1 and 3 run in this repo's own installed
// package (`params.container_image` -- the same one every other process
// uses, since the root Dockerfile now bakes in starcall-workflow's own
// `ops` conda env as a second, isolated environment rather than as a
// separate image; see that Dockerfile's own comments). Phase 2 -- the one
// step that actually needs `ops` (tensorflow/stardist/cellpose/snakemake)
// -- invokes that env's snakemake via `task.ext.snakemake_bin` (its
// absolute path by default, nextflow.config) rather than via PATH, so
// bare `python` here always resolves to this repo's own venv, never
// ambiguously to `ops`' Python 3.10 (`-profile local` overrides
// `ext.snakemake_bin` back to bare `snakemake`, since that profile has no
// `ops` env at all -- see nextflow.config):
//   1. `python -m fisseq_embeddings_pipeline.build_cell_images_enumerate`
//      -- resolves phenotyping_dir/segmentation_dir/sequencing_dir
//      (writing resolved_dirs.env), resolves each well's grid size,
//      enumerates existing tiles (mirrors dataset.py's discover_tiles
//      glob -- tile existence can't be known before Snakemake runs, but
//      the *grid* already exists on disk today, the same precondition
//      discover_tiles already assumed before this stage existed), and
//      writes targets.txt (Snakemake target paths), tiles_manifest.csv
//      (drives phase 3), and symlinks.txt (drives phase 2's collection
//      loop).
//   2. A single `snakemake <targets>` invocation (via
//      task.ext.snakemake_bin) against the REAL, unredirected
//      phenotyping_dir/segmentation_dir/sequencing_dir resolved_dirs.env
//      names (so Snakemake's own mtime caching reuses whatever's already
//      computed -- see decision 3 in the implementation plan for why this
//      stage doesn't instead redirect phenotyping_dir to force a
//      from-scratch rebuild every run), then a plain symlink loop over
//      symlinks.txt to collect just the two per-tile image files (pt_tif,
//      mask_tif -- not the CSVs, which phase 3 reads directly from their
//      real locations) into this task's own working directory, preserving
//      the {well}_grid{N}/tile{x}x{y}y/ substructure. publishDir's own
//      `mode:` (below) then decides whether these become real copies or
//      another layer of symlinks when published.
//   3. `python -m fisseq_embeddings_pipeline.build_cell_images_table` --
//      joins each tile's segmentation-side {segtype}.csv to
//      sequencing_dir's {segtype}_reads{params}.csv (by index value --
//      both are provably the same RangeIndex per tile, see the module
//      docstring on combine_cell_reads/merge_final_tables) and, if
//      cp_features, the tile's CellProfiler CSV (by row position, renamed
//      cp_<name>), into one cell_table.parquet covering the whole
//      experiment -- the ONE complete, self-sufficient cell table
//      BUILD_DATASET/BUILD_CP_FEATURES need; neither reads starcall-
//      workflow's tree directly any more.
//
// An alternative was considered and rejected for phase 2: overriding
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
    container "${params.container_image}"
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
    // starcall_workflow_dir is the one field phase 2's --snakefile/
    // --directory flags need as a literal Groovy value (it's not written
    // to resolved_dirs.env -- only the three *_dir fields it defaults
    // are). Everything else (including phenotyping_dir/segmentation_dir/
    // sequencing_dir themselves, when an entry sets them explicitly) is
    // threaded straight through to phase 1 via the same List-vs-scalar
    // Hydra-override idiom as build_dataset.nf/build_cp_features.nf --
    // no per-key exclusion needed any more, since
    // BuildCellImagesEnumerateConfig now has a field for every key
    // batch_config can carry.
    def starcall_workflow_dir = batch_config.starcall_workflow_dir
    def enumerate_overrides = batch_config.collect { key, value ->
        (value instanceof List) ? "'${key}=[${value.join(",")}]'" : "${key}=${value}"
    }.join(' \\\n        ')
    // --use-conda shells out to a bare `conda` regardless of
    // snakemake_bin's own absolute path -- see the comment at its call
    // site below. Computed once here (Groovy, at script-generation time,
    // not bash runtime) so -profile local's empty conda_bin_dir emits no
    // PATH= prefix at all, rather than a bash-level conditional.
    def conda_path_prefix = task.ext.conda_bin_dir
        ? "PATH=\"${task.ext.conda_bin_dir}:\$PATH\" "
        : ''
    """
    set -euo pipefail

    python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \\
        output_dir=. \\
        ${enumerate_overrides} \\
        random_seed=${params.random_seed}

    # phenotyping_dir/segmentation_dir/sequencing_dir, fully resolved by
    # phase 1 above (resolve_data_dir) -- not recomputed here.
    source resolved_dirs.env

    # Snakemake resolves stitch_tile_pt/stitch_tile_segmentation's temp-
    # wrapped intermediates within this one invocation; only the requested,
    # non-temp final targets persist under phenotyping_dir/sequencing_dir.
    # task.ext.snakemake_bin (nextflow.config): the ops conda env's
    # absolute path by default (not bare `snakemake` -- that env is
    # deliberately NOT on PATH, see the root Dockerfile, so bare
    # `python`/`snakemake` never ambiguously resolves into it); -profile
    # local overrides this back to bare `snakemake`, since that profile
    # has no ops env at all to point at. A process directive (`ext`), not
    # params.yaml -- see that file's own comment on why.
    #
    # --use-conda itself shells out to a bare `conda` (Conda().prefix_path,
    # snakemake/deployment/conda.py) regardless of snakemake_bin's own
    # absolute path -- and conda's own base env (where that binary lives)
    # is deliberately kept off this image's PATH too (same Dockerfile
    # reasoning), so scope it onto PATH just for this one invocation via
    # task.ext.conda_bin_dir (nextflow.config) rather than polluting the
    # whole container's default PATH; empty under -profile local, which
    # has no /opt/conda at all.
    #
    # The trailing '/' appended to each --config value here (not present in
    # resolved_dirs.env itself -- resolve_data_dir's own return value is
    # deliberately slash-free, matching how phase 1/3's own Python always
    # joins onto it with an explicit '/') matters specifically at this one
    # crossing point: workflow/rules/*.smk builds every output path by
    # plain string concatenation (`sequencing_dir + '{well}_grid.../...'`,
    # no path-joining), matching config.yaml/default-config.yaml's own
    # literal defaults ('phenotyping/', 'segmentation/', 'sequencing/') --
    # confirmed directly: a slash-free --config override here still runs,
    # but silently produces a malformed `{path}` wildcard (an unwanted
    # leading '/') that no rule matches (MissingRuleException) or, worse,
    # recurses without bound inside sequencing.smk's own get_aux_data.
    #
    # The '--' immediately before the target list (not just a style choice)
    # stops --config's own arg parser -- which otherwise keeps consuming
    # tokens past its own key=value entries -- from swallowing the target
    # paths themselves as bogus, unparseable config entries; confirmed
    # against a real multi-well/multi-tile run, where non-empty
    # targets.txt actually has entries to swallow (this repo's own fixture-
    # driven tests never catch it: targets.txt is empty whenever the fixed
    # well-name bug or a from-scratch, unprimed tile grid leaves 0 tiles
    # enumerated, so there's nothing after --config's values to swallow).
    ${conda_path_prefix}${task.ext.snakemake_bin} \\
        --snakefile "${starcall_workflow_dir}/workflow/Snakefile" \\
        --directory "${starcall_workflow_dir}" \\
        --cores ${params.snakemake_cores} \\
        --use-conda --conda-frontend conda \\
        --rerun-triggers mtime \\
        --config phenotyping_dir="\$phenotyping_dir/" segmentation_dir="\$segmentation_dir/" sequencing_dir="\$sequencing_dir/" \\
        -- \\
        \$(cat targets.txt)

    while IFS=\$'\\t' read -r rel_path abs_path; do
        mkdir -p "\$(dirname "\$rel_path")"
        ln -s "\$abs_path" "\$rel_path"
    done < symlinks.txt

    python -m fisseq_embeddings_pipeline.build_cell_images_table \\
        output_dir=. \\
        random_seed=${params.random_seed}
    """
}
