// BUILD_CELL_IMAGES -- the ONLY task that touches starcall-workflow's tree
// or runs a nested snakemake. Three phases:
//
//   1. build_cell_images_enumerate resolves phenotyping_dir/segmentation_dir/
//      sequencing_dir (resolved_dirs.env), each well's grid size and tiles,
//      and writes targets.txt + tiles_manifest.csv -- plus, in cluster mode,
//      the jobscript every starcall child job re-enters the image through.
//   2. One snakemake run of starcall-workflow's own, unmodified Snakefile,
//      against the REAL data dirs (so its own mtime caching reuses whatever
//      is already computed), asking only for starcall's native per-tile
//      outputs. BUILD_DATASET does the cropping -- see docs/architecture.md
//      decision 17.
//   3. build_cell_images_table joins the per-tile CSVs into
//      cell_table.parquet, plus tiles.parquet naming each tile's image/mask.
//
// Local mode (no params.starcall_profile): every starcall rule runs inside
// this one task, `--cores params.snakemake_cores`. Cluster mode: the nested
// snakemake gets `--profile params.starcall_profile`, which says how to
// submit jobs (it lives outside this repo -- nothing scheduler-specific is
// here), plus our --jobscript, which re-enters the image on each node.
// See docs/nextflow.md.
//
// One starcall_workflow_dir per experiment, and never two runs against the
// same tree at once: the --unlock below clears a stale lock from a killed
// run unconditionally, which is only safe under that rule.

include { threadEnv; hydraList } from '../functions'

process BUILD_CELL_IMAGES {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cell_images/${plan.batch_stem}" }, mode: 'copy', pattern: '{cell_table,tiles}.parquet'

    input:
    val(plan)

    output:
    tuple val(plan.batch_stem), path("cell_table.parquet"), path("tiles.parquet"), env("phenotyping_dir"), emit: cell_images

    script:
    def starcall_dir = plan.starcall_workflow_dir
    def cache_dir = params.snakemake_cache_dir ?: "${params.pipeline_dir}/.snakemake_cache"
    def conda_prefix = task.ext.conda_bin_dir ? "PATH=\"${task.ext.conda_bin_dir}:\$PATH\" " : ''
    def snakemake = "${conda_prefix}${task.ext.snakemake_bin}"
    def cluster_mode = params.starcall_profile as boolean
    def jobscript_args = !cluster_mode ? '' : [
        "starcall_job_image=${params.starcall_job_image}",
        "starcall_container_bin=${params.starcall_container_bin}",
        "starcall_job_gpu=${params.starcall_gpu}",
        hydraList('jobscript_binds', [starcall_dir, cache_dir]),
    ].join(' ')
    // In cluster mode --cores is left to the profile: there it is the budget
    // across all submitted jobs, and a local-sized value would silently cap
    // every rule's own threads.
    def submission = cluster_mode
        ? "--profile '${params.starcall_profile}' --jobscript \"\$PWD/starcall_jobscript.sh\""
        : "--cores ${params.snakemake_cores}"
    """
    set -euo pipefail
    ${threadEnv(task.cpus)}

    # Snakemake mkdir's \$XDG_CACHE_HOME/snakemake (falling back to
    # \$HOME/.cache) before it parses a single rule, with no flag to move it,
    # and under Apptainer \$HOME is often a read-only cluster home. \$HOME
    # moves too because --use-conda's bare `conda` reads ~/.condarc.
    export XDG_CACHE_HOME="${cache_dir}"
    export HOME="${cache_dir}/home"
    mkdir -p "\$XDG_CACHE_HOME" "\$HOME"

    python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \\
        output_dir=. \\
        ${plan.cell_images_args} \\
        ${jobscript_args} \\
        random_seed=${params.random_seed}

    # Fully resolved by phase 1, not recomputed here. Exported so the
    # env("phenotyping_dir") output (BUILD_DATASET's bind path) sees it.
    source resolved_dirs.env
    export phenotyping_dir segmentation_dir sequencing_dir

    # The trailing '/' on each value is load-bearing: starcall's rules build
    # every path by plain string concatenation onto these, matching its own
    # 'phenotyping/'-style defaults. Without it a path wildcard silently
    # comes out malformed.
    starcall_config=(
        phenotyping_dir="\$phenotyping_dir/"
        segmentation_dir="\$segmentation_dir/"
        sequencing_dir="\$sequencing_dir/"
    )

    # A killed run leaves <starcall_workflow_dir>/.snakemake/locks behind and
    # the next one dies with "Directory cannot be locked".
    ${snakemake} \\
        --snakefile "${starcall_dir}/workflow/Snakefile" \\
        --directory "${starcall_dir}" \\
        --unlock \\
        --config "\${starcall_config[@]}" \\
        || true

    # '--' stops --config's parser from swallowing the targets as bogus
    # config entries.
    ${snakemake} \\
        --snakefile "${starcall_dir}/workflow/Snakefile" \\
        --directory "${starcall_dir}" \\
        ${submission} \\
        --use-conda --conda-frontend conda \\
        --rerun-triggers mtime \\
        --rerun-incomplete \\
        --config "\${starcall_config[@]}" \\
        -- \\
        \$(cat targets.txt)

    python -m fisseq_embeddings_pipeline.build_cell_images_table \\
        output_dir=. \\
        random_seed=${params.random_seed}

    test -s cell_table.parquet
    """
}
