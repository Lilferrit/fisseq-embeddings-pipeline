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
// Forces `make_cell_images_bbox`, a patched copy of starcall-workflow's own
// `rule make_cell_images` injected via `ruleorder:` + plain `include:`
// composition (resources/starcall_overrides/{wrapper.smk,
// fixed_cell_images.smk}) -- NOT the real (broken) `make_cell_images`,
// which reads xpos/ypos columns that don't exist in the real cell table
// schema (confirmed against a real starcall-workflow origin/devel
// checkout; see docs/architecture.md decision 17/18 for the full
// rationale and the bug reproduction this fix is built on). Forcing
// make_cell_images_bbox's own per-tile crop-stack output instead of the
// whole-tile phenotype image/segmentation mask directly lets Snakemake's
// ordinary temp() bookkeeping delete those whole-tile intermediates right
// after use -- the actual disk-space fix. dataset.py no longer crops
// anything itself; it only indexes into the crop stacks this stage
// collects.
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
//      symlinks.txt to collect just the two per-tile crop-stack files
//      (crops_tif, mask_crops_tif -- not the CSVs, which phase 3 reads
//      directly from their real locations) into this task's own working
//      directory, preserving
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
    label 'process_medium'
    // Phase 2 runs starcall-workflow's own segmentation rules
    // (stardist/cellpose/tensorflow, out of the image's `ops` conda env)
    // on a CUDA base image, so this is the pipeline's second GPU-capable
    // stage after EMBED_CELLS -- same two-label shape that one uses. The
    // GPU flag itself is NOT taken from nextflow.config's process_gpu
    // containerOptions: this process has its own withName: containerOptions
    // closure, which is the more specific selector and shadows it
    // entirely, so the flag is folded in there (gated on
    // params.starcall_gpu) instead. The label still matters for executor
    // sizing -- an SGE/Slurm profile's own withLabel: 'process_gpu' block
    // (see scratch/nextflow.config for a worked example, and note it wins
    // over process_medium there by being declared last) is what actually
    // requests the GPU resource from the scheduler.
    label 'process_gpu'
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
    tuple val(batch_stem), path("cell_table.parquet"), path("*_grid*", type: 'dir'), emit: cell_images

    when:
    task.ext.when == null || task.ext.when

    script:
    // starcall_workflow_dir is the one field phase 2's --snakefile/
    // --directory flags need as a literal Groovy value (it's not written
    // to resolved_dirs.env -- only the three *_dir fields it defaults
    // are). Everything else (including phenotyping_dir/segmentation_dir/
    // sequencing_dir themselves, when an entry sets them explicitly) is
    // threaded straight through to phase 1 via the same List-vs-scalar
    // Hydra-override idiom as build_dataset/main.nf/build_cp_features/main.nf --
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
    // Resolved here (Groovy) rather than in bash so params.yaml's null
    // default becomes a concrete path Nextflow can also bind-mount --
    // nextflow.config's containerOptions closure for this process repeats
    // the same `?:` fallback, and the two must agree.
    def snakemake_cache_dir = params.snakemake_cache_dir ?: "${params.pipeline_dir}/.snakemake_cache"
    // Empty by default (nextflow.config), so the command this block emits is
    // byte-for-byte today's on the default/Docker path and under -profile
    // local -- the whole cluster path is additive, gated on an executor
    // profile setting this to a non-empty --cluster ... string. Everything
    // below keyed off `cluster_mode` follows the same rule: emitted only when
    // a profile actually opted in.
    def cluster_args = task.ext.snakemake_cluster_args ?: ''
    def cluster_mode = !cluster_args.isEmpty()
    // Emitted immediately before --cores so the cluster flags sit in the
    // middle of the invocation rather than after the `--` target separator,
    // which would make them unparseable target paths. Empty (not even a
    // line) by default.
    def cluster_args_line = cluster_mode ? "${cluster_args} \\\n        " : ''
    // --cores means two different things to snakemake. In local mode it is
    // the local CPU budget (params.snakemake_cores, default 4). In cluster
    // mode it is the GLOBAL budget across all submitted jobs, and it silently
    // CAPS every rule's own threads: Rule.expand_resources does
    // `threads = min(global_resources['_cores'], rule.threads)` (confirmed
    // against a real snakemake 7.32.4 source tree), so leaving it at 4 would
    // quietly downgrade starcall's `threads: 8` rules (segment_cells_bases,
    // segment_nuclei_bases) to 4 and submit them as `-pe serial 4`. Hence a
    // separate, much larger value in cluster mode.
    def snakemake_cores = cluster_mode
        ? (task.ext.snakemake_cluster_cores ?: 128)
        : params.snakemake_cores
    // Site-specific environment the submit script needs (SGE project/queue/
    // runtime, $SGE_ROOT and friends). A Map in an `ext` directive rather than
    // individual params, so the repo side stays scheduler-agnostic and every
    // site-specific value lives in the executor profile -- see
    // nextflow.config's own note on why these are `ext` and not params.yaml.
    def cluster_env_map = task.ext.starcall_cluster_env ?: [:]
    // A null/empty value here means the profile interpolated something that
    // never got set -- e.g. an executor profile reading $SGE_ROOT out of a
    // param that the launch script forgot to pass. Left alone it exports the
    // literal string "null" and the failure surfaces much later, as an
    // unintelligible scheduler or bind-mount error on every child job. Caught
    // generically rather than per-key so the repo side stays
    // scheduler-agnostic.
    def unset_env = cluster_env_map.findAll { _key, value ->
        value == null || value.toString().trim().isEmpty()
    }.keySet()
    if (cluster_mode && unset_env) {
        throw new IllegalStateException(
            "BUILD_CELL_IMAGES: ext.starcall_cluster_env has no value for " +
            "${unset_env.join(', ')} -- the executor profile references " +
            "something that was never set (check the params its sge block " +
            "interpolates, and that your launch script passes them)."
        )
    }
    def cluster_env = cluster_env_map.collect { key, value ->
        "    export ${key}='${value}'"
    }.join('\n')
    // Short, alnum-only, and unique per task: it prefixes every child job's
    // SGE name (-N), and SGE both truncates long names and rejects some
    // characters. Only used for human-facing identification in qstat -- the
    // cleanup trap matches on recorded job IDs, not on this (see
    // sge_submit.sh's own comment on why).
    //
    // NOT task.hash: that is still null while the script block is being
    // rendered (confirmed by inspecting a real .command.sh, where it came
    // out as the literal string "null"), so every task would share one tag.
    // workflow.sessionId is stable across a run and distinct between runs,
    // and task.index separates the experiments within one run.
    // The .sif every per-rule cluster job re-enters the image with. Explicit
    // params.starcall_child_image always wins; when it's null (the default)
    // fall back to the image Nextflow has ALREADY pulled and converted for
    // this very task, so the common case needs no extra param and no separate
    // `apptainer pull`.
    //
    // The fallback has to reconstruct the path rather than ask for it:
    // `task.container` returns the raw `docker://` URI, not the resolved file
    // (measured against a real Nextflow 26.04.6 run), and the `singularity`
    // config scope is not visible from a task context at all ("No such
    // variable: singularity", same). What IS reachable is the cache
    // directory, plus Nextflow's naming rule, measured across image shapes
    // against that same build:
    //   docker://busybox                             -> busybox.img
    //   docker://quay.io/biocontainers/foo:1.0--py_0 -> quay.io-biocontainers-foo-1.0--py_0.img
    //   docker://ghcr.io/Owner/Repo:v1.2.3           -> ghcr.io-Owner-Repo-v1.2.3.img
    // i.e. strip the scheme, turn '/' and ':' into '-', append '.img' (note:
    // .img, not .sif), preserving case.
    //
    // That rule comes from SingularityCache.simpleName(), which is
    // package-private -- an implementation detail, not an API. The risk is
    // real but bounded: a Nextflow upgrade that changes it makes the path
    // stop existing, and the `-s` guard below then fails this task
    // immediately with both candidates named, rather than letting hundreds of
    // child jobs die on the nodes. Set params.starcall_child_image explicitly
    // to opt out of the guesswork entirely.
    def child_image = params.starcall_child_image
    def child_image_origin = 'params.starcall_child_image'
    if (!child_image) {
        // Nextflow resolves its own cache dir from singularity.cacheDir then
        // $NXF_SINGULARITY_CACHEDIR; only the latter is readable from here,
        // so a profile setting the former must mirror it into this `ext`
        // (scratch/nextflow.config does) -- the two must agree.
        def cache_dir = task.ext.starcall_singularity_cache_dir ?:
            System.getenv('NXF_SINGULARITY_CACHEDIR')
        if (cache_dir && params.container_image) {
            def cached_name = params.container_image
                .replaceFirst(/^[a-z0-9]+:\/\//, '')
                .replaceAll(/[\/:]/, '-')
            child_image = "${cache_dir}/${cached_name}.img"
            child_image_origin = 'derived from params.container_image + the Singularity cache dir'
        }
    }
    def job_tag = 'sc' +
        workflow.sessionId.toString().replaceAll(/[^A-Za-z0-9]/, '').take(6) +
        task.index
    def cluster_preamble = !cluster_mode ? '' : """\
    # ---- cluster submission preamble (executor profile opted in) ----------
    # Every child rule job re-enters this image on its own node, so it needs a
    # real .sif FILE. params.container_image is a docker:// URI on the cluster
    # (see scratch/run.sh) -- hundreds of jobs each re-resolving that against
    # one shared Singularity cache, with registry credentials, is exactly what
    # params.starcall_child_image exists to avoid. Fail loudly here rather
    # than letting every child job fail identically N minutes later.
    export STARCALL_SIF='${child_image ?: ''}'
    if [ ! -s "\$STARCALL_SIF" ]; then
        echo "BUILD_CELL_IMAGES: no usable container image for the per-rule cluster jobs." >&2
        echo "  tried (${child_image_origin}): '\$STARCALL_SIF'" >&2
        echo "  Set params.starcall_child_image to a pre-built .sif, or make sure the" >&2
        echo "  Singularity cache dir is visible here (singularity.cacheDir mirrored into" >&2
        echo "  ext.starcall_singularity_cache_dir, or \$NXF_SINGULARITY_CACHEDIR)." >&2
        exit 1
    fi
    export STARCALL_APPTAINER_BIN='${task.ext.starcall_apptainer_bin ?: 'singularity'}'
    # The two helper scripts live on opposite sides of the container
    # boundary and so are addressed by DIFFERENT paths, which is easy to get
    # wrong in exactly one direction each:
    #   - sge_submit.sh is invoked BY snakemake, i.e. inside this task's own
    #     container, so it takes the in-image path (ext.starcall_overrides_dir,
    #     the same one --snakefile uses).
    #   - sge_job_wrapper.sh is what the SCHEDULER runs, on a bare exec node
    #     with no container around it at all, so it must be a HOST path on
    #     shared storage -- the in-image /opt/... path does not exist there.
    # Exported rather than interpolated into the profile's --cluster string
    # so the profile doesn't have to know either path.
    export STARCALL_SUBMIT_SCRIPT='${task.ext.starcall_overrides_dir}/sge_submit.sh'
    export STARCALL_JOB_WRAPPER='${task.ext.starcall_host_overrides_dir ?: ''}/sge_job_wrapper.sh'
    if [ ! -x "\$STARCALL_JOB_WRAPPER" ]; then
        echo "BUILD_CELL_IMAGES: ext.starcall_host_overrides_dir must point at this repo's" >&2
        echo "  resources/starcall_overrides on SHARED STORAGE the exec nodes can read --" >&2
        echo "  the per-rule job wrapper runs outside any container (got: '\$STARCALL_JOB_WRAPPER')" >&2
        exit 1
    fi
    export STARCALL_LOG_DIR='${params.pipeline_dir}/logs/starcall/${batch_stem}'
    export STARCALL_JOB_TAG='${job_tag}'
    export STARCALL_JOBID_FILE="\$PWD/cluster_jobids.txt"
${cluster_env}
    mkdir -p "\$STARCALL_LOG_DIR"
    : > "\$STARCALL_JOBID_FILE"

    # Every host path a child job can touch, bound at its own unchanged
    # location (src == dest) -- starcall's rules concatenate strings onto
    # phenotyping_dir/segmentation_dir/sequencing_dir, so reaching the data
    # under some other in-container path is not enough. \$PWD is this task's
    # own work dir and is NOT optional: snakemake prefixes every generated
    # jobscript with `cd <the directory the submitter was launched from>`
    # (ClusterExecutor.get_job_exec_prefix, confirmed against a real
    # snakemake 7.32.4 source tree), which is this dir, not --directory.
    STARCALL_BINDS="\$(printf '%s\\n' \\
        '${starcall_workflow_dir}' \\
        "\$phenotyping_dir" "\$segmentation_dir" "\$sequencing_dir" \\
        '${snakemake_cache_dir}' "\$PWD" \\
        | sort -u | sed 's|.*|&:&|' | paste -sd, -)"
    export STARCALL_BINDS

    # qdel whatever is still queued if this task dies. snakemake's own
    # --cluster-cancel only fires on a graceful shutdown, and SGE's default
    # terminate is SIGKILL -- which this trap does NOT catch either, so this
    # is defence in depth, not a guarantee. The other two layers are the
    # bounded -l h_rt on every child (sge_submit.sh) and the documented
    # manual sweep in docs/nextflow.md.
    starcall_cancel_children() {
        if [ -s "\$STARCALL_JOBID_FILE" ]; then
            xargs -r qdel < "\$STARCALL_JOBID_FILE" >/dev/null 2>&1 || true
        fi
    }
    trap starcall_cancel_children EXIT INT TERM

    # A SIGKILLed submitter leaves a lock behind in
    # <starcall_workflow_dir>/.snakemake/locks/ and the NEXT run then dies
    # with "Directory cannot be locked" before doing any work. Safe to clear
    # unconditionally here only because this module already requires one
    # starcall_workflow_dir per experiment and forbids concurrent
    # invocations against the same tree (see this file's own header) -- it
    # would be actively dangerous otherwise. Paired with --rerun-incomplete
    # in the cluster args, which handles the other half of killed-mid-flight
    # state.
    ${conda_path_prefix}${task.ext.snakemake_bin} \\
        --snakefile "${task.ext.starcall_overrides_dir}/wrapper.smk" \\
        --directory "${starcall_workflow_dir}" \\
        --unlock \\
        --config phenotyping_dir="\$phenotyping_dir/" segmentation_dir="\$segmentation_dir/" sequencing_dir="\$sequencing_dir/" starcall_workflow_dir="${starcall_workflow_dir}" \\
        || true
    # ---- end cluster submission preamble ---------------------------------
"""
    """
    set -euo pipefail

    # Snakemake builds its SourceCache -- os.makedirs(\$XDG_CACHE_HOME/
    # snakemake, falling back to \$HOME/.cache -- inside Workflow.__init__,
    # i.e. before it parses a single rule, and exposes no CLI flag to move
    # it. Under Singularity/Apptainer's autoMounts the container's \$HOME is
    # the submitting user's real cluster home, which on the Fowler lab
    # nodes is a read-only NFS mount: the whole stage died with "OSError:
    # [Errno 30] Read-only file system: '/net/noble'" before phase 2 did
    # any work (and, with errorStrategy 'ignore', showed up only as a
    # missing cell_table.parquet on an otherwise exit-0 run). Point both
    # vars at params.snakemake_cache_dir (default <pipeline_dir>/
    # .snakemake_cache) -- writable, and persistent across tasks and runs
    # rather than rebuilt per task. \$HOME is redirected too, not just
    # \$XDG_CACHE_HOME: --use-conda shells out to a bare `conda`, which
    # reads ~/.condarc and appends to ~/.conda/environments.txt.
    #
    # These live here rather than in a nextflow.config `beforeScript` for
    # this process on purpose: an executor profile may well set a generic
    # process.beforeScript of its own (scratch/nextflow.config's sge one
    # exports the thread-pool vars), and a repo-side withName: beforeScript
    # is the more specific selector -- it would replace that rather than
    # add to it.
    export XDG_CACHE_HOME="${snakemake_cache_dir}"
    export HOME="${snakemake_cache_dir}/home"
    mkdir -p "\$XDG_CACHE_HOME" "\$HOME"

    python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \\
        output_dir=. \\
        ${enumerate_overrides} \\
        random_seed=${params.random_seed}

    # phenotyping_dir/segmentation_dir/sequencing_dir, fully resolved by
    # phase 1 above (resolve_data_dir) -- not recomputed here.
    source resolved_dirs.env
${cluster_preamble}
    # --snakefile points directly at the static wrapper.smk template baked
    # into the image (or, under -profile local, the checked-out repo -- see
    # ext.starcall_overrides_dir, nextflow.config); no per-task copy or text
    # substitution needed. wrapper.smk itself pulls starcall_workflow_dir
    # out of `config` (populated by --config below) via `os.path.join`, and
    # its own `include:` of fixed_cell_images.smk is a bare relative
    # filename resolved against --snakefile's own directory, so that sibling
    # file resolves correctly with no copy either. See
    # resources/starcall_overrides/wrapper.smk's own comment for why plain
    # `include:` (not `module:`) is what makes this work.
    #
    # Requesting make_cell_images_bbox's crop-stack targets (not the
    # whole-tile stitch_tile_pt/stitch_tile_segmentation outputs directly)
    # lets Snakemake resolve those temp()-wrapped whole-tile intermediates
    # within this one invocation and delete them right after use; only the
    # requested, non-temp crop-stack targets persist under
    # phenotyping_dir/sequencing_dir. task.ext.snakemake_bin
    # (nextflow.config): the ops conda env's
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
        --snakefile "${task.ext.starcall_overrides_dir}/wrapper.smk" \\
        --directory "${starcall_workflow_dir}" \\
        ${cluster_args_line}--cores ${snakemake_cores} \\
        --use-conda --conda-frontend conda \\
        --rerun-triggers mtime \\
        --config phenotyping_dir="\$phenotyping_dir/" segmentation_dir="\$segmentation_dir/" sequencing_dir="\$sequencing_dir/" starcall_workflow_dir="${starcall_workflow_dir}" \\
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
