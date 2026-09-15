# BUILD_CELL_IMAGES -> BUILD_CELL_METADATA -> QC_FILTER.
#
# QC_FILTER is the point where the two tracks fan out: it hangs off
# BUILD_CELL_METADATA (a flat projection of cell_table.parquet) rather than
# off the expensive, image-reading WebDataset build, so a BUILD_DATASET
# failure can't take the CellProfiler track down with it.


rule build_cell_images:
    """The ONLY rule that touches starcall-workflow's tree or runs a nested
    snakemake. Three phases, ported from the deleted
    modules/local/build_cell_images/main.nf:

    1. `build_cell_images_enumerate` resolves phenotyping_dir/
       segmentation_dir/sequencing_dir (writing resolved_dirs.env), resolves
       each well's grid size, enumerates existing tiles, and writes
       targets.txt / tiles_manifest.csv / symlinks.txt.
    2. One `snakemake <targets>` against the REAL, unredirected data dirs
       (so starcall's own mtime caching reuses whatever is already
       computed), then a collection loop over symlinks.txt.
    3. `build_cell_images_table` joins the per-tile CSVs into the one
       self-sufficient cell_table.parquet everything downstream reads.

    Phase 1's four scratch files go to a SEPARATE declared directory rather
    than into cell_images/{batch}/, which would otherwise publish
    targets.txt and friends into the output tree. Their config fields are
    joined with pathlib's `/`, so an absolute value overrides output_dir.

    `directory()` outputs, not the stable file alone: a rerun then always
    starts from an empty directory (Snakemake rmtree's a directory output
    before the job runs), so a shrinking input can't leave stale per-tile
    trees behind for the next stage's glob to pick up. In symlink mode that
    rmtree removes symlinks only, never their targets under phenotyping_dir.
    The trailing `test -s` turns "wrote nothing" into a non-zero exit, which
    is what makes Snakemake discard the incomplete directory.
    """
    output:
        images=directory("cell_images/{batch}"),
        scratch=directory("work/cell_images/{batch}"),
    params:
        overrides=lambda wc: cell_images_args(wc.batch),
        starcall_dir=lambda wc: BY_STEM[wc.batch]["starcall_workflow_dir"],
        # Was publishDir's `mode:`; now just which command the collection
        # loop runs. -L so a hard copy dereferences starcall's own symlinks
        # rather than copying a dangling link.
        collect="cp -L" if config["cell_images_hard_copy"] else "ln -s",
        cache_dir=config.get("snakemake_cache_dir")
        or f"{PIPELINE_DIR}/.snakemake_cache",
        overrides_dir=STARCALL_OVERRIDES_DIR,
        # --use-conda shells out to a bare `conda` whatever snakemake_bin's
        # own absolute path is, and conda's base env is deliberately kept off
        # the image's PATH -- so scope it on for this one invocation only.
        conda_prefix=(
            f'PATH="{config["conda_bin_dir"]}:$PATH" '
            if config.get("conda_bin_dir")
            else ""
        ),
        snakemake_bin=config["snakemake_bin"],
        cluster_args=config.get("snakemake_cluster_args") or "",
        cluster_preamble=lambda wc: starcall_cluster_preamble(wc.batch),
        cores=lambda wc: (
            config.get("snakemake_cluster_cores", 128)
            if config.get("snakemake_cluster_args")
            else config["snakemake_cores"]
        ),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + r"""
        mkdir -p {output.images} {output.scratch}
        # Absolute for two reasons: the collection loop below cds into the
        # images directory, and phase 3 joins `manifest` onto its own
        # output_dir with pathlib, where only an absolute value overrides.
        SCRATCH="$(cd {output.scratch} && pwd)"

        # Snakemake's SourceCache mkdir's $XDG_CACHE_HOME/snakemake (falling
        # back to $HOME/.cache) inside Workflow.__init__ -- before it parses a
        # single rule, with no CLI flag to move it. Under Apptainer the
        # container's $HOME is the submitting user's real cluster home, which
        # is frequently a read-only NFS mount: the whole stage died with
        # "OSError: [Errno 30] Read-only file system". $HOME is redirected
        # too, not just $XDG_CACHE_HOME, because --use-conda's bare `conda`
        # reads ~/.condarc and appends to ~/.conda/environments.txt.
        export XDG_CACHE_HOME="{params.cache_dir}"
        export HOME="{params.cache_dir}/home"
        mkdir -p "$XDG_CACHE_HOME" "$HOME"

        python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \
            output_dir="$SCRATCH" \
            {params.overrides} \
            random_seed={config[random_seed]}

        # phenotyping_dir/segmentation_dir/sequencing_dir, fully resolved by
        # phase 1 above -- not recomputed here.
        source "$SCRATCH/resolved_dirs.env"
{params.cluster_preamble}
        # The trailing '/' on each --config value (absent from
        # resolved_dirs.env itself) is load-bearing at exactly this crossing
        # point: starcall's rules build every output path by plain string
        # concatenation, matching its own config defaults ('phenotyping/',
        # ...). A slash-free override still runs but silently produces a
        # malformed {{path}} wildcard that no rule matches, or recurses
        # without bound inside sequencing.smk's get_aux_data.
        #
        # The '--' before the target list stops --config's own parser -- which
        # otherwise keeps consuming tokens past its key=value entries -- from
        # swallowing the target paths as bogus config entries.
        {params.conda_prefix}{params.snakemake_bin} \
            --snakefile "{params.overrides_dir}/wrapper.smk" \
            --directory "{params.starcall_dir}" \
            {params.cluster_args} \
            --cores {params.cores} \
            --use-conda --conda-frontend conda \
            --rerun-triggers mtime \
            --config phenotyping_dir="$phenotyping_dir/" \
                     segmentation_dir="$segmentation_dir/" \
                     sequencing_dir="$sequencing_dir/" \
                     starcall_workflow_dir="{params.starcall_dir}" \
            -- \
            $(cat "$SCRATCH/targets.txt")

        # Collect just the two per-tile crop-stack files into this
        # experiment's output directory, preserving the
        # {{well}}_grid{{N}}/tile{{x}}x{{y}}y/ substructure. The CSVs are read
        # by phase 3 straight from their real locations.
        ( cd {output.images} && \
          while IFS=$'\t' read -r rel_path abs_path; do
              mkdir -p "$(dirname "$rel_path")"
              {params.collect} "$abs_path" "$rel_path"
          done < "$SCRATCH/symlinks.txt" )

        python -m fisseq_embeddings_pipeline.build_cell_images_table \
            output_dir={output.images} \
            manifest="$SCRATCH/tiles_manifest.csv" \
            random_seed={config[random_seed]}

        test -s {output.images}/cell_table.parquet
        """


rule build_cell_metadata:
    """cell_table.parquet projected down to the seven meta_* columns QC
    reads. Exists so QC_FILTER depends on the cell table rather than on
    BUILD_DATASET -- see this file's header."""
    input:
        cell_images="cell_images/{batch}",
    output:
        "cell_metadata/{batch}/metadata.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.cell_metadata \
            output_dir=$(dirname {output}) \
            cell_table={input.cell_images}/cell_table.parquet \
            batch_stem={wildcards.batch} \
            random_seed={config[random_seed]}
        """


rule qc_filter:
    """Barcode/variant/edit-distance QC. Its filtered_cells.parquet is the
    one output both tracks consume; the other two are QC-report files.

    QcFilterConfig's field names are bc_threshold/variant_bc_threshold, not
    params.yaml's own barcode_count_threshold/variant_barcode_count_threshold
    -- mapped explicitly below.
    """
    input:
        "cell_metadata/{batch}/metadata.parquet",
    output:
        filtered="qc_filter/{batch}/filtered_cells.parquet",
        barcode_counts="qc_filter/{batch}/barcode_counts.parquet",
        variants="qc_filter/{batch}/variants_per_barcode.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.qcfilter \
            output_dir=$(dirname {output.filtered}) \
            cell_files={input} \
            bc_threshold={config[barcode_count_threshold]} \
            variant_bc_threshold={config[variant_barcode_count_threshold]} \
            edit_distance_threshold={config[edit_distance_threshold]} \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """
