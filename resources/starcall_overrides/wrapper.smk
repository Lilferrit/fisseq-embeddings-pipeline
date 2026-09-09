# Composes starcall-workflow's real Snakefile with this repo's own rule
# patch, via plain `include:` (shares one Python namespace). Both files in
# this directory are used as-is, straight from
# `task.ext.starcall_overrides_dir` (nextflow.config) -- build_cell_images/
# main.nf passes `--snakefile` pointing directly here, no per-task copy or
# text substitution needed. `starcall_workflow_dir` is threaded in via
# `--config` (the same mechanism build_cell_images/main.nf already uses for
# phenotyping_dir/segmentation_dir/sequencing_dir on the same invocation)
# rather than templated into this file, since Snakemake directives are
# plain Python: `include:` accepts any expression, not just a string
# literal, and `--config` is parsed before this file's own top-level
# statements run. `os` needs no explicit import -- starcall-workflow's own
# Snakefile already calls `os.path.exists(...)` at its own top with none,
# confirming Snakemake makes it available in this global namespace.
# --directory still points at starcall_workflow_dir unchanged (STARCall's
# own config-relative dir resolution depends on Snakemake's working
# directory, not the Snakefile's path). The second `include:` below is a
# bare relative filename, resolved against *this file's own* directory
# (not --directory), so it always finds its sibling here regardless of
# where starcall_workflow_dir points.
include: os.path.join(config["starcall_workflow_dir"], "workflow/Snakefile")
include: "fixed_cell_images.smk"
