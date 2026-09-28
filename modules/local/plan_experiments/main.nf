// PLAN_EXPERIMENTS. The workflow's first task: validates the run's params
// and renders every experiment's per-stage Hydra overrides, via
// fisseq_embeddings_pipeline.config.experiments (unit-tested Python rather
// than Groovy). Runs in the pipeline image like every other task.
//
// Deliberately NOT errorStrategy 'ignore': a bad params file must stop the
// run, not silently produce zero experiments.

process PLAN_EXPERIMENTS {
    label 'process_single'
    container "${params.container_image}"

    input:
    path(params_json)

    output:
    path("experiments.json"), emit: plans

    script:
    """
    python -m fisseq_embeddings_pipeline.config.experiments ${params_json} experiments.json
    """
}
