// Small helpers shared by every module's script block.

// Pin every threaded library to the task's own cpus. Each otherwise
// defaults to "all cores on the machine" and oversubscribes badly when
// several tasks share a node. Emitted inside the script itself (not as a
// beforeScript) so it reaches the container: beforeScript runs on the host.
def threadEnv(cpus) {
    def n = cpus ?: 1
    return [
        'POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    ].collect { name -> "export ${name}=${n}" }.join('\n')
}

// Render an ordered list as one shell-safe Hydra override token. A path
// input holding a single file arrives as one Path rather than a list.
def hydraList(key, values) {
    def items = values instanceof Collection ? values : [values]
    return "'${key}=[${items.join(',')}]'"
}
