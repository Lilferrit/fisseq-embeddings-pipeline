"""Variant label string classification into biological categories.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/variant.py (SPEC.md §3 decision 2). Defines
:func:`classify_variant`, which parses a variant label (e.g. ``"A123G"``)
into one of ``Frameshift``, ``3nt Deletion``, ``Nonsense``, ``WT``,
``Synonymous``, ``Single Missense``, or ``Other``. An optional ``:<tag>``
metadata suffix (e.g. the ``:downsampled-half`` pseudo-variant tag produced
by ``qcfilter.py``) is stripped before classification.

Checklist correction (IMPLEMENTATION_CHECKLIST.md Epic 0 Story 0.1): the
checklist bullet for this file also names ``variant_classification`` as
something to vendor here. In the real source repo, ``variant_classification``
is actually defined in ``aggregate.py``, not ``utils/variant.py`` -- this
file only ever defines ``classify_variant``. ``variant_classification``
itself will be ported alongside the rest of ``aggregate.py`` in Epic 4/5,
not here.
"""

import re


def classify_variant(v: str) -> str:
    """
    Classify a variant label string into a biological category.

    Any trailing ``:<tag>`` metadata suffix (e.g. ``"M1K:downsampled-half"``)
    is stripped before classification.

    Parameters
    ----------
    v : str
        Variant label string (e.g. ``"A123G"``, ``"A123fs"``, ``"WT"``).

    Returns
    -------
    str
        One of: ``"Frameshift"``, ``"3nt Deletion"``, ``"Nonsense"``,
        ``"WT"``, ``"Synonymous"``, ``"Single Missense"``, or ``"Other"``.
    """
    v = v.split(":", 1)[0]
    if "fs" in v:
        return "Frameshift"
    if v.endswith("-"):
        parts = v.split("|")
        n = len(parts)
        if n == 1:
            return "3nt Deletion"
        if n == 2 and int(parts[0][1:-1]) == int(parts[1][1:-1]) - 1:
            return "3nt Deletion"
        return "Other"
    if "X" in v or "*" in v:
        return "Nonsense"
    if "WT" in v:
        return "WT"
    m = re.match(r"([A-Z])(\d+)([A-Z])", v)
    if m is None:
        return "Other"
    return "Synonymous" if m.group(1) == m.group(3) else "Single Missense"
