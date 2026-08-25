import pytest

from fisseq_embeddings_pipeline.utils.variant import classify_variant

# ---------------------------------------------------------------------------
# classify_variant
# ---------------------------------------------------------------------------


class TestClassifyVariant:
    @pytest.mark.parametrize(
        "v,expected",
        [
            ("WT", "WT"),
            ("A1A", "Synonymous"),
            ("A1B", "Single Missense"),
            ("A1fs", "Frameshift"),
            ("A1X", "Nonsense"),
            ("A1*", "Nonsense"),
            ("A5-|A6-", "3nt Deletion"),
            ("A5-|A9-", "Other"),
            ("A5-", "3nt Deletion"),
            ("garbage", "Other"),
            ("M1K:downsampled-half", "Single Missense"),
            ("A1A:sometag", "Synonymous"),
            ("M1K:tag:extra", "Single Missense"),
        ],
    )
    def test_classify(self, v, expected):
        assert classify_variant(v) == expected
