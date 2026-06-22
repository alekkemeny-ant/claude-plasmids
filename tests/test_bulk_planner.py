"""Unit tests for app/bulk_planner.py"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from bulk_planner import (
    BulkPlan,
    _build_batch_prompt,
    _cost_per_row,
    build_job_groups,
    estimate_cost,
    generate_bulk_plan,
    generate_from_template,
)


# ── Mock helper ─────────────────────────────────────────────────────────────

def _make_api_response(payload: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(payload))]
    return mock_response


def _patch_client(return_value: MagicMock):
    """Context manager: patches _make_client so messages.create returns return_value."""
    mock_make_client = MagicMock()
    mock_client = MagicMock()
    mock_make_client.return_value = mock_client
    mock_client.messages.create.return_value = return_value
    return mock_make_client


# ── TestCostPerRow ───────────────────────────────────────────────────────────

class TestCostPerRow:
    def test_known_sonnet_standard(self):
        expected = (300_000 * 3.00 + 8_000 * 15.00) / 1_000_000  # 1.02
        assert _cost_per_row("claude-sonnet-4-6", "standard") == pytest.approx(expected)

    def test_known_haiku_simple(self):
        expected = (200_000 * 1.00 + 1_400 * 5.00) / 1_000_000  # 0.207
        assert _cost_per_row("claude-haiku-4-5-20251001", "simple") == pytest.approx(expected)

    def test_unknown_model_falls_back_to_sonnet(self):
        sonnet_result = _cost_per_row("claude-sonnet-4-6", "standard")
        assert _cost_per_row("claude-unknown-xyz", "standard") == pytest.approx(sonnet_result)

    def test_unknown_complexity_falls_back_to_standard(self):
        standard_result = _cost_per_row("claude-sonnet-4-6", "standard")
        assert _cost_per_row("claude-sonnet-4-6", "imaginary") == pytest.approx(standard_result)


# ── TestEstimateCost ─────────────────────────────────────────────────────────

class TestEstimateCost:
    def test_single_row(self):
        cpr = (300_000 * 3.00 + 8_000 * 15.00) / 1_000_000
        assert estimate_cost(1, "claude-sonnet-4-6", "standard") == round(cpr, 4)

    def test_multiple_rows_scales_linearly(self):
        cpr = (300_000 * 3.00 + 8_000 * 15.00) / 1_000_000
        assert estimate_cost(5, "claude-sonnet-4-6", "standard") == round(5 * cpr, 4)

    def test_zero_rows(self):
        assert estimate_cost(0, "claude-sonnet-4-6", "standard") == 0.0


# ── TestBuildJobGroups ───────────────────────────────────────────────────────

class TestBuildJobGroups:
    def test_single_group_when_all_fit(self):
        # haiku+simple: cpr=0.207, max_per_group=96 → 3 rows fit in one group
        groups = build_job_groups(3, "claude-haiku-4-5-20251001", "simple")
        assert groups == [[0, 1, 2]]

    def test_splits_into_multiple_groups(self):
        # opus-4-7+complex: cpr=(450_000*5+17_000*25)/1e6=2.675, max_per_group=7
        # 15 rows → groups of 7, 7, 1
        groups = build_job_groups(15, "claude-opus-4-7", "complex")
        assert len(groups) == 3
        assert sum(len(g) for g in groups) == 15

    def test_all_indices_covered(self):
        groups = build_job_groups(4, "claude-sonnet-4-6", "standard")
        all_indices = [i for g in groups for i in g]
        assert sorted(all_indices) == list(range(4))

    def test_one_row(self):
        groups = build_job_groups(1, "claude-sonnet-4-6", "standard")
        assert groups == [[0]]


# ── TestBuildBatchPrompt ─────────────────────────────────────────────────────

class TestBuildBatchPrompt:
    _rows = [
        {"name": "EGFP_construct", "description": "Clone EGFP into pcDNA3.1"},
        {"name": "mCherry_construct", "description": "Clone mCherry into pcDNA3.1"},
    ]
    _shared = "pcDNA3.1(+) backbone with BbsI Golden Gate"
    _summaries = [
        "EGFP oligos: fwd=CACC..., rev=AAAC...",
        "mCherry oligos: fwd=CACC..., rev=AAAC...",
    ]

    def test_starts_with_bulk_row_batch_marker(self):
        result = _build_batch_prompt(self._rows, self._shared, self._summaries)
        assert result.splitlines()[0] == "<!-- bulk-row-batch -->"

    def test_contains_shared_context(self):
        result = _build_batch_prompt(self._rows, self._shared, self._summaries)
        assert self._shared in result

    def test_contains_all_row_names(self):
        result = _build_batch_prompt(self._rows, self._shared, self._summaries)
        assert "EGFP_construct" in result
        assert "mCherry_construct" in result

    def test_uses_row_summary_when_available(self):
        result = _build_batch_prompt(self._rows, self._shared, self._summaries)
        assert "EGFP oligos: fwd=CACC" in result

    def test_fallback_to_description_when_summaries_short(self):
        # Only 1 summary for 2 rows → second row falls back to r['description']
        result = _build_batch_prompt(self._rows, self._shared, self._summaries[:1])
        assert "Clone mCherry into pcDNA3.1" in result


# ── TestGenerateBulkPlan ─────────────────────────────────────────────────────

_THREE_ROWS = [
    {"name": "EGFP_construct", "description": "Clone EGFP into pcDNA3.1(+)"},
    {"name": "mCherry_construct", "description": "Clone mCherry into pcDNA3.1(+)"},
    {"name": "sfGFP_construct", "description": "Clone sfGFP into pcDNA3.1(+)"},
]

_PLANNER_PAYLOAD_BASE = {
    "summary": "Three FP constructs in pcDNA3.1(+)",
    "complexity": "standard",
    "model_suggestion": "claude-sonnet-4-6",
    "batch_eligible": False,
    "shared_context": "",
    "row_summaries": [],
    "enriched_rows": [
        {"name": "EGFP_construct", "enriched_prompt": "Clone EGFP into pcDNA3.1(+) MCS", "output_format": "genbank"},
        {"name": "mCherry_construct", "enriched_prompt": "Clone mCherry into pcDNA3.1(+) MCS", "output_format": "genbank"},
        {"name": "sfGFP_construct", "enriched_prompt": "Clone sfGFP into pcDNA3.1(+) MCS", "output_format": "genbank"},
    ],
}


class TestGenerateBulkPlan:
    def _run(self, rows, payload_override=None):
        payload = {**_PLANNER_PAYLOAD_BASE, **(payload_override or {})}
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            return generate_bulk_plan(rows)

    def test_returns_bulk_plan_instance(self):
        result = self._run(_THREE_ROWS)
        assert isinstance(result, BulkPlan)

    def test_enriched_rows_count_matches(self):
        result = self._run(_THREE_ROWS)
        assert len(result.enriched_rows) == 3

    def test_enriched_prompt_has_bulk_row_prefix(self):
        result = self._run(_THREE_ROWS)
        assert all(r["enriched_prompt"].startswith("<!-- bulk-row -->\n") for r in result.enriched_rows)

    def test_batch_prompt_set_when_eligible(self):
        two_rows = _THREE_ROWS[:2]
        payload = {
            **_PLANNER_PAYLOAD_BASE,
            "batch_eligible": True,
            "shared_context": "pcDNA3.1(+) with BbsI Golden Gate",
            "row_summaries": ["EGFP", "mCherry"],
            "enriched_rows": _PLANNER_PAYLOAD_BASE["enriched_rows"][:2],
        }
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            result = generate_bulk_plan(two_rows)
        assert result.batch_prompt is not None
        assert result.batch_prompt.startswith("<!-- bulk-row-batch -->")

    def test_batch_prompt_none_when_not_eligible(self):
        result = self._run(_THREE_ROWS, {"batch_eligible": False})
        assert result.batch_prompt is None

    def test_batch_prompt_none_for_single_row_even_if_eligible(self):
        one_row = [_THREE_ROWS[0]]
        payload = {
            **_PLANNER_PAYLOAD_BASE,
            "batch_eligible": True,
            "shared_context": "pcDNA3.1(+) with BbsI Golden Gate",
            "row_summaries": ["EGFP"],
            "enriched_rows": _PLANNER_PAYLOAD_BASE["enriched_rows"][:1],
        }
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            result = generate_bulk_plan(one_row)
        assert result.batch_prompt is None

    def test_pads_when_planner_returns_fewer_rows(self):
        payload = {
            **_PLANNER_PAYLOAD_BASE,
            "enriched_rows": [_PLANNER_PAYLOAD_BASE["enriched_rows"][0]],  # only 1
        }
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            result = generate_bulk_plan(_THREE_ROWS)
        assert len(result.enriched_rows) == 3
        # Padded rows use original description
        assert "Clone mCherry into pcDNA3.1(+)" in result.enriched_rows[1]["enriched_prompt"]

    def test_strips_markdown_fences(self):
        payload = _PLANNER_PAYLOAD_BASE
        raw_with_fence = f"```json\n{json.dumps(payload)}\n```"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=raw_with_fence)]
        mock_client = _patch_client(mock_response)
        with patch("bulk_planner._make_client", mock_client):
            result = generate_bulk_plan(_THREE_ROWS)
        assert isinstance(result, BulkPlan)

    def test_name_fallback_to_construct_N(self):
        rows_no_name = [{"description": "Clone EGFP into pcDNA3.1(+)"}]
        payload = {
            **_PLANNER_PAYLOAD_BASE,
            "enriched_rows": [{"enriched_prompt": "Clone EGFP", "output_format": "genbank"}],
        }
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            result = generate_bulk_plan(rows_no_name)
        assert result.enriched_rows[0]["name"] == "construct_001"


# ── TestGenerateFromTemplate ─────────────────────────────────────────────────

_TEMPLATE = "Clone {gene} into pcDNA3.1(+) using Gibson Assembly. Export as GenBank."
_CSV_ROWS = [
    {"name": "EGFP_v1", "gene": "EGFP"},
    {"name": "mCherry_v1", "gene": "mCherry"},
    {"name": "sfGFP_v1", "gene": "sfGFP"},
]

_TEMPLATE_PAYLOAD_BASE = {
    "rows": [
        {"name": "EGFP_v1", "enriched_prompt": "<!-- bulk-row -->\nClone EGFP into pcDNA3.1(+)"},
        {"name": "mCherry_v1", "enriched_prompt": "<!-- bulk-row -->\nClone mCherry into pcDNA3.1(+)"},
        {"name": "sfGFP_v1", "enriched_prompt": "<!-- bulk-row -->\nClone sfGFP into pcDNA3.1(+)"},
    ]
}


class TestGenerateFromTemplate:
    def _run(self, csv_rows, payload_override=None):
        payload = {**_TEMPLATE_PAYLOAD_BASE, **(payload_override or {})}
        mock_client = _patch_client(_make_api_response(payload))
        with patch("bulk_planner._make_client", mock_client):
            return generate_from_template(_TEMPLATE, csv_rows)

    def test_returns_simple_batch_ineligible(self):
        result = self._run(_CSV_ROWS)
        assert result.complexity == "simple"
        assert result.batch_eligible is False

    def test_enriched_rows_count_matches_csv(self):
        result = self._run(_CSV_ROWS)
        assert len(result.enriched_rows) == 3

    def test_name_from_planner_wins(self):
        csv_rows = [{"name": "egfp", "gene": "EGFP"}]
        payload = {"rows": [{"name": "EGFP_v2", "enriched_prompt": "<!-- bulk-row -->\nClone EGFP"}]}
        result = self._run(csv_rows, payload)
        assert result.enriched_rows[0]["name"] == "EGFP_v2"

    def test_name_from_csv_name_column(self):
        csv_rows = [{"name": "mCherry_v1", "gene": "mCherry"}]
        payload = {"rows": [{"enriched_prompt": "<!-- bulk-row -->\nClone mCherry"}]}  # no name
        result = self._run(csv_rows, payload)
        assert result.enriched_rows[0]["name"] == "mCherry_v1"

    def test_name_from_csv_Name_column_capital(self):
        csv_rows = [{"Name": "sfGFP_construct", "gene": "sfGFP"}]
        payload = {"rows": [{"enriched_prompt": "<!-- bulk-row -->\nClone sfGFP"}]}
        result = self._run(csv_rows, payload)
        assert result.enriched_rows[0]["name"] == "sfGFP_construct"

    def test_name_fallback_to_construct_N(self):
        csv_rows = [{"gene": "EGFP"}]  # no name column of any kind
        payload = {"rows": [{"enriched_prompt": "<!-- bulk-row -->\nClone EGFP"}]}
        result = self._run(csv_rows, payload)
        assert result.enriched_rows[0]["name"] == "construct_001"

    def test_pads_when_planner_returns_fewer_rows(self):
        payload = {"rows": [_TEMPLATE_PAYLOAD_BASE["rows"][0]]}  # only 1 row
        result = self._run(_CSV_ROWS, payload)
        assert len(result.enriched_rows) == 3
        # Padded rows include the template text
        assert _TEMPLATE in result.enriched_rows[1]["enriched_prompt"]
