#!/usr/bin/env python3
"""Tests for pLannotate-based annotation functions.

Unit tests mock out BLAST calls so they run without the pLannotate databases.
Integration tests (marked `slow`) call pLannotate for real and require
`plannotate setupdb` to have been run in the conda environment.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assembler import (
    _build_annotated_record,
    export_genbank_with_plot,
    format_as_genbank,
    get_plasmid_plot_json,
)
from plannotate import resources as rsc

# ── Helpers ─────────────────────────────────────────────────────────────

SIMPLE_SEQ = "ATGAAACCC" * 20  # 180 bp, long enough for pLannotate


def _empty_df() -> pd.DataFrame:
    """Return an empty pLannotate-shaped DataFrame (no BLAST hits)."""
    return pd.DataFrame(columns=rsc.DF_COLS)


def _df_with_cds(qstart: int, qend: int) -> pd.DataFrame:
    """Return a minimal single-row DataFrame simulating a CDS hit from pLannotate.

    get_seq_record reads these columns to build SeqFeature objects.
    """
    row = {col: None for col in rsc.DF_COLS}
    row.update({
        "Feature": "mock_CDS",
        "Type": "CDS",
        "qstart": qstart,
        "qend": qend,
        "sframe": 1,
        "fragment": False,
        "db": "snapgene",
        "pident": 100.0,
        "percmatch": 100.0,
        "abs percmatch": 100.0,
        "pi_permatch": 100.0,
        "Description": "",
        "evalue": 1e-50,
        "score": 200,
        "sseqid": "mock",
        "sstart": 1,
        "send": qend - qstart,
        "length": qend - qstart,
        "slen": qend - qstart,
        "qlen": len(SIMPLE_SEQ),
        "qseq": SIMPLE_SEQ[qstart:qend],
        "priority": 1,
        "wiggle": 0,
        "wstart": qstart,
        "wend": qend,
        "kind": 1,
        "qstart_dup": qstart,
        "qend_dup": qend,
    })
    return pd.DataFrame([row])


# ── _build_annotated_record ──────────────────────────────────────────────


class TestBuildAnnotatedRecord:
    """Unit tests — no BLAST required."""

    def _build(self, df=None, sequence=SIMPLE_SEQ, name="test_construct",
               backbone_name="pTest", insert_name="GFP",
               insert_position=0, insert_length=0,
               reverse_complement_insert=False):
        if df is None:
            df = _empty_df()
        return _build_annotated_record(
            sequence, df, name, backbone_name, insert_name,
            insert_position, insert_length, reverse_complement_insert,
        )

    def test_topology_is_circular(self):
        record = self._build()
        assert record.annotations.get("topology") == "circular"

    def test_molecule_type_is_dna(self):
        record = self._build()
        assert record.annotations.get("molecule_type") == "DNA"

    def test_locus_name_set(self):
        record = self._build(name="my_construct")
        assert record.name == "my_construct"
        assert record.id == "my_construct"

    def test_locus_name_truncated_to_16_chars(self):
        record = self._build(name="a" * 30)
        assert len(record.name) <= 16

    def test_locus_name_special_chars_replaced(self):
        record = self._build(name="my construct+v2")
        assert " " not in record.name
        assert "+" not in record.name
        assert "_" in record.name

    def test_description_with_backbone_and_insert(self):
        record = self._build(insert_name="GFP", backbone_name="pcDNA3")
        assert "GFP" in record.description
        assert "pcDNA3" in record.description

    def test_description_without_backbone(self):
        record = self._build(name="my_construct", backbone_name="", insert_name="")
        assert record.description == "my_construct"

    def test_insert_feature_added_when_not_annotated(self):
        """With an empty df (no pLannotate hits), the insert CDS should be added."""
        record = self._build(
            df=_empty_df(),
            insert_position=10, insert_length=30, insert_name="GFP",
        )
        cds_features = [f for f in record.features if f.type == "CDS"]
        assert len(cds_features) == 1
        assert cds_features[0].qualifiers["label"] == ["GFP"]

    def test_insert_feature_not_added_when_already_annotated(self):
        """If pLannotate already found a feature overlapping the insert region, skip manual add."""
        df = _df_with_cds(qstart=10, qend=40)
        record = self._build(
            df=df,
            insert_position=10, insert_length=30, insert_name="GFP",
        )
        # pLannotate's CDS is present but we should NOT have added a duplicate
        cds_labels = [
            f.qualifiers.get("label", [None])[0]
            for f in record.features if f.type == "CDS"
        ]
        assert "GFP" not in cds_labels

    def test_insert_feature_strand_forward(self):
        record = self._build(
            df=_empty_df(),
            insert_position=0, insert_length=9,
            reverse_complement_insert=False,
        )
        cds = [f for f in record.features if f.type == "CDS"][0]
        assert cds.location.strand == 1

    def test_insert_feature_strand_reverse(self):
        record = self._build(
            df=_empty_df(),
            insert_position=0, insert_length=9,
            reverse_complement_insert=True,
        )
        cds = [f for f in record.features if f.type == "CDS"][0]
        assert cds.location.strand == -1

    def test_insert_feature_position(self):
        record = self._build(
            df=_empty_df(),
            insert_position=20, insert_length=30,
        )
        cds = [f for f in record.features if f.type == "CDS"][0]
        assert int(cds.location.start) == 20
        assert int(cds.location.end) == 50

    def test_no_insert_feature_when_length_zero(self):
        record = self._build(df=_empty_df(), insert_position=0, insert_length=0)
        cds_features = [f for f in record.features if f.type == "CDS"]
        assert len(cds_features) == 0


# ── format_as_genbank ────────────────────────────────────────────────────


class TestFormatAsGenbank:
    """Unit tests — patches annotate() so no BLAST is needed."""

    def _genbank(self, sequence=SIMPLE_SEQ, name="test", insert_name="GFP",
                 backbone_name="pTest", insert_position=0, insert_length=0,
                 reverse_complement_insert=False):
        with patch("assembler.annotate", return_value=_empty_df()):
            return format_as_genbank(
                sequence=sequence, name=name, backbone_name=backbone_name,
                insert_name=insert_name, insert_position=insert_position,
                insert_length=insert_length,
                reverse_complement_insert=reverse_complement_insert,
            )

    def test_valid_genbank_structure(self):
        out = self._genbank()
        assert "LOCUS" in out
        assert "FEATURES" in out
        assert "ORIGIN" in out
        assert "//" in out

    def test_locus_name_in_output(self):
        out = self._genbank(name="my_plasmid")
        assert "my_plasmid" in out

    def test_circular_in_locus_line(self):
        out = self._genbank()
        locus_line = out.split("\n")[0]
        assert "circular" in locus_line.lower()

    def test_sequence_present_in_origin(self):
        out = self._genbank(sequence=SIMPLE_SEQ)
        # ORIGIN section contains the sequence in lowercase, split into chunks
        assert SIMPLE_SEQ[:10].lower() in out.lower()

    def test_insert_cds_in_features(self):
        out = self._genbank(insert_position=0, insert_length=27, insert_name="MyGene")
        assert "MyGene" in out

    def test_returns_string(self):
        assert isinstance(self._genbank(), str)


# ── get_plasmid_plot_json ────────────────────────────────────────────────


class TestGetPlasmidPlotJson:
    """Unit tests — patches get_bokeh so no rendering environment is needed."""

    def _plot_json(self, df=None):
        if df is None:
            df = _empty_df()
        mock_plot = MagicMock()
        with patch("assembler.get_bokeh", return_value=mock_plot), \
             patch("assembler.json_item", return_value={"doc": {"roots": []}, "version": "2.4.1"}):
            return get_plasmid_plot_json(df, linear=False)

    def test_returns_string(self):
        assert isinstance(self._plot_json(), str)

    def test_returns_valid_json(self):
        result = self._plot_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_has_bokeh_doc_structure(self):
        result = self._plot_json()
        parsed = json.loads(result)
        assert "doc" in parsed

    def test_sizing_mode_set_on_plot(self):
        """Verify stretch_width sizing is applied to the plot before serialization."""
        mock_plot = MagicMock()
        with patch("assembler.get_bokeh", return_value=mock_plot), \
             patch("assembler.json_item", return_value={}):
            get_plasmid_plot_json(_empty_df())
        assert mock_plot.sizing_mode == "stretch_width"


# ── export_genbank_with_plot ─────────────────────────────────────────────


class TestExportGenbankWithPlot:
    """Unit tests — patches annotate() so no BLAST is needed."""

    def _export(self, sequence=SIMPLE_SEQ, name="test", insert_name="GFP",
                backbone_name="pTest", insert_position=0, insert_length=0):
        mock_plot = MagicMock()
        with patch("assembler.annotate", return_value=_empty_df()), \
             patch("assembler.get_bokeh", return_value=mock_plot), \
             patch("assembler.json_item", return_value={"doc": {}}):
            return export_genbank_with_plot(
                sequence=sequence, name=name, backbone_name=backbone_name,
                insert_name=insert_name, insert_position=insert_position,
                insert_length=insert_length,
            )

    def test_returns_tuple_of_two_strings(self):
        result = self._export()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_first_element_is_valid_genbank(self):
        gbk, _ = self._export()
        assert "LOCUS" in gbk
        assert "ORIGIN" in gbk
        assert "//" in gbk

    def test_second_element_is_valid_json(self):
        _, plot_json = self._export()
        parsed = json.loads(plot_json)
        assert isinstance(parsed, dict)

    def test_annotate_called_once(self):
        """pLannotate should only run once even though output is used twice."""
        mock_plot = MagicMock()
        with patch("assembler.annotate", return_value=_empty_df()) as mock_annotate, \
             patch("assembler.get_bokeh", return_value=mock_plot), \
             patch("assembler.json_item", return_value={"doc": {}}):
            export_genbank_with_plot(sequence=SIMPLE_SEQ, name="test")
        mock_annotate.assert_called_once()


# ── Integration tests (require plannotate setupdb) ───────────────────────


@pytest.mark.slow
class TestAnnotationIntegration:
    """End-to-end annotation using real pLannotate BLAST databases.

    Run with: pytest tests/test_annotation.py -m slow
    Requires: plannotate setupdb to have been run in the conda environment.
    """

    def test_egfp_in_pcdna31_genbank_has_features(self):
        """A real plasmid assembly should produce a GenBank file with pLannotate features."""
        from library import get_backbone_by_id, get_insert_by_id
        from assembler import assemble_construct, find_mcs_insertion_point

        backbone = get_backbone_by_id("pcDNA3.1(+)")
        insert = get_insert_by_id("EGFP")
        assert backbone and insert

        pos = find_mcs_insertion_point(backbone)
        result = assemble_construct(backbone["sequence"], insert["sequence"], pos)
        assert result.success

        gbk, plot_json = export_genbank_with_plot(
            sequence=result.sequence,
            name="EGFP_pcDNA31",
            backbone_name="pcDNA3.1(+)",
            insert_name="EGFP",
            insert_position=pos,
            insert_length=len(insert["sequence"]),
        )

        assert "LOCUS" in gbk
        assert "FEATURES" in gbk
        # pLannotate should find at least some backbone elements (CMV promoter, etc.)
        assert gbk.count("misc_feature") + gbk.count("CDS") + gbk.count("promoter") > 0
        # Plot JSON should be valid and Bokeh-structured
        parsed = json.loads(plot_json)
        assert "doc" in parsed
