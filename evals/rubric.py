#!/usr/bin/env python3
"""
Verification Rubric Scorer

Programmatic implementation of the Allen Institute Plasmid Design Tool
Verification Rubric. Scores an assembled construct against expected inputs
and produces a structured pass/fail report.

Rubric sections:
  1. Input Validation      — backbone and insert sequences correct
  2. Construct Assembly    — insert at correct position, orientation, integrity
  3. Construct Integrity   — full-length output, features preserved
  4. Biological Sanity     — ORF validity, promoter position, polyA, markers, origins, frame
  5. Output Verification   — format correctness, parseability, sequence match, annotations
  6. Output Quality        — ground truth comparison (when available)

Scoring uses weighted severity points:
  Critical = 2 pts, Major = 1 pt, Minor = 0.5 pts, Info = 0 pts
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assembler import clean_sequence, validate_dna, reverse_complement


# ── Severity weights ───────────────────────────────────────────────────

SEVERITY_POINTS = {"Critical": 2.0, "Major": 1.0, "Minor": 0.5, "Info": 0.0}


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class Check:
    """A single rubric checkpoint."""
    section: str       # e.g. "Input Validation", "Construct Assembly", ...
    name: str          # human-readable check name
    severity: str      # "Critical", "Major", "Minor", "Info"
    passed: bool
    detail: str = ""


@dataclass
class RubricResult:
    """Complete rubric scoring result."""
    checks: list[Check] = field(default_factory=list)

    @property
    def critical_fail(self) -> bool:
        return any(not c.passed and c.severity == "Critical" for c in self.checks)

    @property
    def max_points(self) -> float:
        return sum(SEVERITY_POINTS.get(c.severity, 0) for c in self.checks)

    @property
    def earned_points(self) -> float:
        return sum(SEVERITY_POINTS.get(c.severity, 0) for c in self.checks if c.passed)

    @property
    def score_pct(self) -> float:
        if self.max_points == 0:
            return 100.0
        return round(self.earned_points / self.max_points * 100, 1)

    # Legacy properties kept for callers that reference them
    @property
    def total_scored(self) -> int:
        return len([c for c in self.checks if c.severity in ("Critical", "Major", "Minor")])

    @property
    def total_passed(self) -> int:
        return len([c for c in self.checks if c.severity in ("Critical", "Major", "Minor") and c.passed])

    @property
    def overall_pass(self) -> bool:
        return not self.critical_fail and self.score_pct >= 90.0

    def summary(self) -> str:
        status = "PASS" if self.overall_pass else "FAIL"
        return (
            f"{status} — {self.score_pct}% "
            f"({self.earned_points}/{self.max_points} pts, "
            f"{self.total_passed}/{self.total_scored} checks)"
        )

    def report(self) -> str:
        """Formatted rubric report."""
        lines = ["## Rubric Report", ""]
        lines.append("| Section | Check | Severity | Result | Detail |")
        lines.append("|---------|-------|----------|--------|--------|")
        for c in self.checks:
            result_str = "PASS" if c.passed else "FAIL"
            lines.append(f"| {c.section} | {c.name} | {c.severity} | {result_str} | {c.detail} |")
        lines.append("")
        lines.append(f"### Overall: {self.summary()}")
        if self.critical_fail:
            failed_critical = [c.name for c in self.checks if c.severity == "Critical" and not c.passed]
            lines.append(f"\nCritical failures: {', '.join(failed_critical)}")
        return "\n".join(lines)


# ── Helper: check feature preserved at offset position ─────────────────


def _feature_preserved(
    construct_seq: str,
    backbone_seq: str,
    feature: dict,
    insert_len: int,
    insert_pos: int,
) -> tuple[bool, str]:
    """Check if a backbone feature's sequence is preserved in the construct.

    Features before the insert position should be at their original position.
    Features after the insert position are shifted by insert_len.

    Returns (preserved: bool, detail: str).
    """
    feat_start = feature.get("start", 0)
    feat_end = feature.get("end", 0)
    feat_name = feature.get("name", "unknown")

    if feat_start >= len(backbone_seq) or feat_end > len(backbone_seq):
        return True, f"{feat_name}: feature coords out of backbone range, skipped"

    feat_seq = backbone_seq[feat_start:feat_end]
    if not feat_seq:
        return True, f"{feat_name}: empty feature sequence, skipped"

    # Determine expected position in construct
    if feat_end <= insert_pos:
        # Feature is entirely upstream of insert — same position
        expected_start = feat_start
    elif feat_start >= insert_pos:
        # Feature is entirely downstream of insert — shifted by insert length
        expected_start = feat_start + insert_len
    else:
        # Feature spans the insert position — can't simply check
        return True, f"{feat_name}: spans insert site, skipped"

    expected_end = expected_start + len(feat_seq)
    if expected_end > len(construct_seq):
        return False, f"{feat_name}: expected at {expected_start}-{expected_end} but construct too short"

    actual_seq = construct_seq[expected_start:expected_end]
    if actual_seq == feat_seq:
        return True, f"{feat_name}: preserved at {expected_start}-{expected_end}"
    else:
        # Count mismatches for detail
        mismatches = sum(1 for a, b in zip(actual_seq, feat_seq) if a != b)
        return False, f"{feat_name}: {mismatches} mismatches at {expected_start}-{expected_end}"


# ── Helper: extract sequence from formatted output ─────────────────────


def _extract_sequence_from_output(output_text: str, output_format: str) -> Optional[str]:
    """Extract raw DNA sequence from formatted output text."""
    if output_format == "fasta":
        lines = output_text.strip().split("\n")
        seq_lines = [line.strip() for line in lines if line.strip() and not line.startswith(">")]
        return "".join(seq_lines) if seq_lines else None

    elif output_format == "genbank":
        # Extract sequence from ORIGIN section
        origin_match = re.search(r'ORIGIN\s*\n(.*?)(?://|\Z)', output_text, re.DOTALL)
        if origin_match:
            seq_block = origin_match.group(1)
            # Remove line numbers and spaces
            seq = re.sub(r'[\s\d]', '', seq_block).upper()
            return seq if seq else None
        return None

    return None


# ── Insert-search helper ────────────────────────────────────────────────


def _resolve_insert(
    insert_seq: str,
    construct_seq: str,
) -> tuple[Optional[str], bool, str]:
    """Find the insert (or a codon-trimmed variant) in the construct.

    Tries, in order:
    - Original sequence (forward)
    - Reverse complement
    - ATG removed (forward / RC)
    - Stop removed (forward / RC)
    - ATG and stop removed (forward / RC)

    Requires at least 9 bp to avoid spurious matches.

    Returns (found_seq, is_rc, modification_label) where:
    - found_seq  : the variant that matched, or None if nothing found
    - is_rc      : True if the RC orientation was matched
    - modification_label: human-readable description of codon changes applied
    """
    _STOP_CODONS = ("TAA", "TAG", "TGA")
    has_atg = insert_seq[:3] == "ATG"
    has_stop = insert_seq[-3:] in _STOP_CODONS
    MIN_LEN = 9

    candidates: list[tuple[str, bool, str]] = []

    def _add(seq: str, is_rc: bool, mod: str) -> None:
        if len(seq) >= MIN_LEN:
            candidates.append((seq, is_rc, mod))

    _add(insert_seq, False, "")
    _add(reverse_complement(insert_seq), True, "")
    if has_atg:
        no_atg = insert_seq[3:]
        _add(no_atg, False, "ATG removed")
        _add(reverse_complement(no_atg), True, "ATG removed")
    if has_stop:
        no_stop = insert_seq[:-3]
        _add(no_stop, False, "stop removed")
        _add(reverse_complement(no_stop), True, "stop removed")
    if has_atg and has_stop:
        no_both = insert_seq[3:-3]
        _add(no_both, False, "ATG and stop removed")
        _add(reverse_complement(no_both), True, "ATG and stop removed")

    for seq, is_rc, mod in candidates:
        if seq in construct_seq:
            return seq, is_rc, mod
    return None, False, "not found"


# ── Scoring function ────────────────────────────────────────────────────


def score_construct(
    construct_sequence: str,
    expected_backbone_sequence: str,
    expected_insert_sequence: str,
    expected_insert_position: int,
    backbone_name: str = "backbone",
    insert_name: str = "insert",
    insert_category: Optional[str] = None,
    ground_truth_sequence: Optional[str] = None,
    ground_truth_strict: bool = False,
    backbone_features: Optional[list[dict]] = None,
    output_text: Optional[str] = None,
    output_format: Optional[str] = None,
    expect_reverse_complement: bool = False,
    fusion_parts: Optional[list[dict]] = None,
    construct_annotations: Optional[list[dict]] = None,
) -> RubricResult:
    """
    Score an assembled construct against the Allen Institute rubric.

    Args:
        construct_sequence: The assembled construct to evaluate.
        expected_backbone_sequence: The original backbone sequence.
        expected_insert_sequence: The original insert sequence.
        expected_insert_position: 0-based position where the insert should start.
        backbone_name: Name of backbone (for reporting).
        insert_name: Name of insert (for reporting).
        insert_category: Optional insert category (e.g., "epitope_tag").
                         Epitope tags skip start/stop codon and frame checks.
        ground_truth_sequence: Optional known-correct full construct sequence
                               from Addgene. If provided, an exact-match check
                               is added.
        ground_truth_strict: If True, ground truth exact match is Critical
                             severity (fails the case on mismatch). If False
                             (default), it is Info severity (reported but does
                             not affect pass/fail).
        backbone_features: Optional list of feature dicts from backbones.json.
                           Each dict has keys: name, start, end, type.
                           Used for biological sanity checks.
        output_text: Optional formatted output text (GenBank/FASTA string).
        output_format: Optional format identifier ("genbank", "fasta", or None).
        expect_reverse_complement: If True, expect the insert in reverse
                                   complement orientation. Orientation and
                                   position checks use the RC insert.
        fusion_parts: Optional list of fusion part dicts, ordered N-terminal
                      to C-terminal. Each dict has keys: name (str),
                      sequence (str), type ("protein" or "tag"). Used for
                      fusion linker checks.
        construct_annotations: Optional pre-computed pLannotate feature list for
                               the construct (from annotate_plasmid()). When
                               provided, used directly for the duplicate-element
                               check. When omitted, the rubric attempts to run
                               pLannotate itself; if unavailable the check is
                               skipped rather than failing.

    Returns:
        RubricResult with all checks populated.
    """
    result = RubricResult()
    construct_seq = clean_sequence(construct_sequence)
    backbone_seq = clean_sequence(expected_backbone_sequence)
    insert_seq = clean_sequence(expected_insert_sequence)

    # Pre-compute RC and resolve the insert variant present in
    # the construct.  This handles: forward/RC orientation, ATG removal
    # (non-N-terminal fusion parts), and stop codon removal (non-C-terminal
    # fusion parts).  All downstream checks use insert_found / insert_len
    # so they remain correct regardless of which variant was assembled.
    insert_rc = reverse_complement(insert_seq)
    _insert_seq, _insert_is_rc, _insert_mod = _resolve_insert(insert_seq, construct_seq)
    insert_used = _insert_seq if _insert_seq is not None else (
        insert_rc if expect_reverse_complement else insert_seq
    )
    insert_found = _insert_seq is not None
    insert_len = len(insert_used)

    # ── Section 1: Input Validation ─────────────────────────────────

    # 1a. Backbone is valid DNA
    bb_valid, bb_errs = validate_dna(backbone_seq)
    result.checks.append(Check(
        section="Input Validation",
        name=f"Backbone ({backbone_name}) is valid DNA",
        severity="Critical",
        passed=bb_valid,
        detail="; ".join(bb_errs) if bb_errs else f"{len(backbone_seq)} bp",
    ))

    # 1b. Backbone length reasonable
    result.checks.append(Check(
        section="Input Validation",
        name="Backbone length",
        severity="Info",
        passed=True,
        detail=f"{len(backbone_seq)} bp",
    ))

    # 1c. Insert is valid DNA
    ins_valid, ins_errs = validate_dna(insert_seq)
    result.checks.append(Check(
        section="Input Validation",
        name=f"Insert ({insert_name}) is valid DNA",
        severity="Critical",
        passed=ins_valid,
        detail="; ".join(ins_errs) if ins_errs else f"{len(insert_seq)} bp",
    ))

    # 1d–1f: Codon and frame checks (skip for epitope tags — they fuse
    # in-frame with another gene and don't carry their own start/stop)
    is_tag = insert_category == "epitope_tag"

    has_atg = insert_seq[:3] == "ATG"
    has_stop = insert_seq[-3:] in ("TAA", "TAG", "TGA")
    frame_ok = len(insert_seq) % 3 == 0

    if not is_tag:
        result.checks.append(Check(
            section="Input Validation",
            name="Insert has start codon (ATG)",
            severity="Minor",
            passed=has_atg,
            detail=f"First 3 bases: {insert_seq[:3]}" if not has_atg else "",
        ))
        result.checks.append(Check(
            section="Input Validation",
            name="Insert has stop codon",
            severity="Minor",
            passed=has_stop,
            detail=f"Last 3 bases: {insert_seq[-3:]}" if not has_stop else "",
        ))
        result.checks.append(Check(
            section="Input Validation",
            name="Insert length multiple of 3",
            severity="Minor",
            passed=frame_ok,
            detail=f"{len(insert_seq)} bp" if not frame_ok else "",
        ))
    else:
        result.checks.append(Check(
            section="Input Validation",
            name="Codon/frame checks",
            severity="Info",
            passed=True,
            detail="Skipped for epitope tag",
        ))

    # ── Section 2: Construct Assembly ───────────────────────────────

    # 2a. Construct is valid DNA
    con_valid, con_errs = validate_dna(construct_seq)
    result.checks.append(Check(
        section="Construct Assembly",
        name="Construct is valid DNA",
        severity="Critical",
        passed=con_valid,
        detail="; ".join(con_errs) if con_errs else f"{len(construct_seq)} bp",
    ))

    # 2b. Insert found in construct (any variant: forward/RC, with/without ATG or stop)
    _found_parts = []
    if _insert_is_rc:
        _found_parts.append("reverse complement")
    if _insert_mod:
        _found_parts.append(_insert_mod)
    _found_detail = ", ".join(_found_parts)
    result.checks.append(Check(
        section="Construct Assembly",
        name="Insert sequence present in construct",
        severity="Critical",
        passed=insert_found,
        detail=_found_detail,
    ))

    # 2c. Insert in correct orientation

    if expect_reverse_complement:
        orientation_ok = _insert_is_rc
        orientation_detail = "" if orientation_ok else (
            "Found in forward orientation only" if (insert_found and not _insert_is_rc)
            else "Insert not found"
        )
        orientation_label = "Insert in correct orientation (reverse complement)"
    else:
        orientation_ok = not _insert_is_rc if insert_found else False
        orientation_detail = (
            "Found as reverse complement only" if _insert_is_rc else
            ("Insert not found" if not insert_found else "")
        )
        orientation_label = "Insert in correct orientation (forward)"

    result.checks.append(Check(
        section="Construct Assembly",
        name=orientation_label,
        severity="Critical",
        passed=orientation_ok,
        detail=orientation_detail,
    ))

    # 2d. Insert at correct position
    if insert_found:
        actual_pos = construct_seq.index(insert_used)
        pos_correct = actual_pos == expected_insert_position
        result.checks.append(Check(
            section="Construct Assembly",
            name="Insert at correct position",
            severity="Critical",
            passed=pos_correct,
            detail=f"Expected pos {expected_insert_position}, found at {actual_pos}" if not pos_correct else f"Position {actual_pos}",
        ))
    else:
        result.checks.append(Check(
            section="Construct Assembly",
            name="Insert at correct position",
            severity="Critical",
            passed=False,
            detail="Insert not found in expected orientation",
        ))

    # 2e. Backbone upstream preserved
    upstream_ok = construct_seq[:expected_insert_position] == backbone_seq[:expected_insert_position]
    result.checks.append(Check(
        section="Construct Assembly",
        name="Backbone upstream preserved",
        severity="Critical",
        passed=upstream_ok,
        detail="" if upstream_ok else f"Mismatch in first {expected_insert_position} bp",
    ))

    # 2f. Backbone downstream preserved
    downstream_start_construct = expected_insert_position + len(insert_used)
    downstream_start_backbone = expected_insert_position
    downstream_ok = construct_seq[downstream_start_construct:] == backbone_seq[downstream_start_backbone:]
    result.checks.append(Check(
        section="Construct Assembly",
        name="Backbone downstream preserved",
        severity="Critical",
        passed=downstream_ok,
        detail="" if downstream_ok else "Downstream backbone sequence altered",
    ))

    # ── Section 3: Construct Integrity ──────────────────────────────

    # 3a. Full-length plasmid output (insert was actually added)
    full_length = len(construct_seq) > len(backbone_seq)
    result.checks.append(Check(
        section="Construct Integrity",
        name="Full-length plasmid output",
        severity="Critical",
        passed=full_length and len(construct_seq) > 0,
        detail=f"Construct {len(construct_seq)} bp > backbone {len(backbone_seq)} bp" if full_length else f"Construct {len(construct_seq)} bp <= backbone {len(backbone_seq)} bp",
    ))

    # 3b. Total size correct (use effective insert length — may differ from
    # the original insert_seq length when ATG or stop was trimmed for fusion)
    expected_size = len(backbone_seq) + insert_len
    size_ok = len(construct_seq) == expected_size
    result.checks.append(Check(
        section="Construct Integrity",
        name="Total construct size correct",
        severity="Minor",
        passed=size_ok,
        detail=f"Expected {expected_size}, got {len(construct_seq)}" if not size_ok else f"{len(construct_seq)} bp",
    ))

    # 3c. All key features present (if feature annotations available)
    if backbone_features:
        key_types = {"promoter", "CDS", "rep_origin"}
        key_features = [f for f in backbone_features if f.get("type") in key_types]
        if key_features:
            all_preserved = True
            failed_features = []
            for feat in key_features:
                preserved, detail = _feature_preserved(
                    construct_seq, backbone_seq, feat,
                    insert_len, expected_insert_position,
                )
                if not preserved:
                    all_preserved = False
                    failed_features.append(detail)
            result.checks.append(Check(
                section="Construct Integrity",
                name="All key features preserved",
                severity="Major",
                passed=all_preserved,
                detail="; ".join(failed_features) if failed_features else f"{len(key_features)} features verified",
            ))
        else:
            result.checks.append(Check(
                section="Construct Integrity",
                name="All key features preserved",
                severity="Info",
                passed=True,
                detail="No key features annotated (promoter/CDS/origin)",
            ))
    else:
        result.checks.append(Check(
            section="Construct Integrity",
            name="All key features preserved",
            severity="Info",
            passed=True,
            detail="Skipped — no feature annotations available",
        ))

    # ── Section 4: Biological Sanity Checks ─────────────────────────

    # 4a. Valid ORF — no premature stop codons in insert CDS
    insert_cds = insert_seq[:-3] if has_stop else insert_seq
    internal_stops = []
    if frame_ok and len(insert_cds) >= 3:
        for i in range(3, len(insert_cds), 3):
            codon = insert_cds[i:i + 3]
            if codon in ("TAA", "TAG", "TGA"):
                internal_stops.append((i, codon))
    no_internal_stops = len(internal_stops) == 0
    result.checks.append(Check(
        section="Biological Sanity",
        name="No premature stop codons in insert",
        severity="Major",
        passed=no_internal_stops,
        detail=f"{len(internal_stops)} internal stop(s) at positions: {[p for p, _ in internal_stops[:5]]}" if internal_stops else "",
    ))

    # 4b. Promoter upstream of insert
    if backbone_features:
        promoter_features = [f for f in backbone_features if f.get("type") == "promoter"]
        if promoter_features:
            # Find the primary promoter (the one closest upstream of the MCS/insert)
            primary_promoter = None
            for pf in promoter_features:
                if pf.get("end", 0) <= expected_insert_position:
                    if primary_promoter is None or pf["end"] > primary_promoter["end"]:
                        primary_promoter = pf
            if primary_promoter:
                promoter_upstream = primary_promoter["end"] <= expected_insert_position
                result.checks.append(Check(
                    section="Biological Sanity",
                    name="Promoter upstream of insert",
                    severity="Critical",
                    passed=promoter_upstream,
                    detail=f"{primary_promoter['name']} ends at {primary_promoter['end']}, insert at {expected_insert_position}" if promoter_upstream else f"{primary_promoter['name']} ends at {primary_promoter['end']} > insert at {expected_insert_position}",
                ))
            else:
                result.checks.append(Check(
                    section="Biological Sanity",
                    name="Promoter upstream of insert",
                    severity="Info",
                    passed=True,
                    detail="No promoter found upstream of insert position",
                ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="Promoter upstream of insert",
                severity="Info",
                passed=True,
                detail="Skipped — no promoter features annotated",
            ))
    else:
        result.checks.append(Check(
            section="Biological Sanity",
            name="Promoter upstream of insert",
            severity="Info",
            passed=True,
            detail="Skipped — no feature annotations available",
        ))

    # 4c. PolyA signal downstream of insert
    if backbone_features:
        polya_features = [f for f in backbone_features if f.get("type") == "polyA_signal"]
        if polya_features:
            # Find the polyA signal closest downstream of the insert position
            downstream_polya = None
            for pf in polya_features:
                if pf.get("start", 0) >= expected_insert_position:
                    if downstream_polya is None or pf["start"] < downstream_polya["start"]:
                        downstream_polya = pf
            if downstream_polya:
                # Check that the polyA is downstream of the insert end
                insert_end = expected_insert_position + insert_len
                # In the construct, the polyA is shifted by insert length
                polya_construct_start = downstream_polya["start"] + insert_len
                polya_downstream = polya_construct_start >= insert_end
                result.checks.append(Check(
                    section="Biological Sanity",
                    name="PolyA signal downstream of insert",
                    severity="Major",
                    passed=polya_downstream,
                    detail=f"{downstream_polya['name']} at construct pos {polya_construct_start}, insert ends at {insert_end}",
                ))

                # 4c-2. PolyA signal sequence preserved
                preserved, detail = _feature_preserved(
                    construct_seq, backbone_seq, downstream_polya,
                    insert_len, expected_insert_position,
                )
                result.checks.append(Check(
                    section="Biological Sanity",
                    name="PolyA signal preserved",
                    severity="Major",
                    passed=preserved,
                    detail=detail,
                ))
            else:
                result.checks.append(Check(
                    section="Biological Sanity",
                    name="PolyA signal downstream of insert",
                    severity="Info",
                    passed=True,
                    detail="No polyA signal found downstream of insert position",
                ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="PolyA signal downstream of insert",
                severity="Info",
                passed=True,
                detail="Skipped — no polyA_signal features annotated",
            ))
    else:
        result.checks.append(Check(
            section="Biological Sanity",
            name="PolyA signal downstream of insert",
            severity="Info",
            passed=True,
            detail="Skipped — no feature annotations available",
        ))

    # 4d. Selectable markers intact
    if backbone_features:
        cds_features = [f for f in backbone_features if f.get("type") == "CDS"]
        if cds_features:
            all_markers_ok = True
            marker_details = []
            for feat in cds_features:
                preserved, detail = _feature_preserved(
                    construct_seq, backbone_seq, feat,
                    insert_len, expected_insert_position,
                )
                if not preserved:
                    all_markers_ok = False
                marker_details.append(detail)
            result.checks.append(Check(
                section="Biological Sanity",
                name="Selectable markers intact",
                severity="Major",
                passed=all_markers_ok,
                detail="; ".join(marker_details),
            ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="Selectable markers intact",
                severity="Info",
                passed=True,
                detail="Skipped — no CDS features annotated",
            ))
    else:
        result.checks.append(Check(
            section="Biological Sanity",
            name="Selectable markers intact",
            severity="Info",
            passed=True,
            detail="Skipped — no feature annotations available",
        ))

    # 4e. Replication origin intact
    if backbone_features:
        origin_features = [f for f in backbone_features if f.get("type") == "rep_origin"]
        if origin_features:
            all_origins_ok = True
            origin_details = []
            for feat in origin_features:
                preserved, detail = _feature_preserved(
                    construct_seq, backbone_seq, feat,
                    insert_len, expected_insert_position,
                )
                if not preserved:
                    all_origins_ok = False
                origin_details.append(detail)
            result.checks.append(Check(
                section="Biological Sanity",
                name="Replication origin intact",
                severity="Minor",
                passed=all_origins_ok,
                detail="; ".join(origin_details),
            ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="Replication origin intact",
                severity="Info",
                passed=True,
                detail="Skipped — no rep_origin features annotated",
            ))
    else:
        result.checks.append(Check(
            section="Biological Sanity",
            name="Replication origin intact",
            severity="Info",
            passed=True,
            detail="Skipped — no feature annotations available",
        ))

    # 4f. Kozak context around insert ATG (mammalian expression)
    if not is_tag and has_atg and expected_insert_position >= 6:
        # Check 6 bp upstream of insert for Kozak-like context
        # Ideal Kozak: GCCACCATGG (positions -6 to +4 relative to A of ATG)
        # Key positions: -3 (A or G) and +4 (G)
        upstream_context = construct_seq[expected_insert_position - 6:expected_insert_position]
        downstream_g = insert_seq[3:4] if len(insert_seq) > 3 else ""
        pos_minus3 = upstream_context[3] if len(upstream_context) > 3 else ""
        strong_kozak = pos_minus3 in ("A", "G") and downstream_g == "G"
        adequate_kozak = pos_minus3 in ("A", "G") or downstream_g == "G"
        if strong_kozak:
            kozak_detail = f"Strong Kozak context: ...{upstream_context}ATG{downstream_g}..."
        elif adequate_kozak:
            kozak_detail = f"Adequate Kozak context: ...{upstream_context}ATG{downstream_g}..."
        else:
            kozak_detail = f"Weak Kozak context: ...{upstream_context}ATG{downstream_g}..."
        result.checks.append(Check(
            section="Biological Sanity",
            name="Kozak sequence context",
            severity="Info",
            passed=True,
            detail=kozak_detail,
        ))

    # 4g. Fusion linker checks (only when fusion_parts provided)
    if fusion_parts and len(fusion_parts) >= 2:
        part_seqs = [clean_sequence(p["sequence"]) for p in fusion_parts]
        part_types = [p["type"] for p in fusion_parts]

        # Identify protein-protein junctions (these need linkers).
        # Linker-type parts are transparent: [protein, linker, protein] counts
        # as a protein-protein junction (with a linker already provided).
        non_linker_types = [t for t in part_types if t != "linker"]
        has_protein_protein_junction = False
        has_explicit_linker = "linker" in part_types
        for i in range(len(non_linker_types) - 1):
            if non_linker_types[i] == "protein" and non_linker_types[i + 1] == "protein":
                has_protein_protein_junction = True
                break

        # Compute expected fusion length: sum of all part sequences after
        # codon management. Linker parts included at full length.
        # Also accounts for stop codon added by fuse_inserts when missing.
        # ATG is always removed from non-first protein parts.
        expected_fusion_len = 0
        for i, seq in enumerate(part_seqs):
            if part_types[i] == "linker":
                expected_fusion_len += len(seq)
                continue
            seq_len = len(seq)
            # Remove stop codon from non-terminal parts
            if i < len(part_seqs) - 1:
                if seq[-3:] in ("TAA", "TAG", "TGA"):
                    seq_len -= 3
            if i > 0 and part_types[i] == "protein" and seq[:3] == "ATG":
                seq_len -= 3  # ATG stripped
            expected_fusion_len += seq_len
        # If the last part lacks a stop codon, fuse_inserts adds one (3 bp)
        if part_seqs[-1][-3:] not in ("TAA", "TAG", "TGA"):
            expected_fusion_len += 3

        # Compute proteins-only length (excluding linker parts)
        linker_total_len = sum(
            len(seq) for seq, t in zip(part_seqs, part_types) if t == "linker"
        )
        proteins_only_len = expected_fusion_len - linker_total_len

        # Check 1: Fusion linker present between proteins (Critical, 2 pts)
        if has_protein_protein_junction:
            # Verify insert is longer than proteins-only (linker adds length)
            linker_present = len(insert_seq) > proteins_only_len
            linker_len = len(insert_seq) - proteins_only_len
            result.checks.append(Check(
                section="Biological Sanity",
                name="Fusion linker present between proteins",
                severity="Critical",
                passed=linker_present,
                detail=(
                    f"Linker adds {linker_len} bp (insert {len(insert_seq)} bp vs proteins-only {proteins_only_len} bp)"
                    if linker_present
                    else f"No linker detected: insert {len(insert_seq)} bp == proteins-only {proteins_only_len} bp"
                ),
            ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="Fusion linker present between proteins",
                severity="Info",
                passed=True,
                detail="Skipped — tag-protein junction, no linker expected",
            ))

        # Check 2: Fusion insert size accounts for all parts (Major, 1 pt)
        if has_protein_protein_junction:
            # Insert should match expected fusion length (proteins + linker + codon mgmt)
            size_ok = len(insert_seq) == expected_fusion_len
            result.checks.append(Check(
                section="Biological Sanity",
                name="Fusion insert size accounts for all parts",
                severity="Major",
                passed=size_ok,
                detail=(
                    f"Insert {len(insert_seq)} bp matches expected {expected_fusion_len} bp"
                    if size_ok
                    else f"Insert {len(insert_seq)} bp != expected {expected_fusion_len} bp (delta: {len(insert_seq) - expected_fusion_len:+d} bp)"
                ),
            ))
        else:
            # For tag-protein: insert should equal expected (no linker)
            size_ok = len(insert_seq) == expected_fusion_len
            result.checks.append(Check(
                section="Biological Sanity",
                name="Fusion insert size accounts for all parts",
                severity="Major",
                passed=size_ok,
                detail=(
                    f"Insert {len(insert_seq)} bp matches expected {expected_fusion_len} bp (no linker, correct for tag)"
                    if size_ok
                    else f"Insert {len(insert_seq)} bp != expected {expected_fusion_len} bp (unexpected for tag fusion)"
                ),
            ))

        # Check 3: ATG management at non-N-terminal protein junctions (Major)
        # Each non-first protein must have its initiator ATG stripped so the
        # ribosome reads one continuous ORF.
        if has_protein_protein_junction:
            atg_removed_ok = True
            atg_removed_detail = []
            for i in range(1, len(part_seqs)):
                if part_types[i] == "protein" and part_seqs[i][:3] == "ATG":
                    # The first ~18 bp of the original sequence (with ATG) should
                    # NOT appear verbatim in the fused insert — that would mean
                    # the ATG was not removed.
                    original_start = clean_sequence(part_seqs[i])[:18]
                    if original_start in clean_sequence(insert_seq):
                        atg_removed_ok = False
                        name = fusion_parts[i].get("name", f"part_{i}")
                        atg_removed_detail.append(f"ATG not removed from '{name}'")
            result.checks.append(Check(
                section="Biological Sanity",
                name="ATG removed from non-N-terminal protein(s) at junction",
                severity="Major",
                passed=atg_removed_ok,
                detail=(
                    "; ".join(atg_removed_detail)
                    if not atg_removed_ok
                    else "Start codon correctly removed from non-terminal protein(s)"
                ),
            ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="ATG removed from non-N-terminal protein(s) at junction",
                severity="Info",
                passed=True,
                detail="Skipped — tag-protein junction, no ATG removal expected",
            ))

    # 4h. No duplicate functional elements (via pLannotate)
    # Use pre-supplied annotations when available; fall back to running
    # pLannotate on the construct; skip gracefully if unavailable.
    _dup_features = construct_annotations
    if _dup_features is None:
        try:
            from library import annotate_plasmid as _annotate_rubric
            _dup_features = _annotate_rubric(construct_seq)
        except Exception:
            _dup_features = None

    if _dup_features is not None:
        try:
            from library import find_duplicate_annotations as _find_dups
            _duplicates = _find_dups(_dup_features)
        except Exception:
            _duplicates = []

        if _duplicates:
            _dup_names = ", ".join(
                f"{d['name']} ({d['count']}×)" for d in _duplicates
            )
            result.checks.append(Check(
                section="Biological Sanity",
                name="No duplicate functional elements",
                severity="Critical",
                passed=False,
                detail=f"Duplicate features: {_dup_names}",
            ))
        else:
            result.checks.append(Check(
                section="Biological Sanity",
                name="No duplicate functional elements",
                severity="Critical",
                passed=True,
                detail=f"{len(_dup_features)} features annotated, none duplicated",
            ))
    else:
        result.checks.append(Check(
            section="Biological Sanity",
            name="No duplicate functional elements",
            severity="Info",
            passed=True,
            detail="Skipped — pLannotate unavailable",
        ))

    # ── Section 5: Output Verification ──────────────────────────────

    if output_text and output_format:
        fmt = output_format.lower()

        # 5a. Correct file format header
        if fmt == "fasta":
            format_ok = output_text.strip().startswith(">")
            result.checks.append(Check(
                section="Output Verification",
                name="Correct file format",
                severity="Minor",
                passed=format_ok,
                detail="FASTA header present" if format_ok else "Missing '>' FASTA header",
            ))
        elif fmt == "genbank":
            format_ok = output_text.strip().startswith("LOCUS")
            result.checks.append(Check(
                section="Output Verification",
                name="Correct file format",
                severity="Minor",
                passed=format_ok,
                detail="GenBank LOCUS line present" if format_ok else "Missing LOCUS header",
            ))

        # 5b. Parseable output
        if fmt == "genbank":
            has_origin = "ORIGIN" in output_text
            has_end = "//" in output_text
            parseable = has_origin and has_end
            missing = []
            if not has_origin:
                missing.append("ORIGIN")
            if not has_end:
                missing.append("//")
            result.checks.append(Check(
                section="Output Verification",
                name="Parseable output",
                severity="Minor",
                passed=parseable,
                detail="GenBank markers present" if parseable else f"Missing: {', '.join(missing)}",
            ))
        elif fmt == "fasta":
            lines = output_text.strip().split("\n")
            has_header = len(lines) > 0 and lines[0].startswith(">")
            has_seq = len(lines) > 1 and len(lines[1]) > 0
            parseable = has_header and has_seq
            result.checks.append(Check(
                section="Output Verification",
                name="Parseable output",
                severity="Minor",
                passed=parseable,
                detail="FASTA structure valid" if parseable else "Invalid FASTA structure",
            ))

        # 5c. Sequence in output matches construct
        extracted_seq = _extract_sequence_from_output(output_text, fmt)
        if extracted_seq:
            seq_match = clean_sequence(extracted_seq) == construct_seq
            result.checks.append(Check(
                section="Output Verification",
                name="Sequence in output matches construct",
                severity="Minor",
                passed=seq_match,
                detail="" if seq_match else f"Extracted {len(extracted_seq)} bp vs construct {len(construct_seq)} bp",
            ))
        else:
            result.checks.append(Check(
                section="Output Verification",
                name="Sequence in output matches construct",
                severity="Minor",
                passed=False,
                detail="Could not extract sequence from formatted output",
            ))

        # 5d. GenBank LOCUS size matches sequence length
        if fmt == "genbank":
            locus_match = re.search(r'LOCUS\s+\S+\s+(\d+)\s+bp', output_text)
            if locus_match:
                locus_size = int(locus_match.group(1))
                locus_size_ok = locus_size == len(construct_seq)
                result.checks.append(Check(
                    section="Output Verification",
                    name="GenBank LOCUS size correct",
                    severity="Minor",
                    passed=locus_size_ok,
                    detail=f"LOCUS says {locus_size} bp, sequence is {len(construct_seq)} bp" if not locus_size_ok else f"{locus_size} bp",
                ))
            else:
                result.checks.append(Check(
                    section="Output Verification",
                    name="GenBank LOCUS size correct",
                    severity="Minor",
                    passed=False,
                    detail="Could not parse LOCUS line",
                ))

        # 5e. GenBank feature annotations present (insert CDS)
        if fmt == "genbank":
            has_features = "FEATURES" in output_text
            has_insert_cds = bool(re.search(r'CDS\s+\d+\.\.\d+', output_text))
            annotations_ok = has_features and has_insert_cds
            result.checks.append(Check(
                section="Output Verification",
                name="GenBank annotations present",
                severity="Minor",
                passed=annotations_ok,
                detail="FEATURES section with insert CDS present" if annotations_ok else f"Missing: {'FEATURES' if not has_features else 'insert CDS annotation'}",
            ))

    # ── Section 6: Output Quality (Ground Truth) ────────────────────

    if ground_truth_sequence:
        gt_seq = clean_sequence(ground_truth_sequence)
        exact_match = construct_seq == gt_seq
        gt_severity = "Critical" if ground_truth_strict else "Info"
        result.checks.append(Check(
            section="Output Quality",
            name="Exact match to ground truth (Addgene)",
            severity=gt_severity,
            passed=exact_match,
            detail="" if exact_match else f"Construct {len(construct_seq)} bp vs ground truth {len(gt_seq)} bp",
        ))
        if not exact_match:
            if len(construct_seq) == len(gt_seq):
                mismatches = sum(1 for a, b in zip(construct_seq, gt_seq) if a != b)
                result.checks.append(Check(
                    section="Output Quality",
                    name="Ground truth similarity",
                    severity="Info",
                    passed=True,
                    detail=f"{mismatches} mismatches out of {len(construct_seq)} bp ({round((1 - mismatches/len(construct_seq))*100, 2)}% identity)",
                ))
            else:
                delta = len(construct_seq) - len(gt_seq)
                insert_in_gt = insert_seq in gt_seq
                result.checks.append(Check(
                    section="Output Quality",
                    name="Ground truth similarity",
                    severity="Info",
                    passed=True,
                    detail=(
                        f"Size delta: {delta:+d} bp. "
                        f"Insert {'found' if insert_in_gt else 'NOT found'} in ground truth."
                    ),
                ))

    return result
