"""Design Confidence Score — predicts anticipated construct success.

Composite score (0-100) from weighted sub-checks covering:
- Cryptic regulatory signals (polyA, splice sites)
- Expression optimality (CAI, Kozak, GC content)
- Structural concerns (fusion linker adequacy, repeat runs)
- Architecture (promoter count)

All checks are sequence-level analysis. No external API calls, no DNA
synthesis. Tier B (protein-level: disorder, TM, signal peptides) is
deferred to a later phase.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# Dual-import: src/tools.py imports this as a package (`src.confidence`),
# tests/ and app/app.py import it flat (`confidence`) via sys.path hack.
try:
    from .codon_tables import HUMAN_CODON_W
except ImportError:
    from codon_tables import HUMAN_CODON_W


@dataclass
class ConfidenceCheck:
    name: str
    category: str  # "cryptic_signals", "expression", "structural", "architecture"
    severity: str  # "critical", "warning", "info"
    passed: bool
    score_delta: int  # 0 if passed, negative if failed
    message: str
    position: Optional[int] = None


@dataclass
class ConfidenceReport:
    overall_score: int  # 0-100, clamped
    checks: list[ConfidenceCheck] = field(default_factory=list)
    summary: str = ""
    recommendation: str = ""


# ── Individual checks ──────────────────────────────────────────────────


def check_cryptic_polya(seq: str) -> ConfidenceCheck:
    """Scan for cryptic polyA signals (AATAAA, ATTAAA) in the insert body.

    A polyA signal >100bp before the true 3' end can cause premature
    transcript termination -> truncated protein.
    """
    seq = seq.upper()
    # Exclude the last 150bp where a real polyA signal is expected/harmless
    search_region = seq[:-150] if len(seq) > 150 else ""
    matches = [m.start() for m in re.finditer(r"AATAAA|ATTAAA", search_region)]
    if matches:
        return ConfidenceCheck(
            name="Cryptic polyA signal",
            category="cryptic_signals",
            severity="critical",
            passed=False,
            score_delta=-15,
            message=f"Found {len(matches)} cryptic polyA signal(s) at position(s) {matches[:3]}. Risk of premature transcript termination.",
            position=matches[0],
        )
    return ConfidenceCheck(
        name="Cryptic polyA signal",
        category="cryptic_signals",
        severity="info",
        passed=True,
        score_delta=0,
        message="No cryptic polyA signals detected in insert body.",
    )


def check_cryptic_splice(seq: str) -> ConfidenceCheck:
    """Scan for cryptic splice donor/acceptor sites.

    Strong 5' donor: consensus AG|GTAAGT or AG|GTGAGT -> simplified as [AC]AGGT[AG]AGT
    Strong 3' acceptor: poly-pyrimidine + CAG|G -> simplified as [CT]{6,}[ACGT]CAGG
    """
    seq = seq.upper()
    donors = [m.start() for m in re.finditer(r"[AC]AGGT[AG]AGT", seq)]
    acceptors = [m.start() for m in re.finditer(r"[CT]{6,}[ACGT]CAGG", seq)]

    total = len(donors) + len(acceptors)
    if total > 0:
        positions = sorted(donors + acceptors)[:3]
        penalty = max(-8 * total, -20)  # cap at -20
        return ConfidenceCheck(
            name="Cryptic splice site",
            category="cryptic_signals",
            severity="warning",
            passed=False,
            score_delta=penalty,
            message=f"Found {len(donors)} potential splice donor(s) and {len(acceptors)} acceptor(s) at position(s) {positions}. May cause aberrant splicing in mammalian cells.",
            position=positions[0] if positions else None,
        )
    return ConfidenceCheck(
        name="Cryptic splice site",
        category="cryptic_signals",
        severity="info",
        passed=True,
        score_delta=0,
        message="No strong cryptic splice sites detected.",
    )


def compute_cai(seq: str) -> float:
    """Compute Codon Adaptation Index (CAI) for human expression.

    CAI = geometric mean of w_i for each codon in the sequence.
    Range [0, 1]. >0.8 is good, <0.6 is suboptimal.
    """
    seq = seq.upper()
    seq = seq[: len(seq) - len(seq) % 3]
    if len(seq) < 3:
        return 0.0
    log_sum = 0.0
    n = 0
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        w = HUMAN_CODON_W.get(codon)
        if w is None or w <= 0:
            continue  # skip stops / unknown codons
        log_sum += math.log(w)
        n += 1
    if n == 0:
        return 0.0
    return math.exp(log_sum / n)


def check_cai(seq: str) -> ConfidenceCheck:
    """Assess Codon Adaptation Index for human expression."""
    cai = compute_cai(seq)
    if cai >= 0.8:
        return ConfidenceCheck(
            name="Codon Adaptation Index",
            category="expression",
            severity="info",
            passed=True,
            score_delta=0,
            message=f"CAI {cai:.2f} — good codon usage for human expression.",
        )
    elif cai >= 0.6:
        return ConfidenceCheck(
            name="Codon Adaptation Index",
            category="expression",
            severity="info",
            passed=True,
            score_delta=-3,
            message=f"CAI {cai:.2f} — moderate codon usage. Codon optimization may improve expression.",
        )
    else:
        return ConfidenceCheck(
            name="Codon Adaptation Index",
            category="expression",
            severity="warning",
            passed=False,
            score_delta=-10,
            message=f"CAI {cai:.2f} — suboptimal codon usage for human expression. Strongly recommend codon optimization.",
        )


def check_gc_content(seq: str) -> ConfidenceCheck:
    """Check GC content is within the 35-65% optimal range."""
    seq = seq.upper()
    if not seq:
        return ConfidenceCheck(
            name="GC content",
            category="expression",
            severity="info",
            passed=True,
            score_delta=0,
            message="Empty sequence.",
        )
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    pct = gc * 100
    if 35 <= pct <= 65:
        return ConfidenceCheck(
            name="GC content",
            category="expression",
            severity="info",
            passed=True,
            score_delta=0,
            message=f"GC content {pct:.1f}% — within optimal range (35-65%).",
        )
    elif pct < 35:
        return ConfidenceCheck(
            name="GC content",
            category="expression",
            severity="warning",
            passed=False,
            score_delta=-5,
            message=f"GC content {pct:.1f}% — below optimal (35%). May reduce mRNA stability.",
        )
    else:
        return ConfidenceCheck(
            name="GC content",
            category="expression",
            severity="warning",
            passed=False,
            score_delta=-5,
            message=f"GC content {pct:.1f}% — above optimal (65%). May cause synthesis/PCR difficulties.",
        )


def check_kozak(seq: str) -> ConfidenceCheck:
    """Check Kozak consensus around the first ATG.

    Strong Kozak: (gcc)gccRccATGG — positions -3 (R=A/G) and +4 (G) matter most.
    """
    seq = seq.upper()
    atg = seq.find("ATG")
    if atg < 0:
        return ConfidenceCheck(
            name="Kozak context",
            category="expression",
            severity="warning",
            passed=False,
            score_delta=-5,
            message="No ATG found — cannot assess Kozak context.",
        )
    minus3 = seq[atg - 3] if atg >= 3 else ""
    plus4 = seq[atg + 3] if atg + 3 < len(seq) else ""

    context = seq[max(0, atg - 6) : atg + 4]

    strong = (minus3 in "AG") and (plus4 == "G")
    moderate = (minus3 in "AG") or (plus4 == "G")

    if strong:
        return ConfidenceCheck(
            name="Kozak context",
            category="expression",
            severity="info",
            passed=True,
            score_delta=0,
            message=f"Strong Kozak context: ...{context}... (−3={minus3}, +4={plus4}).",
        )
    elif moderate:
        return ConfidenceCheck(
            name="Kozak context",
            category="expression",
            severity="info",
            passed=True,
            score_delta=-2,
            message=f"Moderate Kozak context: ...{context}... (−3={minus3}, +4={plus4}). Consider optimizing to GCCACCATGG.",
        )
    else:
        return ConfidenceCheck(
            name="Kozak context",
            category="expression",
            severity="warning",
            passed=False,
            score_delta=-5,
            message=f"Weak Kozak context: ...{context}... (−3={minus3 or '?'}, +4={plus4 or '?'}). Recommend GCCACCATGG for strong translation initiation.",
        )


def check_repeat_runs(seq: str) -> ConfidenceCheck:
    """Find long single-base runs (>8) — cause synthesis problems."""
    seq = seq.upper()
    runs = [
        (m.start(), m.group())
        for m in re.finditer(r"(A{9,}|T{9,}|G{9,}|C{9,})", seq)
    ]
    if runs:
        worst = max(runs, key=lambda r: len(r[1]))
        return ConfidenceCheck(
            name="Repeat runs",
            category="structural",
            severity="warning",
            passed=False,
            score_delta=-3,
            message=f"Found {len(runs)} long single-base run(s). Longest: {len(worst[1])}bp at position {worst[0]}. May cause DNA synthesis failures.",
            position=worst[0],
        )
    return ConfidenceCheck(
        name="Repeat runs",
        category="structural",
        severity="info",
        passed=True,
        score_delta=0,
        message="No problematic single-base runs (>8bp).",
    )


def check_fusion_linker(fusion_parts: Optional[list[dict]]) -> ConfidenceCheck:
    """Check if fusion linker is adequate for multi-domain proteins.

    fusion_parts: list of {"name": str, "aa_length": int, "is_linker": bool}
    A fusion of two domains each >=100aa with a linker <5aa risks misfolding.
    """
    if not fusion_parts or len(fusion_parts) < 2:
        return ConfidenceCheck(
            name="Fusion linker adequacy",
            category="structural",
            severity="info",
            passed=True,
            score_delta=0,
            message="Not a fusion construct (single insert).",
        )

    domains = [p for p in fusion_parts if not p.get("is_linker")]
    linkers = [p for p in fusion_parts if p.get("is_linker")]

    if len(domains) < 2:
        return ConfidenceCheck(
            name="Fusion linker adequacy",
            category="structural",
            severity="info",
            passed=True,
            score_delta=0,
            message="Single protein domain — no linker needed.",
        )

    large_domains = [d for d in domains if d.get("aa_length", 0) >= 100]
    shortest_linker = min((l.get("aa_length", 0) for l in linkers), default=0)

    if len(large_domains) >= 2 and shortest_linker < 5:
        return ConfidenceCheck(
            name="Fusion linker adequacy",
            category="structural",
            severity="warning",
            passed=False,
            score_delta=-8,
            message=(
                f"Fusion of {len(large_domains)} large domain(s) (≥100aa) with "
                f"linker only {shortest_linker}aa. Risk of steric interference. "
                f"Recommend ≥15aa flexible linker (e.g., (GGGGS)×3 or ×4)."
            ),
        )
    elif len(large_domains) >= 2 and shortest_linker < 15:
        return ConfidenceCheck(
            name="Fusion linker adequacy",
            category="structural",
            severity="info",
            passed=True,
            score_delta=-2,
            message=(
                f"Fusion linker is {shortest_linker}aa — adequate but short "
                f"for {len(large_domains)} large domains. Consider (GGGGS)×4 (20aa)."
            ),
        )
    return ConfidenceCheck(
        name="Fusion linker adequacy",
        category="structural",
        severity="info",
        passed=True,
        score_delta=0,
        message=f"Fusion linker {shortest_linker}aa — adequate for domain folding.",
    )


def check_promoter_count(backbone: Optional[dict]) -> ConfidenceCheck:
    """Count promoter features in the backbone. >2 -> recombination risk."""
    if not backbone:
        return ConfidenceCheck(
            name="Promoter count",
            category="architecture",
            severity="info",
            passed=True,
            score_delta=0,
            message="No backbone metadata — promoter count not assessed.",
        )
    features = backbone.get("features", []) or []
    promoters = [
        f
        for f in features
        if "promoter" in str(f.get("type", "")).lower()
        or "promoter" in str(f.get("name", "")).lower()
    ]
    n = len(promoters)
    if n <= 2:
        return ConfidenceCheck(
            name="Promoter count",
            category="architecture",
            severity="info",
            passed=True,
            score_delta=0,
            message=f"{n} promoter(s) in backbone — normal.",
        )
    else:
        names = [f.get("name", "?") for f in promoters]
        return ConfidenceCheck(
            name="Promoter count",
            category="architecture",
            severity="warning",
            passed=False,
            score_delta=-5,
            message=(
                f"{n} promoters in backbone ({', '.join(names)}). "
                f"Multiple copies of the same promoter can cause "
                f"recombination and plasmid instability."
            ),
        )


# ── Composite scorer ───────────────────────────────────────────────────


def compute_confidence(
    insert_seq: str,
    backbone: Optional[dict] = None,
    fusion_parts: Optional[list[dict]] = None,
) -> ConfidenceReport:
    """Run all Tier-A confidence checks and compute a composite score.

    Args:
        insert_seq: The insert/CDS DNA sequence to analyze
        backbone: Backbone metadata dict (with 'features' list) for
                  promoter-count check. Optional.
        fusion_parts: For fusion constructs, list of parts with
                      {"name", "aa_length", "is_linker"}. Optional.

    Returns:
        ConfidenceReport with overall 0-100 score + individual check results.
    """
    insert_seq = insert_seq.upper().replace(" ", "").replace("\n", "")

    checks = [
        check_cryptic_polya(insert_seq),
        check_cryptic_splice(insert_seq),
        check_cai(insert_seq),
        check_gc_content(insert_seq),
        check_kozak(insert_seq),
        check_repeat_runs(insert_seq),
        check_fusion_linker(fusion_parts),
        check_promoter_count(backbone),
    ]

    score = 100 + sum(c.score_delta for c in checks)
    score = max(0, min(100, score))

    failed = [c for c in checks if not c.passed]
    critical = [c for c in failed if c.severity == "critical"]

    if score >= 85:
        level = "High"
    elif score >= 70:
        level = "Moderate"
    elif score >= 50:
        level = "Low"
    else:
        level = "Very Low"

    summary = f"Design Confidence: {score}/100 ({level})"

    if critical:
        rec = f"Critical issue detected: {critical[0].message} Address this before proceeding."
    elif failed:
        rec = f"Consider addressing: {failed[0].message}"
    else:
        rec = "All checks passed. Construct looks good for expression."

    return ConfidenceReport(
        overall_score=score,
        checks=checks,
        summary=summary,
        recommendation=rec,
    )


def format_confidence_report(report: ConfidenceReport) -> str:
    """Human-readable report matching the plan's output format."""
    lines = [report.summary, ""]

    by_cat: dict[str, list[ConfidenceCheck]] = {}
    for c in report.checks:
        by_cat.setdefault(c.category, []).append(c)

    cat_labels = {
        "cryptic_signals": "Cryptic regulatory signals",
        "expression": "Expression optimality",
        "structural": "Structural concerns",
        "architecture": "Construct architecture",
    }

    for cat, label in cat_labels.items():
        if cat not in by_cat:
            continue
        lines.append(f"{label}:")
        for c in by_cat[cat]:
            mark = "\u2713" if c.passed else "\u26a0"
            delta = f" (score {c.score_delta:+d})" if c.score_delta != 0 else ""
            lines.append(f"  {mark} {c.message}{delta}")
        lines.append("")

    lines.append(f"Recommendation: {report.recommendation}")
    return "\n".join(lines)
