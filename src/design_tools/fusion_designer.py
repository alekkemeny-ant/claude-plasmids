"""
Fusion construct design advisor.

Given a fluorescent protein + target protein pair, returns ~5 ranked
design proposals. Does NOT assemble sequences — provides plans for the
agent to present to the user before calling fuse_inserts / assemble_construct.

Heuristics used
---------------
- FP suitability    : curated database (pKa, oligomerization, compartment issues)
- Protein topology  : sequence-based signal-peptide / MTS / TM-helix / GPI detection
- Internal sites    : disorder prediction (find_fusion_sites from protein_analysis)
- Linker selection  : context-aware linker library with rationale
"""

from __future__ import annotations

from typing import Optional

from src.design_tools.protein_analysis import translate, find_fusion_sites

# ── Kyte–Doolittle hydrophobicity (normalized 0→1, 1 = most hydrophobic) ──
_KD_RAW: dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
_KD: dict[str, float] = {aa: (v + 4.5) / 9.0 for aa, v in _KD_RAW.items()}

# ── Curated FP properties ──────────────────────────────────────────────────
# pKa: pH at which fluorescence falls to 50% (lower → better for acidic organelles)
# brightness: relative to EGFP = 1.0 (EC × QY product, normalized)
# oligomerization: monomer | weak_dimer | dimer | tandem_dimer | tetramer
# redox_sensitive: True if structural cysteines or cofactor that oxidizes

FP_DB: dict[str, dict] = {
    "mcherry": {
        "canonical_name": "mCherry",
        "color": "red",
        "ex_nm": 587, "em_nm": 610,
        "oligomerization": "monomer",
        "pka": 4.5,
        "redox_sensitive": False,
        "brightness": 0.72,
        "aa_length": 236,
        "maturation": "fast",
        "photostability": "moderate",
        "strengths": [
            "monomeric", "pKa 4.5 — stable in acidic organelles", "extensively validated",
        ],
        "weaknesses": ["moderate brightness", "moderate photostability vs mScarlet3"],
        "compartment_concerns": {},
        "aliases": {"cherry", "mcherry2"},
    },
    "mscarlet": {
        "canonical_name": "mScarlet",
        "color": "red",
        "ex_nm": 569, "em_nm": 594,
        "oligomerization": "monomer",
        "pka": 4.6,
        "redox_sensitive": False,
        "brightness": 2.41,
        "aa_length": 236,
        "maturation": "medium",
        "photostability": "excellent",
        "strengths": ["bright monomeric red", "excellent photostability"],
        "weaknesses": [],
        "compartment_concerns": {},
        "aliases": {"mscarlet-i", "scarlet"},
    },
    "mscarlet3": {
        "canonical_name": "mScarlet3",
        "color": "red",
        "ex_nm": 569, "em_nm": 594,
        "oligomerization": "monomer",
        "pka": 4.5,
        "redox_sensitive": False,
        "brightness": 3.3,
        "aa_length": 236,
        "maturation": "fast",
        "photostability": "excellent",
        "strengths": ["brightest monomeric red FP", "fast maturation", "excellent photostability"],
        "weaknesses": [],
        "compartment_concerns": {},
        "aliases": {"scarlet3"},
    },
    "egfp": {
        "canonical_name": "EGFP",
        "color": "green",
        "ex_nm": 488, "em_nm": 507,
        "oligomerization": "weak_dimer",
        "pka": 6.0,
        "redox_sensitive": False,
        "brightness": 1.0,
        "aa_length": 239,
        "maturation": "medium",
        "photostability": "good",
        "strengths": ["extensively characterized", "wide antibody/tool availability"],
        "weaknesses": [
            "weak dimer tendency — use A206K (mEGFP) for true monomer",
            "pKa 6.0 — dims at pH < 6.5",
        ],
        "compartment_concerns": {
            "lysosome":  "pKa 6.0 — will not fluoresce at lysosomal pH (~5)",
            "late_endosome": "pKa 6.0 — significantly dims in late endosomes",
            "golgi":     "pKa 6.0 — may dim in acidic Golgi subcompartments",
        },
        "aliases": {"gfp", "enhanced gfp", "enhanced green fluorescent protein"},
    },
    "megfp": {
        "canonical_name": "mEGFP",
        "color": "green",
        "ex_nm": 488, "em_nm": 507,
        "oligomerization": "monomer",
        "pka": 6.0,
        "redox_sensitive": False,
        "brightness": 0.9,
        "aa_length": 239,
        "maturation": "medium",
        "photostability": "good",
        "strengths": ["truly monomeric (A206K)", "well-characterized"],
        "weaknesses": ["pKa 6.0 — dims at pH < 6.5"],
        "compartment_concerns": {
            "lysosome":  "pKa 6.0 — not suitable for lysosomes",
            "late_endosome": "pKa 6.0 — dims in late endosomes",
        },
        "aliases": {"monomer egfp", "egfp a206k"},
    },
    "mneongreen": {
        "canonical_name": "mNeonGreen",
        "color": "green",
        "ex_nm": 506, "em_nm": 517,
        "oligomerization": "monomer",
        "pka": 5.1,
        "redox_sensitive": False,
        "brightness": 3.0,
        "aa_length": 237,
        "maturation": "fast",
        "photostability": "excellent",
        "strengths": [
            "brightest monomeric green (~3× EGFP)", "pKa 5.1", "fast maturation",
        ],
        "weaknesses": [],
        "compartment_concerns": {
            "lysosome": "pKa 5.1 — dims at lysosomal pH (~5); consider mTurquoise2 or mTagBFP2",
        },
        "aliases": {"neongreen"},
    },
    "mturquoise2": {
        "canonical_name": "mTurquoise2",
        "color": "cyan",
        "ex_nm": 434, "em_nm": 474,
        "oligomerization": "monomer",
        "pka": 3.1,
        "redox_sensitive": False,
        "brightness": 0.84,
        "aa_length": 239,
        "maturation": "medium",
        "photostability": "excellent",
        "strengths": [
            "pKa 3.1 — stable in all acidic organelles", "excellent photostability",
        ],
        "weaknesses": [
            "cyan emission — more autofluorescence competition than red/far-red",
            "requires 440 nm excitation (less common on many microscopes)",
        ],
        "compartment_concerns": {},
        "aliases": {"turquoise2", "cfp", "cyan fp"},
    },
    "mtagbfp2": {
        "canonical_name": "mTagBFP2",
        "color": "blue",
        "ex_nm": 399, "em_nm": 454,
        "oligomerization": "monomer",
        "pka": 2.7,
        "redox_sensitive": False,
        "brightness": 1.13,
        "aa_length": 236,
        "maturation": "fast",
        "photostability": "excellent",
        "strengths": [
            "lowest pKa of common FPs (2.7) — ultra-stable in all acidic compartments",
        ],
        "weaknesses": [
            "UV/violet excitation (~399 nm) — potential phototoxicity",
            "blue emission has higher autofluorescence competition",
        ],
        "compartment_concerns": {},
        "aliases": {"tagbfp2", "bfp"},
    },
    "tdtomato": {
        "canonical_name": "tdTomato",
        "color": "red-orange",
        "ex_nm": 554, "em_nm": 581,
        "oligomerization": "tandem_dimer",
        "pka": 4.7,
        "redox_sensitive": False,
        "brightness": 4.7,
        "aa_length": 476,
        "maturation": "fast",
        "photostability": "good",
        "strengths": ["extremely bright (~4.7× EGFP)", "pKa 4.7"],
        "weaknesses": [
            "LARGE (~54 kDa / 476 aa) — may interfere with protein function",
            "size makes internal fusions impractical",
        ],
        "compartment_concerns": {
            "general": "Large size can sterically interfere; use mCherry or mScarlet3 if in doubt",
        },
        "aliases": {"tdtom", "td tomato"},
    },
    "dsred": {
        "canonical_name": "DsRed",
        "color": "red",
        "ex_nm": 556, "em_nm": 586,
        "oligomerization": "tetramer",
        "pka": 4.7,
        "redox_sensitive": False,
        "brightness": 2.3,
        "aa_length": 236,
        "maturation": "slow",
        "photostability": "good",
        "strengths": [],
        "weaknesses": [
            "CRITICAL: obligate tetramer — causes artificial protein clustering and aggregation",
            "slow chromophore maturation (~10 h at 37 °C)",
            "NOT suitable for fusion proteins",
        ],
        "compartment_concerns": {
            "all": (
                "CRITICAL: obligate tetramer causes protein aggregation in all contexts — "
                "strongly advise replacing with mCherry or mScarlet3"
            ),
        },
        "aliases": {"dsred2", "dsred-express", "ds-red"},
    },
    "mkate2": {
        "canonical_name": "mKate2",
        "color": "far-red",
        "ex_nm": 588, "em_nm": 633,
        "oligomerization": "monomer",
        "pka": 6.0,
        "redox_sensitive": False,
        "brightness": 1.4,
        "aa_length": 239,
        "maturation": "medium",
        "photostability": "excellent",
        "strengths": [
            "far-red emission — minimal overlap with autofluorescence",
            "good for multicolor imaging alongside GFP/mCherry",
        ],
        "weaknesses": ["pKa 6.0 — dims at pH < 6.5"],
        "compartment_concerns": {
            "lysosome":  "pKa 6.0 — not suitable for lysosomes",
            "late_endosome": "pKa 6.0 — dims in late endosomes",
        },
        "aliases": {"kate2", "far-red fp"},
    },
    "irfp": {
        "canonical_name": "iRFP",
        "color": "near-infrared",
        "ex_nm": 690, "em_nm": 713,
        "oligomerization": "dimer",
        "pka": None,
        "redox_sensitive": True,
        "brightness": 0.4,
        "aa_length": 315,
        "maturation": "fast",
        "photostability": "good",
        "strengths": ["NIR emission (690–713 nm) — excellent for deep tissue/in vivo"],
        "weaknesses": [
            "requires biliverdin chromophore (supplementation may be needed)",
            "dimeric — can cross-link tagged proteins",
            "lower brightness than GFP-class FPs",
        ],
        "compartment_concerns": {
            "general": "Biliverdin availability varies by compartment and cell type",
        },
        "aliases": {"irfp670", "irfp720", "near-infrared fp"},
    },
}

# ── Standard linker library ────────────────────────────────────────────────
# DNA sequences follow the existing project convention in DEFAULT_FUSION_LINKER
LINKER_LIBRARY: dict[str, dict] = {
    "none": {
        "aa": "",
        "dna": "",
        "description": "No linker (direct fusion)",
        "use_case": "Epitope tags only — not appropriate for FP fusions",
    },
    "short_flexible": {
        "aa": "GGGGS",
        "dna": "GGTGGCGGCGGCTCT",
        "description": "(GGGGS)×1 — 5 aa flexible",
        "use_case": "Minimal separation; acceptable when space is constrained",
    },
    "standard_flexible": {
        "aa": "GGGGSGGGGSGGGGSGGGGS",
        "dna": "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC",
        "description": "(GGGGS)×4 — 20 aa flexible",
        "use_case": "Default for most protein–protein fusions; prevents steric clash",
    },
    "long_flexible": {
        "aa": "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGS",
        "dna": (
            "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC"
            "GGTGGCGGTGGCTCTGGCGGTGGTGGTTCCGGTGGCGGTGGCTCCGGCGGTGGCGGTAGC"
            "GGTGGCGGCGGCTCT"
        ),
        "description": "(GGGGS)×7 — 35 aa long flexible",
        "use_case": "Confined organellar environments, very large fusion partners, or when function requires maximum domain independence",
    },
    "rigid_helical": {
        "aa": "EAAAKEAAAKEAAAK",
        "dna": "GAAGCTGCAGCAAAAGAAGCTGCAGCAAAAGAAGCTGCAGCAAAA",
        "description": "(EAAAK)×3 — 15 aa rigid alpha-helical",
        "use_case": "Defined spatial separation without flexibility; minimises FRET between domains",
    },
}

# ── Compartment pH and redox reference ────────────────────────────────────
_COMPARTMENT_PROPS: dict[str, dict] = {
    "cytoplasm":            {"ph_min": 7.0, "ph_max": 7.4, "oxidizing": False},
    "nucleus":              {"ph_min": 7.0, "ph_max": 7.4, "oxidizing": False},
    "er_lumen":             {"ph_min": 7.0, "ph_max": 7.4, "oxidizing": True},
    "golgi":                {"ph_min": 6.0, "ph_max": 6.8, "oxidizing": True},
    "early_endosome":       {"ph_min": 6.0, "ph_max": 6.8, "oxidizing": False},
    "late_endosome":        {"ph_min": 5.5, "ph_max": 6.0, "oxidizing": False},
    "lysosome":             {"ph_min": 4.5, "ph_max": 5.5, "oxidizing": False},
    "mitochondria_matrix":  {"ph_min": 7.8, "ph_max": 8.0, "oxidizing": False},
    "mitochondria_ims":     {"ph_min": 6.9, "ph_max": 7.2, "oxidizing": True},
    "peroxisome":           {"ph_min": 7.0, "ph_max": 7.4, "oxidizing": True},
    "extracellular":        {"ph_min": 7.3, "ph_max": 7.4, "oxidizing": False},
    "secretory_vesicle":    {"ph_min": 5.5, "ph_max": 6.5, "oxidizing": False},
}

# Keyword → compartment key (for free-text localization hints)
_LOCALIZATION_KEYWORDS: dict[str, str] = {
    "cytoplasm": "cytoplasm", "cytosol": "cytoplasm", "cytoplasmic": "cytoplasm",
    "nucleus": "nucleus", "nuclear": "nucleus",
    "er": "er_lumen", "endoplasmic reticulum": "er_lumen",
    "golgi": "golgi",
    "endosome": "early_endosome", "early endosome": "early_endosome",
    "late endosome": "late_endosome",
    "lysosome": "lysosome", "lysosomal": "lysosome",
    "mitochondria": "mitochondria_matrix", "mitochondrial": "mitochondria_matrix",
    "mitochondrial matrix": "mitochondria_matrix",
    "ims": "mitochondria_ims", "intermembrane space": "mitochondria_ims",
    "peroxisome": "peroxisome",
    "secreted": "extracellular", "extracellular": "extracellular",
    "plasma membrane": "extracellular",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _mean_kd(aa_seq: str, start: int, length: int) -> float:
    """Mean normalised Kyte–Doolittle hydrophobicity for a window."""
    window = aa_seq[start: start + length]
    if not window:
        return 0.5
    return sum(_KD.get(r, 0.5) for r in window) / len(window)


def _positive_charge_fraction(aa_seq: str, start: int, length: int) -> float:
    """Fraction of R + K residues in a window."""
    window = aa_seq[start: start + length]
    if not window:
        return 0.0
    return sum(1 for r in window if r in ("R", "K")) / len(window)


def _lookup_fp(name: str) -> Optional[dict]:
    """Resolve an FP name to its DB entry (case-insensitive, alias-aware)."""
    key = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in FP_DB:
        return FP_DB[key]
    for entry in FP_DB.values():
        aliases = {a.lower().replace("-", "").replace("_", "").replace(" ", "")
                   for a in entry.get("aliases", set())}
        if key in aliases:
            return entry
    return None


def _resolve_compartment(localization_hint: str) -> Optional[str]:
    """Map a free-text localization string to a key in _COMPARTMENT_PROPS."""
    hint = localization_hint.lower().strip()
    for kw, comp in _LOCALIZATION_KEYWORDS.items():
        if kw in hint:
            return comp
    return None


# ── Topology prediction ─────────────────────────────────────────────────────

def predict_protein_topology(aa_seq: str) -> dict:
    """Heuristic topology analysis for fusion design guidance.

    Scans for:
    - Signal peptide (SP): hydrophobic N-terminal h-region → ER targeting / cleavage
    - Mitochondrial targeting sequence (MTS): positively charged amphipathic N-terminus
    - Transmembrane helices (TM): internal hydrophobic ~20-aa stretches
    - GPI anchor signal (GPI): C-terminal hydrophobic stretch → GPI attachment / cleavage

    Returns a dict with:
    - features: list of detected topology features with type, region, note
    - n_terminal_accessible: bool — False if N-terminus is predicted to be cleaved
    - c_terminal_accessible: bool — False if C-terminus is predicted to be cleaved
    - tm_count: number of predicted TM helices
    - inferred_localization: str or None — coarse compartment guess
    - warnings: list of design-relevant cautions
    """
    n = len(aa_seq)
    features: list[dict] = []
    n_accessible = True
    c_accessible = True
    tm_positions: list[int] = []
    inferred_loc: Optional[str] = None
    warnings: list[str] = []

    # ── Signal peptide (SP) detection ───────────────────────────────────
    # Criterion: max sliding-window KD in first 35 aa > 0.68 (h-region)
    # AND n-region positive charge (first 5 aa) OR c-region AXA cleavage hint
    if n >= 25:
        max_h = max(
            (_mean_kd(aa_seq, i, 10) for i in range(min(n - 10, 25))),
            default=0.0,
        )
        n_region_charge = _positive_charge_fraction(aa_seq, 0, 5)
        # Distinguish from MTS: SP has high max hydrophobicity, MTS has lower
        # but high overall positive charge across longer stretch
        mts_pos = _positive_charge_fraction(aa_seq, 0, 40) if n >= 40 else 0.0
        mts_h   = _mean_kd(aa_seq, 0, 40) if n >= 40 else 0.0

        if max_h > 0.68 and mts_pos < 0.18:
            features.append({
                "type": "signal_peptide",
                "region": "N-terminal ~1–30 aa",
                "note": (
                    "Predicted signal peptide (hydrophobic h-region at N-terminus). "
                    "The signal peptide is co-translationally cleaved in the ER lumen — "
                    "an N-terminal FP tag would be lost with the signal peptide."
                ),
            })
            n_accessible = False
            inferred_loc = "er_lumen"

        # ── MTS detection ─────────────────────────────────────────────────
        # Criterion: >15% R+K in first 40 aa, moderate hydrophobicity (amphipathic helix)
        elif mts_pos >= 0.15 and 0.40 <= mts_h <= 0.65 and n >= 40:
            features.append({
                "type": "mts",
                "region": "N-terminal ~1–40 aa",
                "note": (
                    "Predicted mitochondrial targeting sequence (MTS): "
                    "positively charged amphipathic N-terminal region. "
                    "MTS is cleaved in the mitochondrial matrix after import — "
                    "an N-terminal FP blocks threading through the import channel."
                ),
            })
            n_accessible = False
            inferred_loc = "mitochondria_matrix"

    # ── Transmembrane helix detection ────────────────────────────────────
    # Criterion: 20-aa windows with mean KD > 0.72 outside first/last 15 aa
    if n >= 60:
        scan_end = max(20, n - 20)
        for i in range(15, scan_end - 20):
            h = _mean_kd(aa_seq, i, 20)
            if h > 0.72:
                if not any(abs(i - prev) < 15 for prev in tm_positions):
                    tm_positions.append(i)
                    features.append({
                        "type": "transmembrane",
                        "region": f"~{i + 1}–{i + 20} aa",
                        "note": (
                            f"Predicted TM helix at residues ~{i + 1}–{i + 20}. "
                            "Membrane proteins have complex topology — terminus accessibility "
                            "depends on the number of TM passes and the orientation of the first."
                        ),
                    })

    if tm_positions and not inferred_loc:
        inferred_loc = "plasma_membrane"

    # ── GPI anchor signal detection ──────────────────────────────────────
    # Criterion: last 15 aa have mean KD > 0.65 (omega site + hydrophobic region)
    if n >= 25:
        c_hydro = _mean_kd(aa_seq, n - 15, 15)
        if c_hydro > 0.65 and not tm_positions:
            features.append({
                "type": "gpi_anchor",
                "region": f"C-terminal ~{n - 15}–{n} aa",
                "note": (
                    "Predicted GPI anchor signal at C-terminus. "
                    "The GPI signal is post-translationally cleaved and replaced with "
                    "a GPI moiety — a C-terminal FP would be removed."
                ),
            })
            c_accessible = False
            if not inferred_loc:
                inferred_loc = "extracellular"

    # ── Multi-pass TM warning ────────────────────────────────────────────
    if len(tm_positions) >= 2:
        warnings.append(
            f"Protein has {len(tm_positions)} predicted TM helices. "
            "Terminus orientation relative to the membrane determines which terminus "
            "is cytoplasmic. For type-I TM proteins (single pass, N-terminal "
            "extracellular, C-terminal cytoplasmic) the C-terminus is accessible "
            "for cytoplasmic FP fusions. Confirm topology before choosing fusion site."
        )

    return {
        "features": features,
        "n_terminal_accessible": n_accessible,
        "c_terminal_accessible": c_accessible,
        "tm_count": len(tm_positions),
        "inferred_localization": inferred_loc,
        "warnings": warnings,
        "protein_length_aa": n,
    }


# ── FP suitability assessment ──────────────────────────────────────────────

def assess_fp_suitability(
    fp_props: dict,
    compartment_key: Optional[str],
    topology: dict,
) -> dict:
    """Score the chosen FP for the target protein's compartment.

    Returns:
    - score: 0–100
    - verdict: "excellent" | "good" | "marginal" | "poor" | "critical"
    - issues: list of concern strings
    - notes: list of informational strings
    """
    score = 100
    issues: list[str] = []
    notes: list[str] = []

    # Oligomerization
    oligo = fp_props.get("oligomerization", "monomer")
    if oligo == "tetramer":
        score -= 50
        issues.append(
            f"{fp_props['canonical_name']} is an obligate tetramer — will cause "
            "artificial protein clustering and aggregation in fusion constructs."
        )
    elif oligo == "tandem_dimer":
        score -= 15
        issues.append(
            f"{fp_props['canonical_name']} is a tandem dimer (~54 kDa, {fp_props.get('aa_length', '?')} aa) — "
            "large size may interfere with protein folding or localization."
        )
    elif oligo in ("dimer", "weak_dimer"):
        score -= 10
        issues.append(
            f"{fp_props['canonical_name']} has {oligo} tendency — can artificially "
            "dimerize fusion proteins at high expression. Consider a monomeric variant."
        )

    # pKa vs compartment
    if compartment_key and compartment_key in _COMPARTMENT_PROPS:
        comp = _COMPARTMENT_PROPS[compartment_key]
        ph_min = comp["ph_min"]
        pka = fp_props.get("pka")
        if pka is not None:
            margin = pka - ph_min
            if margin >= 1.5:
                score -= 30
                issues.append(
                    f"pKa {pka} too high for {compartment_key.replace('_', ' ')} (pH ~{ph_min}–{comp['ph_max']}): "
                    f"FP will be mostly non-fluorescent. Choose an FP with pKa < {ph_min - 0.5:.1f}."
                )
            elif margin > 0.5:
                score -= 15
                issues.append(
                    f"pKa {pka} is borderline for {compartment_key.replace('_', ' ')} (pH ~{ph_min}–{comp['ph_max']}): "
                    "fluorescence will be reduced. Monitor carefully; consider a lower-pKa FP."
                )
            else:
                notes.append(
                    f"pKa {pka} — well-suited for {compartment_key.replace('_', ' ')} (pH ~{ph_min}–{comp['ph_max']})."
                )

        # Redox
        if comp.get("oxidizing") and fp_props.get("redox_sensitive"):
            score -= 20
            issues.append(
                f"{fp_props['canonical_name']} has redox-sensitive chromophore components "
                f"and {compartment_key.replace('_', ' ')} is an oxidizing compartment — "
                "fluorescence may be compromised."
            )

    # Compartment-specific curated concerns
    comp_concerns = fp_props.get("compartment_concerns", {})
    if compartment_key:
        key_short = compartment_key.replace("mitochondria_", "mito_").split("_")[0]
        for concern_key, concern_msg in comp_concerns.items():
            if concern_key in (compartment_key, key_short, "all", "general"):
                score -= 15
                issues.append(concern_msg)

    # Size penalty for very large FPs (internal fusions especially)
    if fp_props.get("aa_length", 0) > 350:
        score -= 10
        issues.append(
            f"FP is large ({fp_props.get('aa_length')} aa) — may interfere with internal fusions "
            "or localization of smaller proteins."
        )

    score = max(0, score)
    if score >= 80:
        verdict = "excellent"
    elif score >= 60:
        verdict = "good"
    elif score >= 40:
        verdict = "marginal"
    elif score > 0:
        verdict = "poor"
    else:
        verdict = "critical"

    return {"score": score, "verdict": verdict, "issues": issues, "notes": notes}


def suggest_fp_alternatives(
    fp_props: dict,
    compartment_key: Optional[str],
    suitability: dict,
) -> list[dict]:
    """Suggest better FP options when the chosen FP has issues.

    Returns a list of dicts: {name, canonical_name, reason}
    """
    if suitability["verdict"] in ("excellent", "good"):
        return []

    color = fp_props.get("color", "")
    chosen_pka = fp_props.get("pka")
    chosen_oligo = fp_props.get("oligomerization", "monomer")

    comp_ph_min = 7.0
    if compartment_key and compartment_key in _COMPARTMENT_PROPS:
        comp_ph_min = _COMPARTMENT_PROPS[compartment_key]["ph_min"]

    suggestions: list[dict] = []

    for key, entry in FP_DB.items():
        if entry["canonical_name"] == fp_props.get("canonical_name"):
            continue
        if entry.get("oligomerization") in ("tetramer", "dimer", "tandem_dimer"):
            continue

        entry_pka = entry.get("pka")
        entry_color = entry.get("color", "")
        same_color = entry_color == color
        reasons: list[str] = []

        # Better oligomerization → suggest monomeric alternatives (same color preferred)
        # In acidic compartments, also require that the alternative improves pKa,
        # otherwise we'd suggest an FP that has the same pH problem.
        if chosen_oligo in ("weak_dimer", "dimer", "tetramer", "tandem_dimer"):
            if entry["oligomerization"] == "monomer" and same_color:
                acidic = comp_ph_min < 6.5
                pka_improves = (
                    not acidic
                    or chosen_pka is None
                    or entry_pka is None
                    or entry_pka < chosen_pka - 0.5
                )
                if pka_improves:
                    reasons.append("truly monomeric, same color channel")

        # Better pKa for acidic compartments → cross-color suggestions are fine
        if chosen_pka is not None and entry_pka is not None and comp_ph_min < 6.5:
            if chosen_pka > comp_ph_min + 0.5 and entry_pka < comp_ph_min - 0.3:
                channel_note = "(same color channel)" if same_color else f"({entry_color})"
                reasons.append(
                    f"pKa {entry_pka} — stable at compartment pH {comp_ph_min:.1f} {channel_note}"
                )

        # Better brightness in same color channel (if chosen FP is dim)
        if fp_props.get("brightness", 1.0) < 1.0 and same_color:
            if entry.get("brightness", 0) > fp_props.get("brightness", 1.0) * 1.5:
                reasons.append(
                    f"brighter ({entry['brightness']:.1f}× EGFP vs "
                    f"{fp_props.get('brightness', '?')}× for {fp_props['canonical_name']})"
                )

        if reasons:
            suggestions.append({
                "key": key,
                "canonical_name": entry["canonical_name"],
                "reason": "; ".join(reasons),
                "pka": entry_pka,
                "brightness": entry.get("brightness"),
                "oligomerization": entry.get("oligomerization"),
                "color": entry_color,
            })

    # Rank: same-color first, then by lowest pKa (most acid-stable)
    suggestions.sort(
        key=lambda s: (
            0 if s.get("color") == color else 1,
            s.get("pka") if s.get("pka") is not None else 99,
        )
    )
    return suggestions[:3]


# ── Design generation ──────────────────────────────────────────────────────

def _score_design(
    config: str,
    linker_key: str,
    topology: dict,
    fp_suitability: dict,
    internal_site: Optional[dict],
) -> int:
    """Compute a 0–100 score for a design proposal."""
    score = 0

    # Terminus accessibility (0–45)
    if config == "N-terminal":
        if topology["n_terminal_accessible"]:
            score += 45
        else:
            score += 5  # heavily penalised but not zero (might still be tried)
    elif config == "C-terminal":
        if topology["c_terminal_accessible"]:
            score += 45
        else:
            score += 5
    elif config == "internal":
        # Internal fusions score on internal site quality
        if internal_site:
            site_score = min(40, int(internal_site["mean_disorder"] * internal_site["length"] * 2))
            score += site_score
        else:
            score += 5

    # FP suitability (0–30)
    score += int(fp_suitability["score"] * 0.3)

    # Linker appropriateness (0–15)
    protein_len = topology.get("protein_length_aa", 200)
    if config == "internal":
        linker_score = 12 if linker_key == "standard_flexible" else 8
    elif protein_len > 400:
        linker_score = 12 if linker_key in ("standard_flexible", "long_flexible") else 8
    else:
        linker_score = 13 if linker_key == "standard_flexible" else 10
    score += linker_score

    # Penalise internal fusions (inherently riskier) by 10
    if config == "internal":
        score -= 10

    return max(0, min(100, score))


def _confidence_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Low"
    return "Very Low"


def generate_fusion_designs(
    fp_name: str,
    target_gene_name: str,
    fp_suitability: dict,
    topology: dict,
    internal_sites: list[dict],
) -> list[dict]:
    """Generate ~5 fusion design proposals and rank them.

    Design space:
    1. C-terminal FP, standard flexible linker
    2. N-terminal FP, standard flexible linker
    3. C-terminal FP, long flexible linker (extra separation)
    4. N-terminal FP, rigid helical linker (defined spacing)
    5. Internal FP at best disordered loop (or C-terminal short-linker variant)
    """
    designs: list[dict] = []

    # Design 1: C-terminal, standard flexible
    s = _score_design("C-terminal", "standard_flexible", topology, fp_suitability, None)
    concerns = []
    if not topology["c_terminal_accessible"]:
        concerns.append("C-terminus predicted to be non-accessible (signal peptide / GPI / TM anchor).")
    designs.append({
        "name": f"{target_gene_name}–{fp_name} (C-terminal, standard flexible linker)",
        "configuration": "C-terminal",
        "orientation": f"{target_gene_name} — (GGGGS)×4 — {fp_name}",
        "linker": LINKER_LIBRARY["standard_flexible"]["description"],
        "linker_rationale": (
            "(GGGGS)×4 provides 20 aa of flexible spacing, reducing steric interference "
            "between the target protein's folded C-terminus and the FP beta-barrel."
        ),
        "design_rationale": (
            f"C-terminal fusions place the FP after the last residue of {target_gene_name}. "
            "This preserves the native N-terminus (including any targeting sequences) and is "
            "the most common configuration for cytoplasmic and organellar localisation studies."
        ),
        "concerns": concerns,
        "score": s,
        "confidence": _confidence_label(s),
    })

    # Design 2: N-terminal, standard flexible
    s = _score_design("N-terminal", "standard_flexible", topology, fp_suitability, None)
    concerns = []
    if not topology["n_terminal_accessible"]:
        for f in topology["features"]:
            if f["type"] in ("signal_peptide", "mts"):
                concerns.append(f["note"])
    designs.append({
        "name": f"{fp_name}–{target_gene_name} (N-terminal, standard flexible linker)",
        "configuration": "N-terminal",
        "orientation": f"{fp_name} — (GGGGS)×4 — {target_gene_name}",
        "linker": LINKER_LIBRARY["standard_flexible"]["description"],
        "linker_rationale": (
            "(GGGGS)×4 allows the FP to fold independently before the target protein sequence begins."
        ),
        "design_rationale": (
            f"N-terminal fusions tag {target_gene_name} at its first residue. "
            "Works well when the N-terminus is cytoplasmic and not involved in signal sequences, "
            "import, or functional interactions."
        ),
        "concerns": concerns,
        "score": s,
        "confidence": _confidence_label(s),
    })

    # Design 3: C-terminal, long flexible linker (extra space)
    s = _score_design("C-terminal", "long_flexible", topology, fp_suitability, None)
    concerns = []
    if not topology["c_terminal_accessible"]:
        concerns.append("C-terminus predicted to be non-accessible.")
    designs.append({
        "name": f"{target_gene_name}–{fp_name} (C-terminal, long flexible linker)",
        "configuration": "C-terminal",
        "orientation": f"{target_gene_name} — (GGGGS)×7 — {fp_name}",
        "linker": LINKER_LIBRARY["long_flexible"]["description"],
        "linker_rationale": (
            "(GGGGS)×7 (35 aa) provides additional separation for proteins in crowded environments "
            "(e.g., organellar lumen, membrane proximity) or when the C-terminal domain of the "
            "target protein is large and might occlude the FP."
        ),
        "design_rationale": (
            "Functionally equivalent to Design 1 but with a longer linker. "
            "Useful if the standard construct shows reduced function, suggesting steric clash."
        ),
        "concerns": concerns,
        "score": max(0, s - 5),  # slight penalty vs standard (more complexity)
        "confidence": _confidence_label(max(0, s - 5)),
    })

    # Design 4: N-terminal, rigid helical linker
    s = _score_design("N-terminal", "rigid_helical", topology, fp_suitability, None)
    concerns = []
    if not topology["n_terminal_accessible"]:
        concerns.append("N-terminus predicted to be non-accessible.")
    designs.append({
        "name": f"{fp_name}–{target_gene_name} (N-terminal, rigid helical linker)",
        "configuration": "N-terminal",
        "orientation": f"{fp_name} — (EAAAK)×3 — {target_gene_name}",
        "linker": LINKER_LIBRARY["rigid_helical"]["description"],
        "linker_rationale": (
            "(EAAAK)×3 forms a stable alpha helix, maintaining a fixed distance between the FP "
            "and the target protein without flexibility. Useful when FRET between domains must "
            "be minimised, or when independent folding of each domain is critical."
        ),
        "design_rationale": (
            "Alternative N-terminal orientation with a rigid linker. "
            "Less common than flexible-linker designs but preferred in FRET-based assays or "
            "when the flexible linker is suspected of allowing undesired inter-domain contacts."
        ),
        "concerns": concerns,
        "score": max(0, s - 5),
        "confidence": _confidence_label(max(0, s - 5)),
    })

    # Design 5: Internal fusion at best loop OR fallback C-terminal short linker
    if internal_sites:
        best_site = internal_sites[0]
        s = _score_design("internal", "standard_flexible", topology, fp_suitability, best_site)
        mid = (best_site["start"] + best_site["end"]) // 2
        designs.append({
            "name": (
                f"{target_gene_name}[{best_site['start'] + 1}–{best_site['end']}]–{fp_name}–{target_gene_name} "
                f"(internal at disordered loop)"
            ),
            "configuration": "internal",
            "orientation": (
                f"{target_gene_name}[1–{mid}] — (GGGGS)×4 — {fp_name} — (GGGGS)×4 — "
                f"{target_gene_name}[{mid + 1}–end]"
            ),
            "insertion_site": best_site,
            "linker": (
                f"(GGGGS)×4 flanking on both sides of {fp_name} insertion "
                f"(residues {best_site['start'] + 1}–{best_site['end']} of {target_gene_name})"
            ),
            "linker_rationale": (
                "Disordered loops tolerate insertions best. Flanking flexible linkers let the FP "
                "fold independently without distorting the target protein's core fold."
            ),
            "design_rationale": (
                f"Inserts {fp_name} into a predicted disordered loop at residues "
                f"{best_site['start'] + 1}–{best_site['end']} "
                f"(disorder score {best_site['mean_disorder']:.2f}, {best_site['length']} aa). "
                "Internal fusions are most useful when both terminal fusions are predicted to fail "
                "(e.g., N-terminus has an import signal and C-terminus is membrane-anchored), or "
                "for proteins where both termini are functionally critical. "
                "Verify the insertion site against AlphaFold2 structure for high-stakes designs."
            ),
            "concerns": [
                "Internal fusions are structurally riskier than terminal fusions — "
                "even in disordered loops, insertions can disrupt function.",
                "Disorder predictor is a sequence heuristic — validate against AlphaFold2.",
            ],
            "score": s,
            "confidence": _confidence_label(s),
        })
    else:
        # Fallback: C-terminal with short linker as fifth option
        s = _score_design("C-terminal", "short_flexible", topology, fp_suitability, None)
        concerns = []
        if not topology["c_terminal_accessible"]:
            concerns.append("C-terminus predicted to be non-accessible.")
        designs.append({
            "name": f"{target_gene_name}–{fp_name} (C-terminal, short flexible linker)",
            "configuration": "C-terminal",
            "orientation": f"{target_gene_name} — (GGGGS)×1 — {fp_name}",
            "linker": LINKER_LIBRARY["short_flexible"]["description"],
            "linker_rationale": (
                "(GGGGS)×1 is more compact than the standard linker. "
                "Appropriate when minimal separation is desired and the C-terminus is not "
                "directly involved in target protein contacts."
            ),
            "design_rationale": (
                "Compact C-terminal fusion. No predicted internal disordered loops were found "
                "in this protein, so a terminal fusion is the primary recommendation. "
                "Short linker reduces the conformational freedom of the FP relative to the "
                "target, which may improve co-localisation accuracy."
            ),
            "concerns": concerns,
            "score": max(0, s - 3),
            "confidence": _confidence_label(max(0, s - 3)),
        })

    # Rank by score descending
    designs.sort(key=lambda d: d["score"], reverse=True)
    for i, d in enumerate(designs):
        d["rank"] = i + 1

    return designs


# ── Main entry point ───────────────────────────────────────────────────────

def design_fusion_variants(
    fp_name: str,
    target_gene_name: str,
    target_aa_sequence: Optional[str] = None,
    target_dna_sequence: Optional[str] = None,
    fp_aa_sequence: Optional[str] = None,
    known_localization: Optional[str] = None,
) -> dict:
    """Generate and rank ~5 fluorescent fusion construct designs.

    Args:
        fp_name            : FP identifier (e.g. "mCherry", "mNeonGreen")
        target_gene_name   : Target gene/protein name (e.g. "CHCHD4")
        target_aa_sequence : Target protein AA sequence (preferred)
        target_dna_sequence: Target CDS in DNA (translated if no AA provided)
        fp_aa_sequence     : FP amino acid sequence (used only for length)
        known_localization : Free-text localization hint (e.g. "mitochondria")

    Returns:
        dict with keys:
        - fp_assessment   : FP suitability score + issues
        - alternatives    : list of alternative FP suggestions (if issues found)
        - target_topology : topology prediction for the target protein
        - designs         : list of ~5 ranked design proposals
        - summary         : text summary
    """
    # Resolve AA sequence
    if not target_aa_sequence:
        if target_dna_sequence:
            target_aa_sequence = translate(target_dna_sequence)
        else:
            raise ValueError("Provide either target_aa_sequence or target_dna_sequence.")

    # Look up FP properties
    fp_props = _lookup_fp(fp_name)
    if fp_props is None:
        # Graceful degradation: use a minimal placeholder
        fp_props = {
            "canonical_name": fp_name,
            "oligomerization": "unknown",
            "pka": None,
            "redox_sensitive": False,
            "brightness": 1.0,
            "aa_length": len(fp_aa_sequence) if fp_aa_sequence else 239,
            "strengths": [],
            "weaknesses": [],
            "compartment_concerns": {},
        }

    # Resolve compartment key
    compartment_key: Optional[str] = None
    if known_localization:
        compartment_key = _resolve_compartment(known_localization)

    # Topology analysis
    topology = predict_protein_topology(target_aa_sequence)

    # If no explicit localization provided, use inferred one
    if not compartment_key and topology.get("inferred_localization"):
        compartment_key = topology["inferred_localization"]

    # FP suitability
    fp_suitability = assess_fp_suitability(fp_props, compartment_key, topology)

    # Alternative FP suggestions
    alternatives = suggest_fp_alternatives(fp_props, compartment_key, fp_suitability)

    # Internal fusion sites
    internal_sites = find_fusion_sites(target_aa_sequence, min_window=10)

    # Generate designs
    designs = generate_fusion_designs(
        fp_name=fp_props.get("canonical_name", fp_name),
        target_gene_name=target_gene_name,
        fp_suitability=fp_suitability,
        topology=topology,
        internal_sites=internal_sites,
    )

    # Build summary
    top = designs[0] if designs else None
    summary_lines = [
        f"Design analysis for {fp_props.get('canonical_name', fp_name)}–{target_gene_name} fusion:",
        f"  FP suitability  : {fp_suitability['verdict'].upper()} (score {fp_suitability['score']}/100)",
        f"  Target length   : {topology['protein_length_aa']} aa",
        f"  N-terminus      : {'accessible' if topology['n_terminal_accessible'] else 'BLOCKED (signal/MTS/TM)'}",
        f"  C-terminus      : {'accessible' if topology['c_terminal_accessible'] else 'BLOCKED (GPI/TM)'}",
        f"  TM helices      : {topology['tm_count']}",
        f"  Internal sites  : {len(internal_sites)} predicted disordered loop(s)",
        f"  Top design      : #{top['rank']} — {top['name']} (confidence: {top['confidence']})" if top else "",
    ]
    if alternatives:
        alt_names = ", ".join(a["canonical_name"] for a in alternatives)
        summary_lines.append(f"  Alternatives    : {alt_names}")

    return {
        "fp_name": fp_props.get("canonical_name", fp_name),
        "fp_assessment": fp_suitability,
        "fp_properties": {
            "pka": fp_props.get("pka"),
            "oligomerization": fp_props.get("oligomerization"),
            "brightness": fp_props.get("brightness"),
            "aa_length": fp_props.get("aa_length"),
            "strengths": fp_props.get("strengths", []),
            "weaknesses": fp_props.get("weaknesses", []),
        },
        "alternatives": alternatives,
        "target_topology": topology,
        "internal_sites": internal_sites[:5],
        "designs": designs,
        "compartment": compartment_key,
        "summary": "\n".join(l for l in summary_lines if l),
    }
