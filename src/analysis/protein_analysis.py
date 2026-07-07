"""Protein-level analysis for fusion design.

Provides:
- translate(): DNA → AA (standard genetic code)
- predict_disorder(): per-residue disorder score (simplified heuristic)
- find_fusion_sites(): rank disordered regions as candidate fusion points

Disorder prediction uses a simplified Uversky-style heuristic (mean
hydrophobicity vs. mean net charge, windowed) — NOT full IUPred3. This
is documented as a heuristic approximation sufficient for Tier-A
screening. Replace with IUPred3 API or AlphaFoldDB pLDDT for Tier B.
"""

from typing import Optional  # noqa: F401 — kept for future type annotations

# ── Standard genetic code ──────────────────────────────────────────────

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def translate(dna_seq: str, frame: int = 0, to_stop: bool = True) -> str:
    """Translate DNA → AA using the standard genetic code.

    Args:
        dna_seq: DNA sequence (will be uppercased, whitespace stripped)
        frame: Reading frame offset (0, 1, or 2)
        to_stop: If True, stop at first stop codon. If False, include '*' and continue.

    Returns:
        Amino acid sequence (single-letter code). Stop codons are '*'.
    """
    seq = dna_seq.upper().replace(" ", "").replace("\n", "")
    seq = seq[frame:]
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        residue = CODON_TABLE.get(codon, "X")  # X for unknown (e.g., N)
        if to_stop and residue == "*":
            break
        aa.append(residue)
    return "".join(aa)


# ── Disorder prediction (simplified Uversky heuristic) ─────────────────

# Kyte-Doolittle hydrophobicity scale, normalized to [0, 1]
# (original range -4.5 to 4.5)
_KD_RAW: dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
# Normalize to [0, 1]: (x - min) / (max - min) where min=-4.5, max=4.5
_KD_NORM: dict[str, float] = {
    aa: (v + 4.5) / 9.0 for aa, v in _KD_RAW.items()
}

# Net charge at pH 7 (approx)
_CHARGE: dict[str, int] = {
    "D": -1, "E": -1, "K": +1, "R": +1, "H": 0,  # His ~neutral at pH7
}

# Per-residue disorder propensity (TOP-IDP-scale-derived, Campen et al. 2008).
# Positive = disorder-promoting, negative = order-promoting.
_DISORDER_PROPENSITY: dict[str, float] = {
    "A": 0.06, "R": 0.18, "N": -0.01, "D": 0.10, "C": -0.20,
    "E": 0.14, "Q": 0.12, "G": 0.17, "H": -0.05, "I": -0.39,
    "L": -0.27, "K": 0.17, "M": -0.17, "F": -0.35, "P": 0.19,
    "S": 0.09, "T": -0.04, "W": -0.35, "Y": -0.20, "V": -0.29,
}


def predict_disorder(aa_seq: str, window: int = 9) -> list[float]:
    """Per-residue disorder score (0=ordered, 1=disordered).

    Combines three biophysical features in a sliding window:
    1. Inverse mean hydrophobicity (Kyte-Doolittle, normalized)
    2. Mean absolute net charge
    3. Mean intrinsic disorder propensity (TOP-IDP scale)

    Score = clamp((1 - <H>) + |<Q>| * 0.5 + <P>) * 2.0 + 0.1, 0, 1)

    This is NOT IUPred — it's a fast heuristic approximation. Treat
    scores as relative rankings within a sequence, not absolute cutoffs
    comparable across sequences.

    Args:
        aa_seq: Amino acid sequence (single-letter code)
        window: Sliding window size (odd number preferred)

    Returns:
        List of per-residue disorder scores, same length as aa_seq.
    """
    n = len(aa_seq)
    if n == 0:
        return []

    half = window // 2

    # Precompute per-residue features
    hydro = [_KD_NORM.get(r, 0.5) for r in aa_seq]
    charge = [_CHARGE.get(r, 0) for r in aa_seq]
    propensity = [_DISORDER_PROPENSITY.get(r, 0.0) for r in aa_seq]

    scores = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        wlen = hi - lo
        win_h = sum(hydro[lo:hi]) / wlen
        win_q = sum(charge[lo:hi]) / wlen
        win_p = sum(propensity[lo:hi]) / wlen
        # Low hydrophobicity + high charge + high propensity = disordered
        raw = (1.0 - win_h) + abs(win_q) * 0.5 + win_p
        score = max(0.0, min(1.0, raw * 2.0 + 0.1))
        scores.append(score)
    return scores


def find_fusion_sites(
    aa_seq: str,
    min_window: int = 10,
    threshold: float = 0.5,
) -> list[dict]:
    """Find disordered regions suitable for fusion partner insertion.

    Scans the disorder profile for contiguous runs ≥min_window residues
    with mean disorder ≥threshold. These are candidate "safe" insertion
    points where a fusion partner is less likely to disrupt the fold.

    Args:
        aa_seq: Amino acid sequence
        min_window: Minimum contiguous disordered residues to call a site
        threshold: Disorder score threshold (0-1)

    Returns:
        List of sites sorted by suitability (mean_disorder × length descending):
        [{"start": int, "end": int, "length": int, "mean_disorder": float, "context": str}]
    """
    disorder = predict_disorder(aa_seq)
    n = len(disorder)

    sites = []
    i = 0
    while i < n:
        if disorder[i] >= threshold:
            j = i
            while j < n and disorder[j] >= threshold:
                j += 1
            length = j - i
            if length >= min_window:
                mean_d = sum(disorder[i:j]) / length
                ctx_lo = max(0, i - 3)
                ctx_hi = min(n, j + 3)
                context = aa_seq[ctx_lo:ctx_hi]
                sites.append({
                    "start": i,
                    "end": j,
                    "length": length,
                    "mean_disorder": round(mean_d, 3),
                    "context": context,
                })
            i = j
        else:
            i += 1

    # Rank by suitability: longer + more disordered = better
    sites.sort(key=lambda s: s["mean_disorder"] * s["length"], reverse=True)
    return sites
