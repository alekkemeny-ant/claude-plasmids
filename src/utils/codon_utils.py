from src.data.codon_tables import CODON_TO_AA, HUMAN_FREQ, HUMAN_OPTIMAL_CODONS


def compute_w() -> dict[str, float]:
    """Derive relative adaptiveness w_i = freq / max(freq for same AA)."""
    # Group by AA
    aa_groups: dict[str, list[tuple[str, float]]] = {}
    for codon, freq in HUMAN_FREQ.items():
        aa = CODON_TO_AA[codon]
        aa_groups.setdefault(aa, []).append((codon, freq))
    w: dict[str, float] = {}
    for aa, pairs in aa_groups.items():
        max_freq = max(f for _, f in pairs)
        for codon, freq in pairs:
            w[codon] = round(freq / max_freq, 3)
    return w


# Relative adaptiveness w_i = freq(codon) / max(freq(synonym) for that AA)
# This is what CAI uses directly.
HUMAN_CODON_W: dict[str, float] = compute_w()


