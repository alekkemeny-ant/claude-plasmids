"""Human codon usage tables for CAI (Codon Adaptation Index) calculation.

Data source: Kazusa Codon Usage Database (Homo sapiens, GenBank release).
Frequencies are per-1000-codons, normalized so the most-used codon for each
AA has relative adaptiveness w_i = 1.0.
"""

# Per-1000 frequencies from Kazusa (used to derive w_i below)
_HUMAN_FREQ: dict[str, float] = {
    "TTT": 17.6, "TTC": 20.3,                                                      # Phe
    "TTA": 7.7,  "TTG": 12.9, "CTT": 13.2, "CTC": 19.6, "CTA": 7.2, "CTG": 39.6, # Leu
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5,                                          # Ile
    "ATG": 22.0,                                                                     # Met
    "GTT": 11.0, "GTC": 14.5, "GTA": 7.1, "GTG": 28.1,                             # Val
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4,                             # Ser (part 1)
    "AGT": 12.1, "AGC": 19.5,                                                       # Ser (part 2)
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,                             # Pro
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1,                             # Thr
    "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,                             # Ala
    "TAT": 12.2, "TAC": 15.3,                                                       # Tyr
    "CAT": 10.9, "CAC": 15.1,                                                       # His
    "CAA": 12.3, "CAG": 34.2,                                                       # Gln
    "AAT": 17.0, "AAC": 19.1,                                                       # Asn
    "AAA": 24.4, "AAG": 31.9,                                                       # Lys
    "GAT": 21.8, "GAC": 25.1,                                                       # Asp
    "GAA": 29.0, "GAG": 39.6,                                                       # Glu
    "TGT": 10.6, "TGC": 12.6,                                                       # Cys
    "TGG": 13.2,                                                                     # Trp
    "CGT": 4.5,  "CGC": 10.4, "CGA": 6.2, "CGG": 11.4,                             # Arg (part 1)
    "AGA": 12.2, "AGG": 12.0,                                                       # Arg (part 2)
    "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,                            # Gly
}

# Amino acid → list of synonymous codons (genetic code)
_CODON_TO_AA: dict[str, str] = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _compute_w() -> dict[str, float]:
    """Derive relative adaptiveness w_i = freq / max(freq for same AA)."""
    # Group by AA
    aa_groups: dict[str, list[tuple[str, float]]] = {}
    for codon, freq in _HUMAN_FREQ.items():
        aa = _CODON_TO_AA[codon]
        aa_groups.setdefault(aa, []).append((codon, freq))
    w: dict[str, float] = {}
    for aa, pairs in aa_groups.items():
        max_freq = max(f for _, f in pairs)
        for codon, freq in pairs:
            w[codon] = round(freq / max_freq, 3)
    return w


# Relative adaptiveness w_i = freq(codon) / max(freq(synonym) for that AA)
# This is what CAI uses directly.
HUMAN_CODON_W: dict[str, float] = _compute_w()

# Convenience: the most-used codon per AA (w_i = 1.0 codons)
HUMAN_OPTIMAL_CODONS: dict[str, str] = {
    "F": "TTC", "L": "CTG", "I": "ATC", "M": "ATG", "V": "GTG",
    "S": "AGC", "P": "CCC", "T": "ACC", "A": "GCC", "Y": "TAC",
    "H": "CAC", "Q": "CAG", "N": "AAC", "K": "AAG", "D": "GAC",
    "E": "GAG", "C": "TGC", "W": "TGG", "R": "AGA", "G": "GGC",
}
