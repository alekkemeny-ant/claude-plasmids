"""Smart Mutation Design — curated GoF/LoF mutations + deterministic codon edits.

Provides:
- KNOWN_MUTATIONS: curated database of well-characterized gain/loss-of-function
  mutations for common oncogenes and tumor suppressors
- lookup_known_mutations(): query the curated DB
- apply_point_mutation(): deterministic single-codon substitution
- design_premature_stop(): introduce an in-frame stop codon

SAFETY: All sequence modifications are deterministic single-codon swaps.
The input sequence is unchanged except at the ONE specified codon position.
The replacement codon comes from PREFERRED_CODONS (most-used human codons).
No DNA is synthesized or generated — this is targeted editing only.
"""

from typing import Optional

# ── Standard genetic code (local copy to avoid cross-module import) ────

_CODON_TABLE: dict[str, str] = {
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

# Most-used human codon per amino acid (from Kazusa CUG data)
PREFERRED_CODONS: dict[str, str] = {
    "F": "TTC", "L": "CTG", "I": "ATC", "M": "ATG", "V": "GTG",
    "S": "AGC", "P": "CCC", "T": "ACC", "A": "GCC", "Y": "TAC",
    "H": "CAC", "Q": "CAG", "N": "AAC", "K": "AAG", "D": "GAC",
    "E": "GAG", "C": "TGC", "W": "TGG", "R": "AGG", "G": "GGC",
    "*": "TGA",
}


def _translate(dna: str) -> str:
    """Local DNA→AA helper. Translates full length, no stop-at-first-stop."""
    dna = dna.upper().replace(" ", "").replace("\n", "")
    aa = []
    for i in range(0, len(dna) - 2, 3):
        aa.append(_CODON_TABLE.get(dna[i:i+3], "X"))
    return "".join(aa)


# ── Curated mutation database ──────────────────────────────────────────

KNOWN_MUTATIONS: dict[str, list[dict]] = {
    "BRAF": [
        {"mutation": "V600E", "type": "GoF", "phenotype": "Constitutive MEK/ERK activation", "reference": "PMID:12068308", "codon_change": "GTG>GAG"},
        {"mutation": "V600K", "type": "GoF", "phenotype": "Constitutive kinase activation", "reference": "PMID:12068308", "codon_change": "GTG>AAG"},
    ],
    "KRAS": [
        {"mutation": "G12D", "type": "GoF", "phenotype": "Impaired GTP hydrolysis, constitutively active", "reference": "PMID:6092920", "codon_change": "GGT>GAT"},
        {"mutation": "G12V", "type": "GoF", "phenotype": "Impaired GTP hydrolysis, constitutively active", "reference": "PMID:6092920", "codon_change": "GGT>GTT"},
        {"mutation": "G12C", "type": "GoF", "phenotype": "Impaired GTP hydrolysis, drugable by sotorasib", "reference": "PMID:31666701", "codon_change": "GGT>TGT"},
        {"mutation": "G13D", "type": "GoF", "phenotype": "Impaired GTP hydrolysis", "reference": "PMID:3034404", "codon_change": "GGC>GAC"},
        {"mutation": "Q61H", "type": "GoF", "phenotype": "Impaired GTP hydrolysis", "reference": "", "codon_change": "CAA>CAT"},
    ],
    "TP53": [
        {"mutation": "R175H", "type": "LoF", "phenotype": "Structural mutation, DNA-binding domain disruption", "reference": "PMID:8479525", "codon_change": "CGC>CAC"},
        {"mutation": "R248Q", "type": "LoF", "phenotype": "Contact mutation, loss of DNA binding", "reference": "PMID:8479525", "codon_change": "CGG>CAG"},
        {"mutation": "R248W", "type": "LoF", "phenotype": "Contact mutation, loss of DNA binding", "reference": "PMID:8479525", "codon_change": "CGG>TGG"},
        {"mutation": "R273H", "type": "LoF", "phenotype": "Contact mutation, loss of DNA binding", "reference": "PMID:8479525", "codon_change": "CGT>CAT"},
        {"mutation": "R273C", "type": "LoF", "phenotype": "Contact mutation, loss of DNA binding", "reference": "", "codon_change": "CGT>TGT"},
        {"mutation": "G245S", "type": "LoF", "phenotype": "Structural mutation", "reference": "", "codon_change": "GGC>AGC"},
    ],
    "EGFR": [
        {"mutation": "L858R", "type": "GoF", "phenotype": "Constitutive kinase activation, TKI-sensitive", "reference": "PMID:15118073", "codon_change": "CTG>CGG"},
        {"mutation": "T790M", "type": "GoF", "phenotype": "Gatekeeper mutation, TKI resistance", "reference": "PMID:15788700", "codon_change": "ACG>ATG"},
    ],
    "PTEN": [
        {"mutation": "R130G", "type": "LoF", "phenotype": "Phosphatase-dead, PI3K/AKT hyperactivation", "reference": "PMID:9072974", "codon_change": "CGA>GGA"},
        {"mutation": "R130Q", "type": "LoF", "phenotype": "Phosphatase-dead", "reference": "", "codon_change": "CGA>CAA"},
        {"mutation": "C124S", "type": "LoF", "phenotype": "Catalytic-dead (active site Cys)", "reference": "PMID:9616126", "codon_change": "TGT>TCT"},
    ],
    "PIK3CA": [
        {"mutation": "H1047R", "type": "GoF", "phenotype": "Kinase domain, increased lipid kinase activity", "reference": "PMID:15289330", "codon_change": "CAT>CGT"},
        {"mutation": "E545K", "type": "GoF", "phenotype": "Helical domain, increased activity", "reference": "PMID:15289330", "codon_change": "GAG>AAG"},
        {"mutation": "E542K", "type": "GoF", "phenotype": "Helical domain, increased activity", "reference": "PMID:15289330", "codon_change": "GAA>AAA"},
    ],
    "IDH1": [
        {"mutation": "R132H", "type": "GoF", "phenotype": "Neomorphic 2-hydroxyglutarate production", "reference": "PMID:19228619", "codon_change": "CGT>CAT"},
        {"mutation": "R132C", "type": "GoF", "phenotype": "Neomorphic 2-HG production", "reference": "PMID:19228619", "codon_change": "CGT>TGT"},
    ],
    "IDH2": [
        {"mutation": "R172K", "type": "GoF", "phenotype": "Neomorphic 2-HG production", "reference": "PMID:19228619", "codon_change": "AGG>AAG"},
        {"mutation": "R140Q", "type": "GoF", "phenotype": "Neomorphic 2-HG production", "reference": "", "codon_change": "CGG>CAG"},
    ],
    "NRAS": [
        {"mutation": "Q61K", "type": "GoF", "phenotype": "Impaired GTP hydrolysis", "reference": "", "codon_change": "CAA>AAA"},
        {"mutation": "Q61R", "type": "GoF", "phenotype": "Impaired GTP hydrolysis", "reference": "", "codon_change": "CAA>CGA"},
        {"mutation": "G12D", "type": "GoF", "phenotype": "Impaired GTP hydrolysis", "reference": "", "codon_change": "GGT>GAT"},
    ],
    "CTNNB1": [
        {"mutation": "S33Y", "type": "GoF", "phenotype": "Beta-catenin stabilization (loss of degron)", "reference": "PMID:9065401", "codon_change": "TCT>TAT"},
        {"mutation": "S37F", "type": "GoF", "phenotype": "Beta-catenin stabilization", "reference": "", "codon_change": "TCT>TTT"},
        {"mutation": "T41A", "type": "GoF", "phenotype": "Beta-catenin stabilization", "reference": "", "codon_change": "ACC>GCC"},
    ],
    "AKT1": [
        {"mutation": "E17K", "type": "GoF", "phenotype": "Membrane-localized, constitutively active", "reference": "PMID:17611497", "codon_change": "GAG>AAG"},
    ],
    "FBXW7": [
        {"mutation": "R465C", "type": "LoF", "phenotype": "Substrate recognition loss (MYC, cyclin E accumulation)", "reference": "", "codon_change": "CGT>TGT"},
        {"mutation": "R505C", "type": "LoF", "phenotype": "Substrate recognition loss", "reference": "", "codon_change": "CGC>TGC"},
    ],
    "RB1": [
        {"mutation": "R661W", "type": "LoF", "phenotype": "Pocket domain disruption, E2F release", "reference": "", "codon_change": "CGG>TGG"},
    ],
    "MYC": [
        {"mutation": "T58A", "type": "GoF", "phenotype": "Degron-dead, protein stabilization", "reference": "PMID:10949026", "codon_change": "ACC>GCC"},
    ],
}


def lookup_known_mutations(gene_symbol: str, mutation_type: Optional[str] = None) -> list[dict]:
    """Look up curated GoF/LoF mutations for a gene.

    Args:
        gene_symbol: Gene symbol (case-insensitive, e.g., "BRAF", "braf")
        mutation_type: Optional filter — "GoF" or "LoF" (case-insensitive)

    Returns:
        List of mutation dicts. Empty list if gene not in database.
    """
    gene_upper = gene_symbol.upper()
    muts = KNOWN_MUTATIONS.get(gene_upper, [])
    if mutation_type:
        mt = mutation_type.lower()
        muts = [m for m in muts if m["type"].lower() == mt]
    return muts


def apply_point_mutation(dna_seq: str, aa_position: int, new_aa: str) -> dict:
    """Apply a single amino-acid substitution via deterministic codon swap.

    The input sequence is preserved EXCEPT at one codon, which is
    replaced with the most-used human codon for new_aa (from PREFERRED_CODONS).

    Args:
        dna_seq: Input CDS DNA sequence (must start in-frame at position 0)
        aa_position: 1-indexed amino acid position to mutate
        new_aa: Single-letter code for the replacement amino acid (or '*' for stop)

    Returns:
        {
            "sequence": str,  # mutated DNA, same length as input
            "original_aa": str,
            "new_aa": str,
            "original_codon": str,
            "new_codon": str,
            "dna_position": int,  # 0-indexed start of the swapped codon
            "aa_position": int,   # 1-indexed AA position (echoed)
        }

    Raises:
        ValueError: if aa_position out of range, new_aa invalid, or seq length
                    not a multiple of 3 covering aa_position.
    """
    seq = dna_seq.upper().replace(" ", "").replace("\n", "")
    new_aa = new_aa.upper()

    if new_aa not in PREFERRED_CODONS:
        raise ValueError(f"Invalid amino acid code: '{new_aa}'. Must be one of {sorted(PREFERRED_CODONS.keys())}.")

    if aa_position < 1:
        raise ValueError(f"aa_position must be ≥1 (1-indexed), got {aa_position}")

    dna_pos = (aa_position - 1) * 3
    if dna_pos + 3 > len(seq):
        aa_len = len(seq) // 3
        raise ValueError(f"aa_position {aa_position} out of range — sequence has only {aa_len} codons.")

    original_codon = seq[dna_pos:dna_pos+3]
    original_aa = _CODON_TABLE.get(original_codon, "X")
    new_codon = PREFERRED_CODONS[new_aa]

    mutated = seq[:dna_pos] + new_codon + seq[dna_pos+3:]

    return {
        "sequence": mutated,
        "original_aa": original_aa,
        "new_aa": new_aa,
        "original_codon": original_codon,
        "new_codon": new_codon,
        "dna_position": dna_pos,
        "aa_position": aa_position,
    }


def design_premature_stop(dna_seq: str, position_fraction: float = 0.1) -> dict:
    """Introduce an in-frame stop codon early in the CDS for loss-of-function.

    Replaces a single codon at approximately position_fraction through
    the CDS with TGA (stop). Skips the first codon (preserves ATG).

    Args:
        dna_seq: Input CDS DNA sequence (in-frame from position 0)
        position_fraction: Where to place the stop (0-1, default 0.1 = 10% in)

    Returns:
        {
            "sequence": str,          # mutated DNA, same length as input
            "stop_position_aa": int,  # 1-indexed AA position of the new stop
            "stop_position_dna": int, # 0-indexed DNA position
            "original_codon": str,
            "original_aa": str,
        }

    Raises:
        ValueError: if sequence too short (<6 codons) or fraction out of [0,1]
    """
    seq = dna_seq.upper().replace(" ", "").replace("\n", "")
    n_codons = len(seq) // 3

    if n_codons < 6:
        raise ValueError(f"Sequence too short ({n_codons} codons) for premature stop.")
    if not (0 <= position_fraction <= 1):
        raise ValueError(f"position_fraction must be in [0, 1], got {position_fraction}")

    # Target codon index (0-based), min 1 to skip ATG, max n_codons-2 to
    # leave at least one codon after (keeps seq length >= minimal CDS)
    target = max(1, min(n_codons - 2, round(position_fraction * n_codons)))
    dna_pos = target * 3
    original_codon = seq[dna_pos:dna_pos+3]
    original_aa = _CODON_TABLE.get(original_codon, "X")

    mutated = seq[:dna_pos] + "TGA" + seq[dna_pos+3:]

    return {
        "sequence": mutated,
        "stop_position_aa": target + 1,  # 1-indexed
        "stop_position_dna": dna_pos,
        "original_codon": original_codon,
        "original_aa": original_aa,
    }


def parse_mutation_notation(notation: str) -> Optional[dict]:
    """Parse standard mutation notation like "V600E" or "R175H".

    Returns {"original_aa": str, "position": int, "new_aa": str}
    or None if unparseable.
    """
    import re
    m = re.match(r"^([A-Z])(\d+)([A-Z*])$", notation.upper())
    if not m:
        return None
    return {
        "original_aa": m.group(1),
        "position": int(m.group(2)),
        "new_aa": m.group(3),
    }
