#!/usr/bin/env python3
"""
FPbase Integration

FPbase (fpbase.org) is the canonical reference database for fluorescent
proteins. This module retrieves DNA coding sequences for engineered FPs
like mRuby, mScarlet, etc. that are NOT in NCBI Gene (since they're
engineered proteins, not natural genes).

Using FPbase's REST API:
  https://www.fpbase.org/api/proteins/?name__iexact=<name>&format=json
"""

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FPBASE_API = "https://www.fpbase.org/api/proteins/"
_HTTP_TIMEOUT = 10

# Standard codon table for reverse-translation (human-preferred codons).
# Used ONLY as a last resort when FPbase has the protein but no DNA seq.
# When this is used, the result is flagged as reverse-translated.
_CODON_TABLE = {
    "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC",
    "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    "*": "TGA",
}

# Common FP name patterns — used to decide when to try FPbase before NCBI.
# Covers mCherry/mRuby/mScarlet (m + capital), eGFP/eYFP, tdTomato, etc.
_FP_NAME_PATTERNS = [
    r"^m[A-Z]",           # mCherry, mRuby, mScarlet, mNeonGreen, mTagBFP
    r"^e[A-Z]?[A-Z]FP",   # eGFP, eYFP, eCFP
    r"^E[A-Z]FP",         # EGFP, EYFP
    r"^td[A-Z]",          # tdTomato, tdKatushka
    r"^sf[A-Z]",          # sfGFP
    r"^i[A-Z]FP",         # iRFP
    r"^d[A-Z]",           # dTomato, dKatushka
    r"[A-Z]FP\d*$",       # GFP, RFP, YFP, CFP, BFP (with optional version)
    r"^Ds[A-Z]",          # DsRed
    r"^Venus$",
    r"^Citrine$",
    r"^Cerulean",
    r"^Turquoise",
    r"^Emerald",
    r"^Clover",
]

# Curated list of known FP names for direct matching (supplements patterns).
_KNOWN_FP_NAMES = {
    "mruby", "mruby2", "mruby3",
    "mcherry", "mscarlet", "mscarlet-i", "mscarlet3",
    "mneongreen", "mturquoise2", "mtagbfp2",
    "tdtomato", "dsred", "dsred2", "dsred-express",
    "egfp", "eyfp", "ecfp", "ebfp", "ebfp2",
    "sfgfp", "gfp", "yfp", "cfp", "rfp",
    "venus", "citrine", "cerulean", "emerald", "clover",
    "irfp", "irfp670", "irfp720",
    "mkate", "mkate2", "mko", "morange", "morange2",
    "mplum", "mraspberry", "mstrawberry", "mtfp1",
}


def looks_like_fp_name(name: str) -> bool:
    """Check if a name looks like a fluorescent protein.

    Used to decide whether FPbase should be tried before NCBI Gene
    in the get_insert_by_id fallback chain.
    """
    name_stripped = name.strip()
    if name_stripped.lower().replace("-", "").replace("_", "") in _KNOWN_FP_NAMES:
        return True
    for pat in _FP_NAME_PATTERNS:
        if re.search(pat, name_stripped):
            return True
    return False


def _normalize_fp_query(name: str) -> str:
    """Normalize an FP name for querying FPbase.

    FPbase uses names like 'mRuby', 'mRuby2', 'mScarlet-I'.
    """
    return name.strip()


def search_fpbase(name: str, limit: int = 5) -> list[dict]:
    """Search FPbase for fluorescent proteins by name.

    Args:
        name: FP name (e.g., "mRuby", "mScarlet")
        limit: Max results to return

    Returns:
        List of dicts with keys: slug, name, seq (AA), states
    """
    query = _normalize_fp_query(name)
    results: list[dict] = []

    # Try exact match first, then contains
    for param, val in [("name__iexact", query), ("name__icontains", query)]:
        try:
            resp = requests.get(
                FPBASE_API,
                params={param: val, "format": "json"},
                timeout=_HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            for item in data:
                slug = item.get("slug")
                if slug and not any(r.get("slug") == slug for r in results):
                    results.append({
                        "slug": slug,
                        "name": item.get("name", ""),
                        "seq": item.get("seq", ""),  # amino acid seq
                        "ex_max": _first_state_attr(item, "ex_max"),
                        "em_max": _first_state_attr(item, "em_max"),
                    })
                    if len(results) >= limit:
                        return results
            # If exact match found something, don't bother with contains
            if results and param == "name__iexact":
                return results
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"FPbase search ({param}={val}) failed: {e}")

    return results


def _first_state_attr(protein: dict, attr: str) -> Optional[int]:
    """Pull an attribute from the first 'state' of an FPbase protein."""
    states = protein.get("states", [])
    if states and isinstance(states, list):
        return states[0].get(attr)
    return None


def fetch_fpbase_sequence(name_or_slug: str) -> Optional[dict]:
    """Fetch the DNA coding sequence for a fluorescent protein from FPbase.

    FPbase stores amino-acid sequences for all proteins. Many entries also
    include a DNA sequence via the GraphQL API or the protein detail
    endpoint. When DNA is unavailable, this returns None (we do NOT
    reverse-translate silently — that would violate the "no hallucinated
    sequence" principle unless explicitly flagged).

    Args:
        name_or_slug: FP name (e.g., "mRuby") or FPbase slug (e.g., "mruby")

    Returns:
        Dict with: name, slug, sequence (DNA), aa_sequence, length,
        source, ex_max, em_max, sequence_origin ("fpbase_dna" or
        "reverse_translated"). None if not found or no DNA available.
    """
    query = _normalize_fp_query(name_or_slug)

    # Step 1: search to get the slug + AA seq
    hits = search_fpbase(query, limit=3)
    if not hits:
        logger.info(f"FPbase: no results for '{query}'")
        return None

    # Prefer exact name match
    best = hits[0]
    for h in hits:
        if h.get("name", "").lower() == query.lower():
            best = h
            break

    slug = best.get("slug")
    aa_seq = best.get("seq", "")
    if not slug:
        return None

    # Step 2: try to fetch DNA sequence from the protein detail endpoint.
    # The REST list endpoint only returns AA seq; the GraphQL API and
    # per-protein page may have the published DNA sequence.
    dna_seq = _fetch_dna_via_graphql(slug)

    # Step 3: if no DNA from FPbase, try reverse-translation from AA
    # (flagged clearly — this is a synthetic codon-optimized sequence,
    # not the published one).
    sequence_origin = None
    if dna_seq:
        sequence_origin = "fpbase_dna"
    elif aa_seq:
        dna_seq = _reverse_translate(aa_seq)
        if dna_seq:
            sequence_origin = "reverse_translated"
            logger.warning(
                f"FPbase: no published DNA for '{slug}'; reverse-translated "
                f"from AA sequence using human-preferred codons. This is "
                f"functionally equivalent but NOT the original published "
                f"sequence."
            )

    if not dna_seq:
        logger.info(f"FPbase: found '{slug}' but no DNA sequence available")
        return None

    return {
        "name": best.get("name", slug),
        "slug": slug,
        "sequence": dna_seq,
        "aa_sequence": aa_seq,
        "length": len(dna_seq),
        "source": "FPbase",
        "sequence_origin": sequence_origin,
        "ex_max": best.get("ex_max"),
        "em_max": best.get("em_max"),
        "url": f"https://www.fpbase.org/protein/{slug}/",
    }


def _fetch_dna_via_graphql(slug: str) -> Optional[str]:
    """Try to fetch the DNA sequence via the FPbase GraphQL API.

    Not all proteins have a DNA sequence stored. Returns None if
    unavailable or on any error.
    """
    graphql_url = "https://www.fpbase.org/graphql/"
    # The GraphQL schema exposes `seq` (AA). DNA is not reliably
    # exposed in the public API. We attempt a best-effort detail fetch
    # via the REST detail endpoint and look for a dna_seq field.
    try:
        resp = requests.get(
            f"{FPBASE_API}{slug}/",
            params={"format": "json"},
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            # FPbase detail responses sometimes include 'seq' (AA only)
            # but not DNA. Check for any DNA-like field.
            for key in ("dna_seq", "dna", "cds", "nucleotide_seq"):
                val = data.get(key)
                if val and _is_dna(val):
                    return val.upper().replace(" ", "").replace("\n", "")
    except (requests.RequestException, ValueError):
        pass

    # Try GraphQL as a secondary route (some fields only exposed here)
    try:
        query = """
        query($slug: String!) {
          protein(id: $slug) {
            name
            seq
          }
        }
        """
        resp = requests.post(
            graphql_url,
            json={"query": query, "variables": {"slug": slug}},
            timeout=_HTTP_TIMEOUT,
        )
        # GraphQL seq is AA, not DNA — this route doesn't help for DNA
        # but we keep the call in case the schema evolves.
    except requests.RequestException:
        pass

    return None


def _is_dna(s: str) -> bool:
    """Check if a string looks like a DNA sequence."""
    if not s or len(s) < 20:
        return False
    s_clean = s.upper().replace(" ", "").replace("\n", "")
    valid = set("ACGTN")
    return all(c in valid for c in s_clean) and len(s_clean) >= 20


def _reverse_translate(aa_seq: str) -> Optional[str]:
    """Reverse-translate an amino-acid sequence to DNA using human-preferred codons.

    Adds ATG start and TGA stop. Returns None if the AA sequence
    contains unknown residues.
    """
    aa_seq = aa_seq.strip().upper()
    if not aa_seq:
        return None

    # Strip leading M if present (we add ATG explicitly)
    if aa_seq.startswith("M"):
        aa_seq = aa_seq[1:]
    # Strip trailing stop if present
    if aa_seq.endswith("*"):
        aa_seq = aa_seq[:-1]

    codons = ["ATG"]  # start codon
    for aa in aa_seq:
        codon = _CODON_TABLE.get(aa)
        if not codon:
            logger.warning(f"Unknown amino acid '{aa}' in FP sequence; cannot reverse-translate")
            return None
        codons.append(codon)
    codons.append("TGA")  # stop codon

    return "".join(codons)
