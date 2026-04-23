#!/usr/bin/env python3
"""
Plasmid Library Core Functions

Core functionality for the plasmid library, independent of MCP.
Can be used directly for testing or integration.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import os, sys

try:
    from assembler import reverse_complement as rc
    from assembler import validate_dna
except ModuleNotFoundError:
    from src.assembler import reverse_complement as rc
    from src.assembler import validate_dna

# Library path
LIBRARY_PATH = Path(__file__).parent.parent / "library"

logger = logging.getLogger(__name__)

# Optional Addgene integration (gracefully degrades if not available)
# Try relative import first (when loaded as a package), then fall back to
# absolute import (when src/ is on sys.path directly, as app.py does).
try:
    from .addgene_integration import AddgeneClient, AddgeneLibraryIntegration
    ADDGENE_AVAILABLE = True
except ImportError:
    try:
        from addgene_integration import AddgeneClient, AddgeneLibraryIntegration
        ADDGENE_AVAILABLE = True
    except ImportError:
        ADDGENE_AVAILABLE = False

# Optional NCBI integration
try:
    from .ncbi_integration import (
        fetch_gene_sequence as _ncbi_fetch_gene,
        search_gene as _ncbi_search_gene,
    )
    NCBI_AVAILABLE = True
except ImportError:
    try:
        from ncbi_integration import (
            fetch_gene_sequence as _ncbi_fetch_gene,
            search_gene as _ncbi_search_gene,
        )
        NCBI_AVAILABLE = True
    except ImportError:
        NCBI_AVAILABLE = False

# Optional user library (BYOL — bring your own library)
try:
    from .user_library import load_user_backbones, load_user_inserts
    USER_LIBRARY_AVAILABLE = True
except ImportError:
    try:
        from user_library import load_user_backbones, load_user_inserts
        USER_LIBRARY_AVAILABLE = True
    except ImportError:
        USER_LIBRARY_AVAILABLE = False

# Optional custom annotation DB (BYOA — bring your own annotations)
try:
    from .custom_annotations import setup_custom_annotations, query_custom_db, merge_annotation_results
    _CUSTOM_ANNOTATIONS_AVAILABLE = True
except ImportError:
    try:
        from custom_annotations import setup_custom_annotations, query_custom_db, merge_annotation_results
        _CUSTOM_ANNOTATIONS_AVAILABLE = True
    except ImportError:
        _CUSTOM_ANNOTATIONS_AVAILABLE = False

if _CUSTOM_ANNOTATIONS_AVAILABLE:
    setup_custom_annotations()

# Optional FPbase integration (fluorescent proteins — engineered, not in NCBI Gene)
try:
    from .fpbase_integration import (
        fetch_fpbase_sequence as _fpbase_fetch,
        looks_like_fp_name as _looks_like_fp,
    )
    FPBASE_AVAILABLE = True
except ImportError:
    try:
        from fpbase_integration import (
            fetch_fpbase_sequence as _fpbase_fetch,
            looks_like_fp_name as _looks_like_fp,
        )
        FPBASE_AVAILABLE = True
    except ImportError:
        FPBASE_AVAILABLE = False


# ── Read-only mode (for evals / parallel-safe operation) ───────────────
# When enabled, library lookups still work (incl. Addgene/NCBI/FPbase fallbacks)
# but results are NOT written back to library/*.json. Prevents eval side-effects
# and eliminates the unprotected read-modify-write race under parallelism.
_LIBRARY_READONLY = False

# ── Test fixture injection ───────────────────────────────────────────────
# Extra entries registered via register_test_fixtures() are appended to
# load_backbones() / load_inserts() results at runtime.  Intended for eval
# and test cases that need sequences not present in the curated library.
# Call clear_test_fixtures() after the case to avoid cross-contamination.
_EXTRA_BACKBONES: list[dict] = []
_EXTRA_INSERTS: list[dict] = []


def register_test_fixtures(backbones: list[dict] = (), inserts: list[dict] = ()) -> None:
    """Append extra backbone/insert entries for the duration of an eval run.

    Entries are added to the in-memory lists checked by load_backbones() and
    load_inserts(); the on-disk JSON files are never touched.
    """
    global _EXTRA_BACKBONES, _EXTRA_INSERTS
    _EXTRA_BACKBONES = list(backbones)
    _EXTRA_INSERTS = list(inserts)


def clear_test_fixtures() -> None:
    """Remove all fixture entries registered via register_test_fixtures()."""
    global _EXTRA_BACKBONES, _EXTRA_INSERTS
    _EXTRA_BACKBONES = []
    _EXTRA_INSERTS = []


def set_library_readonly(readonly: bool = True) -> None:
    """Disable writes to library/*.json. Call before running evals."""
    global _LIBRARY_READONLY
    _LIBRARY_READONLY = readonly


def _load_builtin_backbones() -> dict:
    """Load built-in backbone library from JSON file (no runtime extensions).

    This is the cache-write-safe loader. The Addgene auto-cache path uses
    this to re-read fresh from disk before appending, ensuring neither
    user-library entries nor test fixtures leak into backbones.json.
    """
    with open(LIBRARY_PATH / "backbones.json", "r") as f:
        return json.load(f)


def _load_builtin_inserts() -> dict:
    """Load built-in insert library from JSON file (no runtime extensions)."""
    with open(LIBRARY_PATH / "inserts.json", "r") as f:
        return json.load(f)


def load_backbones() -> dict:
    """Load backbone library: built-in + test fixtures + user library.

    User entries (from $PLASMID_USER_LIBRARY/backbones/) are appended with
    `user:` ID prefix. Callers that persist to disk must use
    _load_builtin_backbones instead to avoid writing runtime-only entries.
    """
    data = _load_builtin_backbones()
    if _EXTRA_BACKBONES:
        data["backbones"] = data["backbones"] + _EXTRA_BACKBONES
    if USER_LIBRARY_AVAILABLE:
        user_entries = load_user_backbones()
        if user_entries:
            data["backbones"] = data["backbones"] + user_entries
    return data


def load_inserts() -> dict:
    """Load insert library: built-in + test fixtures + user library."""
    data = _load_builtin_inserts()
    if _EXTRA_INSERTS:
        data["inserts"] = data["inserts"] + _EXTRA_INSERTS
    if USER_LIBRARY_AVAILABLE:
        user_entries = load_user_inserts()
        if user_entries:
            data["inserts"] = data["inserts"] + user_entries
    return data


def normalize_name(name: str) -> str:
    """Normalize a plasmid/insert name for matching.

    Preserves polarity indicators: (+)/(-) and trailing +/- are converted
    to 'plus'/'minus' so that pcDNA3.1(+) and pcDNA3.1(-) remain distinct.
    """
    name = name.replace('(+)', 'plus').replace('(-)', 'minus')
    name = re.sub(r'\+\s*$', 'plus', name)
    name = re.sub(r'-\s*$', 'minus', name)
    return re.sub(r'[^a-z0-9]', '', name.lower())


# Known promoter properties for natural language search
_PROMOTER_PROPERTIES = {
    "cmv": "strong constitutive",
    "cag": "very strong constitutive",
    "ef1a": "strong constitutive",
    "pgk": "moderate constitutive",
    "ubiquitin": "moderate constitutive",
    "sv40": "moderate constitutive",
    "t7": "strong inducible",
    "tac": "strong inducible",
    "lac": "moderate inducible",
    "u6": "constitutive pol-iii",
    "cbh": "strong constitutive",
    "ltr": "moderate constitutive",
}

# ── Bespoke promoter detection ───────────────────────────────────────────
#
# KNOWN_PROMOTERS: if the user requests one of these, it's a standard part
# available in the library or easily fetchable. Anything NOT in this set is
# a "bespoke" promoter → agent should offer: (a) research mode, (b) user
# pastes sequence, or (c) fetch native upstream region from NCBI genomic.
KNOWN_PROMOTERS: frozenset[str] = frozenset({
    # Mammalian constitutive
    "cmv", "ef1a", "ef1alpha", "ef1", "cag", "cagg", "pgk", "sv40", "ubc",
    "ubiquitin", "cbh", "cba", "rsv",
    # Mammalian Pol III (for shRNA/gRNA)
    "u6", "h1", "7sk",
    # Tissue-specific (common)
    "synapsin", "syn", "camkii", "camk2a", "gfap", "mbp", "alb", "albumin",
    # Bacterial
    "t7", "sp6", "t3", "lac", "tac", "trc", "arap", "arapbad", "arabad",
    "tet", "ptet", "rhap", "rhab",
    # Inducible
    "tre", "tre3g", "tetO",
    # Viral/LTR
    "ltr", "5ltr",
})


def is_known_promoter(name: str) -> bool:
    """Check if a promoter name is a known/standard promoter.

    Normalizes the name (lowercase, strip non-alphanumeric) before lookup.
    Returns False for anything not in KNOWN_PROMOTERS — this triggers the
    "bespoke promoter" workflow in the agent.
    """
    normalized = re.sub(r'[^a-z0-9]', '', name.lower())
    return normalized in KNOWN_PROMOTERS


# ── Disambiguation aids ──────────────────────────────────────────────────
#
# GENE_FAMILIES: ambiguous query → list of specific family members.
# When a user asks for "TRAF" or "H2B", the agent should present these
# options rather than auto-picking one.
GENE_FAMILIES = {
    "TRAF": [
        "TRAF1", "TRAF2", "TRAF3", "TRAF4", "TRAF5", "TRAF6", "TRAF7",
    ],
    "H2B": [
        # Human histone H2B has 20+ variants. Most common in cell biology:
        "H2BC21",  # HIST1H2BJ — broadly expressed, common fusion choice
        "H2BC11",  # HIST1H2BK
        "H2BC12",  # HIST1H2BN
        "H2BC4",   # HIST1H2BC
        "H2BC5",   # HIST1H2BD
        "HIST1H2BJ",  # legacy alias for H2BC21
    ],
    "RFP": [
        "mCherry", "tdTomato", "mScarlet", "mScarlet-I", "DsRed",
        "DsRed2", "mRFP1", "TagRFP", "mKate2",
    ],
    "GFP": [
        "EGFP", "sfGFP", "mNeonGreen", "mClover3", "GFP",
    ],
    "YFP": [
        "EYFP", "Venus", "Citrine", "mVenus",
    ],
    "CFP": [
        "ECFP", "Cerulean", "mTurquoise2", "mCerulean3",
    ],
    # TNF receptor associated factor → same as TRAF
    "TNF RECEPTOR ASSOCIATED FACTOR": [
        "TRAF1", "TRAF2", "TRAF3", "TRAF4", "TRAF5", "TRAF6", "TRAF7",
    ],
}

# CELL_LINE_SPECIES: common cell line name → species.
# Used so the agent can infer organism when the user only mentions a cell line.
CELL_LINE_SPECIES = {
    # Human
    "HEK293": "human", "HEK293T": "human", "HEK-293": "human",
    "293T": "human", "293": "human",
    "HELA": "human", "HeLa": "human",
    "A549": "human", "U2OS": "human", "U-2 OS": "human",
    "HCT116": "human", "MCF7": "human", "MCF-7": "human",
    "JURKAT": "human", "K562": "human", "THP-1": "human", "THP1": "human",
    "HUH7": "human", "HUH-7": "human", "SH-SY5Y": "human",
    # Mouse
    "RAW264": "mouse", "RAW 264.7": "mouse", "RAW264.7": "mouse",
    "RAW 264": "mouse",
    "NIH3T3": "mouse", "NIH 3T3": "mouse", "3T3": "mouse",
    "MEF": "mouse", "C2C12": "mouse", "4T1": "mouse",
    "B16": "mouse", "CT26": "mouse", "MC38": "mouse",
    # Hamster
    "CHO": "hamster", "CHO-K1": "hamster",
    # Monkey
    "COS-7": "monkey", "COS7": "monkey", "VERO": "monkey",
    # Rat
    "PC12": "rat", "RAT1": "rat",
    # Dog
    "MDCK": "dog",
    # Insect
    "SF9": "insect", "SF21": "insect", "HIGH FIVE": "insect", "S2": "fly",
}


def check_gene_family_ambiguity(query: str) -> Optional[dict]:
    """Check if a query is an ambiguous gene-family name.

    Args:
        query: Gene name the user provided

    Returns:
        None if unambiguous, or a dict with:
        - family: the family name matched
        - members: list of specific family members to present to the user
    """
    q_upper = query.strip().upper()
    for family, members in GENE_FAMILIES.items():
        if q_upper == family.upper():
            return {"family": family, "members": members}
    return None


def infer_species_from_cell_line(cell_line: str) -> Optional[str]:
    """Infer organism species from a cell line name.

    Args:
        cell_line: Cell line name (e.g., "HEK293", "RAW 264.7")

    Returns:
        Species string (e.g., "human", "mouse") or None if unknown
    """
    cl_upper = cell_line.strip().upper()
    # Try direct lookup (case-insensitive)
    for name, species in CELL_LINE_SPECIES.items():
        if name.upper() == cl_upper:
            return species
    # Try substring (e.g., "RAW 264.7 macrophages" -> "RAW 264.7")
    for name, species in CELL_LINE_SPECIES.items():
        if name.upper() in cl_upper:
            return species
    return None


def _backbone_searchable_text(backbone: dict) -> str:
    """Build a single lowercase string of all searchable backbone fields."""
    parts = [
        backbone.get("id", ""),
        backbone.get("name", ""),
        backbone.get("description", ""),
        backbone.get("bacterial_resistance", ""),
        backbone.get("mammalian_selection", ""),
        backbone.get("promoter", ""),
        backbone.get("organism", ""),
        backbone.get("origin", ""),
    ]
    parts.extend(backbone.get("aliases", []))
    # Add promoter property descriptors for natural language matching
    promoter_parts = (backbone.get("promoter") or "").lower().split()
    promoter = promoter_parts[0].rstrip(",") if promoter_parts else ""
    if promoter in _PROMOTER_PROPERTIES:
        parts.append(_PROMOTER_PROPERTIES[promoter])
    # Add expression type descriptors based on vector class
    desc_lower = (backbone.get("description") or "").lower()
    organism = (backbone.get("organism") or "").lower()
    if any(kw in desc_lower for kw in ("lentiv", "retrovir", "aav", "gene therapy")):
        parts.append("stable expression")
    if organism == "mammalian" and "lentiv" not in desc_lower and "retrovir" not in desc_lower:
        parts.append("transient expression")
    return " ".join(p or "" for p in parts).lower()


def search_backbones(query: str, organism: Optional[str] = None, promoter: Optional[str] = None) -> list[dict]:
    """
    Search for backbones matching the query.

    Supports multi-term queries: all terms must match somewhere across the
    searchable fields (name, aliases, description, resistance, selection,
    promoter, organism, origin).

    Args:
        query: Search term (name, feature, or keyword)
        organism: Filter by organism type (mammalian, bacterial, etc.)
        promoter: Filter by promoter type (CMV, T7, etc.)

    Returns:
        List of matching backbone dictionaries
    """
    data = load_backbones()
    results = []
    query_terms = query.lower().split()

    for backbone in data["backbones"]:
        searchable = _backbone_searchable_text(backbone)

        # All query terms must appear somewhere in the searchable text
        if not all(term in searchable for term in query_terms):
            continue

        # Apply filters
        if organism and backbone.get("organism", "").lower() != organism.lower():
            continue
        if promoter and promoter.lower() not in backbone.get("promoter", "").lower():
            continue
        results.append(backbone)

    return results


def search_inserts(query: str, category: Optional[str] = None) -> list[dict]:
    """
    Search for inserts matching the query.
    
    Args:
        query: Search term (name or keyword)
        category: Filter by category (fluorescent_protein, reporter, epitope_tag)
    
    Returns:
        List of matching insert dictionaries
    """
    data = load_inserts()
    results = []
    query_normalized = normalize_name(query)
    
    for insert in data["inserts"]:
        # Check name and aliases
        names_to_check = [insert["id"], insert["name"]] + insert.get("aliases", [])
        name_match = any(query_normalized in normalize_name(n) for n in names_to_check)
        
        # Check description
        desc_match = query.lower() in insert.get("description", "").lower()
        
        if name_match or desc_match:
            if category and insert.get("category", "").lower() != category.lower():
                continue
            results.append(insert)
    
    return results


def get_backbone_by_id(backbone_id: str) -> Optional[dict]:
    """
    Get a specific backbone by ID or alias.

    Checks the local library first. If not found and Addgene integration is
    available, searches Addgene by name, fetches the GenBank file (sequence +
    feature annotations), and caches the result in backbones.json for future
    fast lookups.

    Args:
        backbone_id: Backbone identifier or alias

    Returns:
        Backbone dictionary or None if not found
    """
    data = load_backbones()
    id_normalized = normalize_name(backbone_id)

    for backbone in data["backbones"]:
        names_to_check = [backbone["id"]] + backbone.get("aliases", [])
        if any(normalize_name(n) == id_normalized for n in names_to_check):
            return backbone

    # ── Addgene fallback ──
    if not ADDGENE_AVAILABLE:
        return None

    try:
        logger.info(f"Backbone '{backbone_id}' not in local library, searching Addgene...")
        client = AddgeneClient()
        results = client.search(backbone_id, limit=5)
        if not results:
            logger.info(f"No Addgene results for '{backbone_id}'")
            return None

        # Pick the best match: prefer exact normalized name match, else first result
        best = results[0]
        exact_name_match = False
        for r in results:
            if normalize_name(r.get("name", "")) == id_normalized:
                best = r
                exact_name_match = True
                break

        addgene_id = best.get("addgene_id")
        if not addgene_id:
            return None

        logger.info(f"Found Addgene #{addgene_id} ({best.get('name', '?')}), fetching plasmid data...")

        plasmid = client.get_plasmid(addgene_id)
        if not plasmid:
            return None

        backbone = plasmid.to_backbone_dict()

        # If we only fuzzy-matched the name, do NOT silently cache.
        # Tag as unconfirmed so the agent can present it to the user first.
        # The agent should call import_addgene_to_library with the confirmed
        # addgene_id to commit it.
        if not exact_name_match:
            backbone["unconfirmed"] = True
            backbone["addgene_search_alternatives"] = [
                {"name": r.get("name"), "addgene_id": r.get("addgene_id")}
                for r in results[:5]
            ]
            logger.info(
                f"Addgene fuzzy match for '{backbone_id}' → #{addgene_id} "
                f"({backbone.get('name', '?')}). Returning unconfirmed; not caching."
            )
            return backbone

        # Exact-name match: cache to local library for future fast lookups.
        # Re-read built-in from disk — `data` above came from load_backbones()
        # which may include user-library entries or test fixtures we must not persist.
        if not _LIBRARY_READONLY:
            builtin = _load_builtin_backbones()
            builtin["backbones"].append(backbone)
            with open(LIBRARY_PATH / "backbones.json", "w") as f:
                json.dump(builtin, f, indent=2)

        logger.info(
            f"Cached Addgene #{addgene_id} as '{backbone['id']}' "
            f"({backbone.get('size_bp', '?')} bp, "
            f"{len(backbone.get('features', []))} features)"
        )
        return backbone

    except Exception as e:
        logger.warning(f"Addgene fallback failed for '{backbone_id}': {e}")
        return None


def get_insert_by_id(insert_id: str, organism: Optional[str] = None) -> Optional[dict]:
    """
    Get a specific insert by ID or alias.

    Fallback chain: Local library → FPbase (for FP-like names) → NCBI Gene.

    When the query is ambiguous (multiple species, gene family), returns a
    dict with "needs_disambiguation": True and "options" list — the caller
    should present these to the user rather than proceeding.

    Args:
        insert_id: Insert identifier, alias, FP name, or gene symbol
        organism: Optional organism for NCBI fallback (e.g., "human", "mouse")

    Returns:
        Insert dictionary, disambiguation dict, or None if not found
    """
    data = load_inserts()
    id_normalized = normalize_name(insert_id)

    # Prefer exact ID match, then fall back to alias match
    alias_match = None
    for insert in data["inserts"]:
        if normalize_name(insert["id"]) == id_normalized:
            return insert
        if alias_match is None:
            if any(normalize_name(a) == id_normalized for a in insert.get("aliases", [])):
                alias_match = insert
    if alias_match is not None:
        return alias_match

    # ── Gene family ambiguity check ──
    # Catch ambiguous family names (TRAF, H2B, RFP, ...) BEFORE hitting
    # remote databases. NCBI would return something for "H2B" but it'd be
    # an arbitrary variant — the user needs to choose.
    family_check = check_gene_family_ambiguity(insert_id)
    if family_check:
        logger.info(
            f"'{insert_id}' is an ambiguous gene family; "
            f"{len(family_check['members'])} members"
        )
        return {
            "needs_disambiguation": True,
            "reason": "gene_family",
            "query": insert_id,
            "family": family_check["family"],
            "members": family_check["members"],
        }

    # Skip remote fallback if the query doesn't look like a gene/FP name
    # (e.g., "pcDNA3.1(+)" contains parens/dots → backbone, not a gene)
    if not re.match(r'^[A-Za-z0-9_\-]+$', insert_id.strip()):
        return None

    alias_candidate = insert_id.strip()

    # ── FPbase fallback (for engineered fluorescent proteins) ──
    # Try FPbase FIRST when the name looks like an FP (mRuby, mScarlet, etc.)
    # These are engineered proteins — NCBI Gene won't have them as genes,
    # and a broader NCBI search can return wildly wrong results.
    if FPBASE_AVAILABLE and _looks_like_fp(insert_id):
        try:
            logger.info(f"Insert '{insert_id}' looks like an FP; trying FPbase...")
            fp_result = _fpbase_fetch(insert_id)
            if fp_result:
                if fp_result.get("no_dna"):
                    # FPbase found the protein but only has the amino-acid
                    # sequence, not DNA. We do NOT reverse-translate (that
                    # would synthesize sequence — against project invariant).
                    # Return a signal so the agent can tell the user what
                    # we found and ask for the DNA sequence.
                    #
                    # Importantly: return here rather than falling through to
                    # NCBI. An FP found on FPbase is confirmed engineered —
                    # NCBI Gene WILL return a wrong result if we try it.
                    logger.info(
                        f"FPbase confirmed '{fp_result['name']}' but no DNA; "
                        f"signaling agent to ask user for sequence"
                    )
                    return {
                        "needs_disambiguation": True,
                        "reason": "fpbase_no_dna",
                        "query": insert_id,
                        "fpbase_name": fp_result["name"],
                        "fpbase_url": fp_result.get("url"),
                        "aa_sequence": fp_result.get("aa_sequence"),
                        "aa_length": fp_result.get("aa_length"),
                        "ex_max": fp_result.get("ex_max"),
                        "em_max": fp_result.get("em_max"),
                    }

                # FPbase has DNA — build insert dict and cache
                insert = {
                    "id": fp_result["name"],
                    "name": fp_result["name"],
                    "aliases": [alias_candidate] if alias_candidate != fp_result["name"] else [],
                    "description": (
                        f"Fluorescent protein from FPbase. "
                        f"Ex/Em: {fp_result.get('ex_max','?')}/{fp_result.get('em_max','?')} nm. "
                        f"{fp_result.get('url','')}"
                    ),
                    "category": "fluorescent_protein",
                    "size_bp": fp_result["length"],
                    "sequence": fp_result["sequence"],
                    "source": "FPbase",
                    "fpbase_slug": fp_result.get("slug"),
                }
                # Cache to local library (FPbase DNA is canonical)
                existing_ids = {i["id"] for i in data["inserts"]}
                if insert["id"] not in existing_ids and not _LIBRARY_READONLY:
                    data["inserts"].append(insert)
                    with open(LIBRARY_PATH / "inserts.json", "w") as f:
                        json.dump(data, f, indent=2)
                logger.info(
                    f"Cached FPbase protein '{fp_result['name']}' ({fp_result['length']} bp)"
                )
                return insert
        except Exception as e:
            logger.warning(f"FPbase fallback failed for '{insert_id}': {e}")
            # fall through to NCBI

    # ── NCBI Gene fallback ──
    if not NCBI_AVAILABLE:
        return None

    try:
        logger.info(f"Insert '{insert_id}' not in library/FPbase, searching NCBI Gene...")
        result = _ncbi_fetch_gene(gene_symbol=insert_id, organism=organism)
        if not result:
            logger.info(f"No NCBI result for '{insert_id}'")
            return None

        # Check for ambiguity signal from fetch_gene_sequence
        if result.get("needs_disambiguation"):
            logger.info(
                f"NCBI search for '{insert_id}' returned multiple species; "
                f"disambiguation required"
            )
            return result  # pass disambiguation dict up to caller

        if not result.get("sequence"):
            logger.info(f"No NCBI CDS found for '{insert_id}'")
            return None

        store_alias = (
            alias_candidate != result["symbol"]
            and re.match(r'^[A-Za-z0-9_\-]+$', alias_candidate)
        )
        insert = {
            "id": result["symbol"],
            "name": result["symbol"],
            "aliases": [alias_candidate] if store_alias else [],
            "description": result.get("full_name", ""),
            "category": "gene",
            "size_bp": result["length"],
            "sequence": result["sequence"],
            "genbank_accession": result.get("accession"),
            "organism": result.get("organism", ""),
            "source": "NCBI",
        }

        # Cache to local library (skip if gene already cached).
        # Re-read built-in from disk — `data` above came from load_inserts()
        # which may include user-library entries or test fixtures we must not persist.
        if not _LIBRARY_READONLY:
            builtin = _load_builtin_inserts()
            existing_ids = {i["id"] for i in builtin["inserts"]}
            if insert["id"] not in existing_ids:
                builtin["inserts"].append(insert)
                with open(LIBRARY_PATH / "inserts.json", "w") as f:
                    json.dump(builtin, f, indent=2)

        logger.info(
            f"Cached NCBI gene '{result['symbol']}' "
            f"({result['length']} bp, {result.get('organism', '?')})"
        )
        return insert

    except Exception as e:
        logger.warning(f"NCBI fallback failed for '{insert_id}': {e}")
        return None


def validate_dna_sequence(sequence: str) -> dict:
    """
    Validate a DNA sequence and return statistics.
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - length: int
        - gc_content: float (percentage)
        - invalid_characters: list or None
        - has_start_codon: bool
        - has_stop_codon: bool
    """
    # Remove whitespace and convert to uppercase
    clean_seq = re.sub(r'\s', '', sequence.upper())
    
    # Check for valid characters
    invalid_chars = set(clean_seq) - set('ATCGN')
    
    result = {
        "is_valid": len(invalid_chars) == 0,
        "length": len(clean_seq),
        "gc_content": None,
        "invalid_characters": list(invalid_chars) if invalid_chars else None,
        "has_start_codon": clean_seq[:3] == "ATG" if len(clean_seq) >= 3 else False,
        "has_stop_codon": clean_seq[-3:] in ["TAA", "TAG", "TGA"] if len(clean_seq) >= 3 else False,
    }
    
    if result["is_valid"] and len(clean_seq) > 0:
        gc_count = clean_seq.count('G') + clean_seq.count('C')
        result["gc_content"] = round(gc_count / len(clean_seq) * 100, 1)
    
    return result


def format_backbone_summary(backbone: dict) -> str:
    """Format a backbone entry as a readable summary."""
    lines = [
        f"## {backbone['name']}",
        f"**ID:** {backbone['id']}",
        f"**Size:** {backbone['size_bp']} bp",
        f"**Source:** {backbone.get('source', 'Unknown')}",
        f"**Organism:** {backbone.get('organism', 'Unknown')}",
        f"**Promoter:** {backbone.get('promoter', 'Unknown')}",
        f"**Bacterial Resistance:** {backbone.get('bacterial_resistance', 'Unknown')}",
    ]
    
    if backbone.get('mammalian_selection'):
        lines.append(f"**Mammalian Selection:** {backbone['mammalian_selection']}")
    
    if backbone.get('description'):
        lines.append(f"\n**Description:** {backbone['description']}")
    
    if backbone.get('mcs_position'):
        mcs = backbone['mcs_position']
        lines.append(f"\n**MCS Position:** {mcs['start']}-{mcs['end']} ({mcs.get('description', '')})")
    
    if backbone.get('addgene_id'):
        lines.append(f"\n**Addgene ID:** {backbone['addgene_id']}")
    
    return "\n".join(lines)


def format_insert_summary(insert: dict) -> str:
    """Format an insert entry as a readable summary."""
    lines = [
        f"## {insert['name']}",
        f"**ID:** {insert['id']}",
        f"**Size:** {insert['size_bp']} bp",
        f"**Category:** {insert.get('category', 'Unknown')}",
    ]
    
    if insert.get('protein_size_aa'):
        lines.append(f"**Protein Size:** {insert['protein_size_aa']} aa")
    
    if insert.get('excitation_nm') and insert.get('emission_nm'):
        lines.append(f"**Excitation/Emission:** {insert['excitation_nm']}/{insert['emission_nm']} nm")
    
    if insert.get('description'):
        lines.append(f"\n**Description:** {insert['description']}")
    
    if insert.get('genbank_accession'):
        lines.append(f"\n**GenBank Accession:** {insert['genbank_accession']}")
    
    return "\n".join(lines)


def get_all_backbones() -> list[dict]:
    """Get all backbones in the library."""
    return load_backbones()["backbones"]


def get_all_inserts() -> list[dict]:
    """Get all inserts in the library."""
    return load_inserts()["inserts"]


def design_construct(backbone_id: str, insert_id: str) -> dict:
    """
    Design a simple expression construct by combining backbone and insert.
    
    Args:
        backbone_id: Backbone identifier
        insert_id: Insert identifier
    
    Returns:
        Dictionary with construct information:
        - backbone: backbone details
        - insert: insert details
        - estimated_size: total construct size
        - insertion_site: MCS information
        - validation: sequence validation for insert
    """
    backbone = get_backbone_by_id(backbone_id)
    if not backbone:
        return {"error": f"Backbone '{backbone_id}' not found"}
    
    insert = get_insert_by_id(insert_id)
    if not insert:
        return {"error": f"Insert '{insert_id}' not found"}
    
    # Validate insert sequence if available
    validation = None
    if insert.get("sequence"):
        validation = validate_dna_sequence(insert["sequence"])
    
    # Calculate estimated size
    estimated_size = backbone["size_bp"] + insert["size_bp"]
    
    return {
        "backbone": {
            "id": backbone["id"],
            "name": backbone["name"],
            "size_bp": backbone["size_bp"],
            "promoter": backbone.get("promoter"),
            "organism": backbone.get("organism"),
        },
        "insert": {
            "id": insert["id"],
            "name": insert["name"],
            "size_bp": insert["size_bp"],
            "category": insert.get("category"),
            "has_sequence": insert.get("sequence") is not None,
        },
        "estimated_size": estimated_size,
        "insertion_site": backbone.get("mcs_position"),
        "insert_validation": validation,
    }


def search_all_sources(
    query: str,
    organism: Optional[str] = None,
    timeout: float = 20.0,
) -> dict:
    """Search local library, NCBI, and Addgene concurrently for a gene or plasmid.

    Runs all available searches in parallel using a thread pool, returns
    combined results within the timeout. Inspired by the concurrent lookup
    pattern in the metadata-capture project.

    Args:
        query: Gene symbol, plasmid name, or search term.
        organism: Optional organism filter for NCBI (e.g., "human", "mouse").
        timeout: Max seconds to wait for all sources (default 20s).

    Returns:
        Dict with keys:
        - local_inserts: list of matching inserts from the local library
        - local_backbones: list of matching backbones from the local library
        - ncbi_genes: list of NCBI Gene matches (if available)
        - addgene_plasmids: list of Addgene matches (if available)
        - sources_searched: list of source names that were queried
        - errors: dict of source -> error message for any failures
    """
    results = {
        "local_inserts": [],
        "local_backbones": [],
        "ncbi_genes": [],
        "addgene_plasmids": [],
        "sources_searched": [],
        "errors": {},
    }

    def _search_local_inserts():
        return search_inserts(query)

    def _search_local_backbones():
        return search_backbones(query, organism)

    def _search_ncbi():
        if not NCBI_AVAILABLE:
            return None
        return _ncbi_search_gene(query, organism)

    def _search_addgene():
        if not ADDGENE_AVAILABLE:
            return None
        client = AddgeneClient()
        return client.search(query, limit=5)

    # Map task names to callables
    tasks = {
        "local_inserts": _search_local_inserts,
        "local_backbones": _search_local_backbones,
        "ncbi_genes": _search_ncbi,
        "addgene_plasmids": _search_addgene,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}

        try:
            for future in as_completed(futures, timeout=timeout):
                name = futures[future]
                results["sources_searched"].append(name)
                try:
                    data = future.result()
                    if data is not None:
                        results[name] = data
                except Exception as e:
                    results["errors"][name] = str(e)
                    logger.warning(f"Concurrent search error ({name}): {e}")
        except TimeoutError:
            # as_completed(timeout=...) raises if any future is still pending.
            # Record which sources didn't finish; return partial results.
            for future, name in futures.items():
                if not future.done():
                    future.cancel()
                    results["errors"][name] = f"timed out after {timeout}s"
                    logger.warning(f"Concurrent search ({name}) timed out after {timeout}s")


def _rc(seq: str) -> str:
    comp = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(comp)[::-1]


def _extract_circular_region(seq: str, start: int, end: int) -> str:
    """Extract a region from a circular sequence, handling origin-spanning coords.

    If start < end, returns seq[start:end] (normal case).
    If start >= end, the feature wraps around position 0: returns seq[start:] + seq[:end].
    """
    if start < end:
        return seq[start:end]
    return seq[start:] + seq[:end]

def annotate_plasmid(plasmid_sequence: str) -> list[dict]:
    """Run pLannotate on a plasmid and return its feature annotations.

    Returns a list of feature dicts sorted by start position, each with:
      - name        : feature name (str)
      - type        : feature type, e.g. "CDS", "promoter", "rep_origin" (str)
      - start       : 0-based start position (int)
      - end         : 0-based end position, exclusive (int)
      - length      : feature length in bp (int)
      - strand      : 1 (forward) or -1 (reverse) (int)
      - origin_spanning : True if the feature wraps around position 0 (bool)
      - pct_identity : % identity to the database reference (float, 0–100)
      - description : free-text description from the pLannotate database (str)

    Coordinates are 0-based, end is exclusive (Python slice convention).
    Origin-spanning features have start > end; the feature covers
    seq[start:] + seq[:end].

    Returns an empty list if pLannotate is not installed or finds nothing.
    """
    import re as _re
    import os, sys
    plasmid_sequence = _re.sub(r'\s', '', plasmid_sequence.upper())

    try:
        from plannotate.annotate import annotate
    except ImportError:
        logger.error("pLannotate is not installed. Run: conda install -c bioconda plannotate")
        return []

    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        df = annotate(plasmid_sequence, linear=False)
    except Exception as e:
        logger.error(f"annotate_plasmid: pLannotate failed: {e}")
        return []

    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        custom_df = query_custom_db(plasmid_sequence)
        if custom_df is not None:
            df = merge_annotation_results(df, custom_df)

    if df is None or df.empty:
        return []

    features = []
    for _, row in df.iterrows():
        qstart_0 = int(row["qstart"])   # 0-based
        qend_0   = int(row["qend"])     # 0-based inclusive
        strand   = int(row["sframe"])

        # pLannotate qend is already 0-based exclusive
        # Origin-spanning: after pLannotate's wrap-around adjustment, qstart > qend
        origin_spanning = qstart_0 > qend_0

        features.append({
            "name":           str(row.get("Feature", "")),
            "type":           str(row.get("Type", "")),
            "start":          qstart_0,
            "end":            qend_0,
            "length":         int(row.get("length", abs(qend_0 - qstart_0) + 1)),
            "strand":         strand,
            "origin_spanning": origin_spanning,
            "pct_identity":   round(float(row.get("pident", 0)), 1),
            "description":    str(row.get("Description", "")),
        })

    features.sort(key=lambda f: f["start"])
    return features


def swap_feature(
    plasmid_sequence: str,
    feature_name: str,
    replacement_sequence: str,
) -> dict:
    """Replace a named feature in a circular plasmid with a new sequence.

    Locates feature_name via pLannotate, removes its region from the plasmid,
    and splices in replacement_sequence at the same position.

    Orientation handling: replacement_sequence must be provided in coding
    (5'→3' functional) orientation. If the feature is on the reverse strand,
    this function reverse-complements replacement_sequence before insertion so
    it is stored correctly in the plasmid.

    For multiple sequential swaps, pass the 'sequence' from this function's
    return dict as the plasmid_sequence for the next call — coordinates shift
    automatically because pLannotate re-annotates each time.

    Origin-spanning features (feature wraps around position 0) are not supported;
    returns an error dict in that case.

    Args:
        plasmid_sequence:    Full circular plasmid DNA sequence.
        feature_name:        Name of the feature to replace (case-insensitive,
                             matched via pLannotate annotation).
        replacement_sequence: New sequence in coding orientation (5'→3').

    Returns dict with keys:
        sequence         : updated plasmid DNA string
        replaced_feature : exact name matched by pLannotate
        replaced_coords  : (start, end) 0-based, end exclusive
        replaced_strand  : 1 or -1
        replaced_length  : bp removed
        inserted_length  : bp inserted
        size_delta       : inserted_length - replaced_length
        new_size         : total length of updated plasmid
    Or on failure:
        error            : human-readable error string
    """
    import re as _re
    import os, sys

    plasmid_sequence = _re.sub(r'\s', '', plasmid_sequence.upper())
    replacement_sequence = _re.sub(r'\s', '', replacement_sequence.upper())

    try:
        from plannotate.annotate import annotate
    except ImportError:
        return {"error": "pLannotate is not installed. Run: conda install -c bioconda plannotate"}

    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        df = annotate(plasmid_sequence, linear=False)
    except Exception as e:
        return {"error": f"pLannotate annotation failed: {e}"}

    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        try:
            custom_df = query_custom_db(plasmid_sequence)
            if custom_df is not None:
                df = merge_annotation_results(df, custom_df)
        except Exception:
            pass

    if df is None or df.empty:
        return {"error": "pLannotate found no features in the plasmid sequence."}

    name_lower = feature_name.lower()
    exact = df[df["Feature"].str.lower() == name_lower]
    match = exact if not exact.empty else df[df["Feature"].str.lower().str.contains(name_lower, na=False)]

    if match.empty:
        available = df["Feature"].tolist()
        return {"error": f"No feature matching '{feature_name}' found. Available: {available}"}

    row = match.iloc[0]
    qstart_0 = int(row["qstart"])   # 0-based
    qend_0   = int(row["qend"])     # 0-based inclusive
    strand   = int(row["sframe"])

    if qstart_0 > qend_0:
        return {
            "error": (
                f"Feature '{feature_name}' is origin-spanning (spans position 0). "
                "swap_feature does not support origin-spanning features. "
                "Use extract_insert_from_plasmid with explicit coordinates instead."
            )
        }

    # pLannotate qend is already 0-based exclusive
    slice_start = qstart_0
    slice_end   = qend_0

    # Apply orientation: if feature is on reverse strand, RC the replacement
    insert_seq = replacement_sequence
    if strand == -1:
        insert_seq = _rc(replacement_sequence)

    new_plasmid = plasmid_sequence[:slice_start] + insert_seq + plasmid_sequence[slice_end:]

    return {
        "sequence":         new_plasmid,
        "replaced_feature": str(row["Feature"]),
        "replaced_coords":  (qstart_0, qend_0),   # 0-based, end exclusive
        "replaced_strand":  strand,
        "replaced_length":  slice_end - slice_start,
        "inserted_length":  len(insert_seq),
        "size_delta":       len(insert_seq) - (slice_end - slice_start),
        "new_size":         len(new_plasmid),
    }


def _extract_insert_coordinates(df, insert_name):
    """Return (qstart, qend, strand) for the best match of insert_name in df.

    pLannotate qend is already 0-based exclusive (Python slice convention).
    Raises ValueError if no match found.
    """
    exact = df[df["Feature"].str.lower() == insert_name]
    match = exact if not exact.empty else df[df["Feature"].str.lower().str.contains(insert_name, na=False)]

    if match.empty:
        available = df["Feature"].tolist()
        raise ValueError(
            f"No feature matching '{insert_name}' found in pLannotate annotations. "
            f"Available: {available}"
        )

    row = match.iloc[0]
    qstart = int(row["qstart"])
    qend = int(row["qend"])  # already 0-based exclusive
    strand = int(row["sframe"])
    return qstart, qend, strand

def extract_inserts_from_plasmid(plasmid_sequence: str, insert_names: list) -> list:
    """Extract multiple named CDS inserts from a plasmid using pLannotate.

    Runs a single pLannotate annotation pass and extracts all requested features,
    handling origin-spanning and reverse-strand features automatically.

    Args:
        plasmid_sequence: Full plasmid DNA sequence string.
        insert_names: List of gene/feature names to extract (case-insensitive).

    Returns:
        List of insert dicts (id, name, sequence, size_bp, source) for each
        successfully matched feature. Features not found are omitted.
    """
    plasmid_sequence = re.sub(r'\s', '', plasmid_sequence.upper())

    invalid_chars = set(plasmid_sequence) - set("ACGTN")
    if invalid_chars or len(plasmid_sequence) < 10:
        raise ValueError(
            f"plasmid_sequence does not look like a DNA sequence "
            f"(length={len(plasmid_sequence)}, unexpected chars: {sorted(invalid_chars)}). "
            f"If you passed a cache key, resolve it to a sequence first."
        )

    try:
        from plannotate.annotate import annotate
    except ImportError:
        logger.error(
            "pLannotate is not installed. Cannot annotate plasmid to extract inserts. "
            "Run: conda install -c bioconda plannotate"
        )
        return []

 
    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        df = annotate(plasmid_sequence, linear=False)
    except Exception as e:
        logger.error(f"extract_inserts_from_plasmid: pLannotate annotation failed: {e}")
        raise RuntimeError(f"pLannotate failed to annotate the plasmid sequence: {e}") from e

    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        custom_df = query_custom_db(plasmid_sequence)
        if custom_df is not None:
            df = merge_annotation_results(df, custom_df)

    if df.empty:
        logger.info("extract_inserts_from_plasmid: pLannotate found no features in plasmid")
        return []

    # Use first and last gene in the ordered list to bound the region
    first_insert = insert_names[0].lower()
    last_insert = insert_names[-1].lower()

    start_1, end_1, strand_1 = _extract_insert_coordinates(df, first_insert)
    start_2, end_2, strand_2 = _extract_insert_coordinates(df, last_insert)

    # Determine strand from first gene (both genes should agree)
    is_reverse = strand_1 < 0

    # Select region boundaries based on strand:
    # Forward: region_start = first_gene.qstart (5′ end), region_end = last_gene.qend (3′ end)
    # Reverse: region_start = last_gene.qstart (3′ on forward = 5′ on reverse),
    #          region_end = first_gene.qend (5′ on forward = 3′ on reverse)
    # Origin-spanning is auto-detected: region_start > region_end
    if is_reverse:
        region_start = start_2  # last gene's qstart
        region_end = end_1      # first gene's qend
    else:
        region_start = start_1  # first gene's qstart
        region_end = end_2      # last gene's qend

    seq = _extract_circular_region(plasmid_sequence, region_start, region_end)
    if is_reverse:
        seq = _rc(seq)

    logger.debug(
        f"extract_inserts_from_plasmid: spanning region [{region_start}:{region_end}] "
        f"covering {insert_names}, strand={strand_1}, size={len(seq)} bp"
    )

    return {
        "id": "_".join(insert_names),
        "name": ", ".join(insert_names),
        "sequence": seq,
        "size_bp": len(seq),
        "source": "extracted_from_plasmid",
    }


def extract_insert_from_plasmid(
    plasmid_sequence: str,
    insert_name: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    strand: int = 1,
) -> Optional[dict]:
    """Extract an insert CDS from a plasmid sequence.

    Handles two cases automatically:
    - Origin-spanning features: start >= end means the feature wraps around
      position 0 (e.g. start=950, end=100 in a 1000 bp plasmid extracts
      seq[950:] + seq[:100]).
    - Reverse-complement features: strand=-1 reverse-complements the extracted
      region so the returned sequence is in 5'→3' coding orientation.

    If start and end are provided, slices directly at those coordinates.
    Otherwise, runs pLannotate to annotate the plasmid and finds the feature
    whose name best matches insert_name (strand and wraparound are inferred
    automatically from the annotation).

    Args:
        plasmid_sequence: Full plasmid DNA sequence string.
        insert_name: Name of the gene/feature to extract (case-insensitive).
        start: 0-based start coordinate (optional, skips annotation if given).
        end: 0-based end coordinate, exclusive (optional).
              If start >= end, the region is treated as origin-spanning.
        strand: 1 (default, forward) or -1 (reverse complement the result).

    Returns:
        Insert dict with keys id, name, sequence, size_bp, source — or None if
        not found.
    """
    plasmid_sequence = re.sub(r'\s', '', plasmid_sequence.upper())

    # Guard: reject anything that isn't a DNA sequence (e.g. a cache key string)
    dna_is_valid, errors = validate_dna(plasmid_sequence)
    if not dna_is_valid:
        raise ValueError(
            f"plasmid_sequence does not look like a DNA sequence "
            f"(length={len(plasmid_sequence)}, unexpected chars: {sorted(e)}). "
            f"If you passed a cache key, resolve it to a sequence first."
        )

    # ── Explicit coordinates: slice directly ──────────────────────────────
    # Coordinates are 0-based, end is exclusive (Python slice convention).
    # Origin-spanning: start >= end means the feature wraps around position 0.
    if start is not None and end is not None:
        seq = _extract_circular_region(plasmid_sequence, start, end)
        if not seq:
            logger.warning(f"extract_insert_from_plasmid: empty slice [{start}:{end}]")
            return None
        if strand == -1:
            seq = _rc(seq)
        return {
            "id": insert_name,
            "name": insert_name,
            "sequence": seq,
            "size_bp": len(seq),
            "source": "extracted_from_plasmid",
            "warning": (
                f"Explicit coordinates used ([{start}:{end}], {len(seq)} bp) — "
                "pLannotate boundary verification was bypassed. "
                "If this sequence is a replacement cassette for a parts swap, "
                "verify that these coordinates match pLannotate-annotated feature "
                "boundaries before proceeding. If they extend into unannotated regions "
                "(gaps, 5'UTRs, cloning junctions), stop and ask the user to confirm "
                "the intended boundaries."
            ),
        }

    # ── pLannotate annotation: find feature by name ───────────────────────
    try:
        from plannotate.annotate import annotate
    except ImportError:
        logger.error(
            "pLannotate is not installed. Cannot annotate plasmid to extract insert. "
            "Run: conda install -c bioconda plannotate"
        )
        return None

    # Ensure conda env bin dir is on PATH so pLannotate can find blastn/cmscan
    import os, sys
    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = conda_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        df = annotate(plasmid_sequence, linear=False)
    except Exception as e:
        logger.error(f"extract_insert_from_plasmid: pLannotate annotation failed: {e}")
        raise RuntimeError(f"pLannotate failed to annotate the plasmid sequence: {e}") from e

    if _CUSTOM_ANNOTATIONS_AVAILABLE:
        custom_df = query_custom_db(plasmid_sequence)
        if custom_df is not None:
            df = merge_annotation_results(df, custom_df)

    if df.empty:
        logger.info(f"extract_insert_from_plasmid: pLannotate found no features in plasmid")
        return None

    # Case-insensitive match: prefer exact, then partial
    name_lower = insert_name.lower()
    exact = df[df["Feature"].str.lower() == name_lower]
    match = exact if not exact.empty else df[df["Feature"].str.lower().str.contains(name_lower, na=False)]

    if match.empty:
        logger.info(
            f"extract_insert_from_plasmid: no feature matching '{insert_name}' found. "
            f"Available: {df['Feature'].tolist()}"
        )
        return None

    # Use the highest-scoring (first) match
    row = match.iloc[0]

    # Re-extract from the plasmid using qstart/qend.
    # pLannotate qend is already 0-based exclusive; use directly as slice end.
    seq = _extract_circular_region(plasmid_sequence, int(row["qstart"]), int(row["qend"]))

    # sframe < 0 means the feature is on the reverse strand — RC to coding orientation.
    if int(row["sframe"]) < 0:
        seq = _rc(seq)
        logger.debug(f"extract_insert_from_plasmid: reverse-complemented '{insert_name}' (sframe={row['sframe']})")

    logger.debug(
        f"extract_insert_from_plasmid: extracted '{insert_name}' "
        f"qstart={row['qstart']} qend={row['qend']} sframe={row['sframe']} "
        f"origin_spanning={int(row['qstart']) >= int(row['qend'])} size={len(seq)} bp"
    )

    return {
        "id": insert_name,
        "name": str(row["Feature"]),
        "sequence": seq,
        "size_bp": len(seq),
        "source": "extracted_from_plasmid",
        "start": int(row["qstart"]),
        "end": int(row["qend"]),  # already 0-based exclusive
        "strand": int(row["sframe"]),
    }
