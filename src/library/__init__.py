#!/usr/bin/env python3
"""
Plasmid library package — public facade.

Groups the three library concerns into one package:
    core.py     built-in library loading, search, get-by-id, validation,
                formatting, annotation, and plasmid feature operations
    user.py     Bring-Your-Own-Library (BYOL) — GenBank files from
                $PLASMID_USER_LIBRARY, namespaced with a `user:` ID prefix
    vendor.py   vendor-supplied backbones (Ansa, Twist, ...) persisted to
                library/vendor_backbones.json with a `vendor:` ID prefix

core.py merges all three sources in load_backbones()/load_inserts(). This
module re-exports the public API so callers keep importing from
`src.library` unchanged (e.g. `from src.library import load_backbones`).

Note: `src/library/` is CODE; the top-level `library/` directory is DATA
(backbones.json, inserts.json, vendor_backbones.json).
"""

from src.library.core import (
    KNOWN_PROMOTERS,
    annotate_plasmid,
    check_gene_family_ambiguity,
    clear_test_fixtures,
    design_construct,
    extract_insert_from_plasmid,
    extract_inserts_from_plasmid,
    find_duplicate_annotations,
    format_backbone_summary,
    format_insert_summary,
    get_all_backbones,
    get_all_inserts,
    get_backbone_by_id,
    get_insert_by_id,
    infer_species_from_cell_line,
    is_known_promoter,
    load_backbones,
    load_inserts,
    register_test_fixtures,
    search_all_sources,
    search_backbones,
    search_inserts,
    set_library_readonly,
    swap_feature,
    validate_dna_sequence,
)
from src.library.user import (
    GENBANK_EXTENSIONS,
    USER_PREFIX,
    load_user_backbones,
    load_user_designed_constructs,
    load_user_inserts,
)
from src.library.vendor import (
    get_vendor_backbone_by_id,
    load_vendor_backbones,
    save_vendor_backbone,
    update_vendor_backbone_mcs,
)

__all__ = [
    # core
    "KNOWN_PROMOTERS",
    "annotate_plasmid",
    "check_gene_family_ambiguity",
    "clear_test_fixtures",
    "design_construct",
    "extract_insert_from_plasmid",
    "extract_inserts_from_plasmid",
    "find_duplicate_annotations",
    "format_backbone_summary",
    "format_insert_summary",
    "get_all_backbones",
    "get_all_inserts",
    "get_backbone_by_id",
    "get_insert_by_id",
    "infer_species_from_cell_line",
    "is_known_promoter",
    "load_backbones",
    "load_inserts",
    "register_test_fixtures",
    "search_all_sources",
    "search_backbones",
    "search_inserts",
    "set_library_readonly",
    "swap_feature",
    "validate_dna_sequence",
    # user (BYOL)
    "GENBANK_EXTENSIONS",
    "USER_PREFIX",
    "load_user_backbones",
    "load_user_designed_constructs",
    "load_user_inserts",
    # vendor
    "get_vendor_backbone_by_id",
    "load_vendor_backbones",
    "save_vendor_backbone",
    "update_vendor_backbone_mcs",
]
