#!/usr/bin/env python3
"""
Plasmid Library Core Functions

Core functionality for the plasmid library, independent of MCP.
Can be used directly for testing or integration.
"""

import json
import re
from pathlib import Path
from typing import Optional

# Library path
LIBRARY_PATH = Path(__file__).parent.parent / "library"


def load_backbones() -> dict:
    """Load backbone library from JSON file."""
    with open(LIBRARY_PATH / "backbones.json", "r") as f:
        return json.load(f)


def load_inserts() -> dict:
    """Load insert library from JSON file."""
    with open(LIBRARY_PATH / "inserts.json", "r") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    """Normalize a plasmid/insert name for matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def search_backbones(query: str, organism: Optional[str] = None, promoter: Optional[str] = None) -> list[dict]:
    """
    Search for backbones matching the query.
    
    Args:
        query: Search term (name, feature, or keyword)
        organism: Filter by organism type (mammalian, bacterial, etc.)
        promoter: Filter by promoter type (CMV, T7, etc.)
    
    Returns:
        List of matching backbone dictionaries
    """
    data = load_backbones()
    results = []
    query_normalized = normalize_name(query)
    
    for backbone in data["backbones"]:
        # Check name and aliases
        names_to_check = [backbone["id"], backbone["name"]] + backbone.get("aliases", [])
        name_match = any(query_normalized in normalize_name(n) for n in names_to_check)
        
        # Check description
        desc_match = query.lower() in backbone.get("description", "").lower()
        
        if name_match or desc_match:
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
    return None


def get_insert_by_id(insert_id: str) -> Optional[dict]:
    """
    Get a specific insert by ID or alias.
    
    Args:
        insert_id: Insert identifier or alias
    
    Returns:
        Insert dictionary or None if not found
    """
    data = load_inserts()
    id_normalized = normalize_name(insert_id)
    
    for insert in data["inserts"]:
        names_to_check = [insert["id"]] + insert.get("aliases", [])
        if any(normalize_name(n) == id_normalized for n in names_to_check):
            return insert
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
