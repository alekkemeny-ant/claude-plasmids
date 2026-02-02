#!/usr/bin/env python3
"""
Test script for the plasmid library functionality.
Tests the core library functions without requiring MCP.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from library import (
    load_backbones,
    load_inserts,
    search_backbones,
    search_inserts,
    get_backbone_by_id,
    get_insert_by_id,
    validate_dna_sequence,
    format_backbone_summary,
    format_insert_summary,
    design_construct,
)


def test_load_data():
    """Test that library data loads correctly."""
    print("Testing data loading...")
    
    backbones = load_backbones()
    assert "backbones" in backbones
    assert len(backbones["backbones"]) > 0
    print(f"  ✓ Loaded {len(backbones['backbones'])} backbones")
    
    inserts = load_inserts()
    assert "inserts" in inserts
    assert len(inserts["inserts"]) > 0
    print(f"  ✓ Loaded {len(inserts['inserts'])} inserts")


def test_search_backbones():
    """Test backbone search functionality."""
    print("\nTesting backbone search...")
    
    # Search by name
    results = search_backbones("pcDNA3.1")
    assert len(results) >= 2  # Should find pcDNA3.1(+) and pcDNA3.1(-)
    print(f"  ✓ Search 'pcDNA3.1' found {len(results)} results")
    
    # Search with organism filter
    results = search_backbones("CMV", organism="mammalian")
    assert len(results) > 0
    print(f"  ✓ Search 'CMV' (mammalian) found {len(results)} results")
    
    # Search for bacterial vectors
    results = search_backbones("pET", organism="bacterial")
    assert len(results) > 0
    print(f"  ✓ Search 'pET' (bacterial) found {len(results)} results")


def test_search_inserts():
    """Test insert search functionality."""
    print("\nTesting insert search...")
    
    # Search for fluorescent proteins
    results = search_inserts("GFP")
    assert len(results) > 0
    print(f"  ✓ Search 'GFP' found {len(results)} results")
    
    # Search with category filter
    results = search_inserts("tag", category="epitope_tag")
    assert len(results) > 0
    print(f"  ✓ Search 'tag' (epitope_tag) found {len(results)} results")


def test_get_by_id():
    """Test retrieval by ID."""
    print("\nTesting ID lookup...")
    
    # Get backbone by exact ID
    bb = get_backbone_by_id("pcDNA3.1(+)")
    assert bb is not None
    assert bb["size_bp"] == 5428
    print(f"  ✓ Found pcDNA3.1(+): {bb['size_bp']} bp")
    
    # Get backbone by alias
    bb = get_backbone_by_id("PX459")
    assert bb is not None
    assert "Cas9" in bb["name"]
    print(f"  ✓ Found PX459 (alias): {bb['name']}")
    
    # Get insert by ID
    ins = get_insert_by_id("EGFP")
    assert ins is not None
    assert ins["size_bp"] == 720
    assert "sequence" in ins
    print(f"  ✓ Found EGFP: {ins['size_bp']} bp")


def test_validate_sequence():
    """Test DNA sequence validation."""
    print("\nTesting sequence validation...")
    
    # Valid sequence with start and stop codon
    result = validate_dna_sequence("ATGGCTAGCTAA")
    assert result["is_valid"] == True
    assert result["length"] == 12
    assert result["has_start_codon"] == True
    assert result["has_stop_codon"] == True
    print(f"  ✓ Valid sequence: length={result['length']}, GC={result['gc_content']}%")
    
    # Invalid sequence
    result = validate_dna_sequence("ATGXYZ")
    assert result["is_valid"] == False
    assert "X" in result["invalid_characters"]
    print(f"  ✓ Invalid sequence detected: {result['invalid_characters']}")
    
    # Test EGFP sequence
    egfp = get_insert_by_id("EGFP")
    result = validate_dna_sequence(egfp["sequence"])
    assert result["is_valid"] == True
    assert result["length"] == 720
    assert result["has_start_codon"] == True
    print(f"  ✓ EGFP sequence valid: {result['length']} bp, GC={result['gc_content']}%")


def test_format_output():
    """Test formatting functions."""
    print("\nTesting output formatting...")
    
    bb = get_backbone_by_id("pcDNA3.1(+)")
    summary = format_backbone_summary(bb)
    assert "pcDNA3.1(+)" in summary
    assert "5428" in summary
    assert "CMV" in summary
    print("  ✓ Backbone summary formatted correctly")
    
    ins = get_insert_by_id("EGFP")
    summary = format_insert_summary(ins)
    assert "EGFP" in summary
    assert "720" in summary
    print("  ✓ Insert summary formatted correctly")


def test_mcs_info():
    """Test MCS information retrieval."""
    print("\nTesting MCS information...")
    
    bb = get_backbone_by_id("pcDNA3.1(+)")
    mcs = bb.get("mcs_position")
    assert mcs is not None
    assert mcs["start"] == 895
    assert mcs["end"] == 1010
    print(f"  ✓ pcDNA3.1(+) MCS: {mcs['start']}-{mcs['end']}")
    
    bb = get_backbone_by_id("pET-28a(+)")
    mcs = bb.get("mcs_position")
    assert mcs is not None
    print(f"  ✓ pET-28a(+) MCS: {mcs['start']}-{mcs['end']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PLASMID LIBRARY TEST SUITE")
    print("=" * 60)
    
    try:
        test_load_data()
        test_search_backbones()
        test_search_inserts()
        test_get_by_id()
        test_validate_sequence()
        test_format_output()
        test_mcs_info()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
