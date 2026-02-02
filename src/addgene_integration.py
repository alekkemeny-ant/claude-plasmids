#!/usr/bin/env python3
"""
Addgene Integration Module

This module provides integration with Addgene's plasmid repository.
It supports two modes:
1. Web scraping (default, no auth required) - for immediate use
2. Official API (requires credentials) - for production use

The module provides:
- Search for plasmids by name, ID, or features
- Fetch plasmid metadata and sequences
- Import plasmids to the local library
"""

import json
import re
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, quote
import logging

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


@dataclass
class AddgenePlasmid:
    """Represents an Addgene plasmid."""
    addgene_id: str
    name: str = ""
    description: Optional[str] = None
    size_bp: Optional[int] = None
    backbone: Optional[str] = None
    promoter: Optional[str] = None
    bacterial_resistance: Optional[str] = None
    mammalian_selection: Optional[str] = None
    vector_type: Optional[str] = None
    species: Optional[str] = None
    gene_insert: Optional[str] = None
    tags: Optional[List[str]] = None
    depositor: Optional[str] = None
    article_title: Optional[str] = None
    pubmed_id: Optional[str] = None
    sequence: Optional[str] = None
    sequence_source: Optional[str] = None  # 'depositor', 'addgene_full', 'addgene_partial'
    genbank_file_url: Optional[str] = None
    snapgene_file_url: Optional[str] = None
    url: Optional[str] = None
    parsed_features: Optional[List[Dict[str, Any]]] = None
    mcs_position: Optional[Dict[str, Any]] = None

    def to_backbone_dict(self) -> Dict[str, Any]:
        """Convert to backbone library format."""
        return {
            "id": self.name or f"Addgene_{self.addgene_id}",
            "aliases": [f"Addgene #{self.addgene_id}", f"addgene:{self.addgene_id}"],
            "name": self.name or f"Addgene #{self.addgene_id}",
            "description": self.description,
            "size_bp": self.size_bp,
            "source": f"Addgene (depositor: {self.depositor})" if self.depositor else "Addgene",
            "organism": self._infer_organism(),
            "promoter": self.promoter,
            "bacterial_resistance": self.bacterial_resistance,
            "mammalian_selection": self.mammalian_selection,
            "origin": None,
            "copy_number": None,
            "mcs_position": self.mcs_position,
            "features": self.parsed_features or [],
            "genbank_accession": None,
            "addgene_id": self.addgene_id,
            "sequence": self.sequence,
            "sequence_source": self.sequence_source,
        }

    def _infer_organism(self) -> str:
        """Infer organism type from vector characteristics."""
        vector_type = (self.vector_type or "").lower()
        if "mammalian" in vector_type:
            return "mammalian"
        elif "bacterial" in vector_type or "e. coli" in vector_type:
            return "bacterial"
        elif "lentiv" in vector_type:
            return "mammalian"
        elif "yeast" in vector_type:
            return "yeast"
        elif "insect" in vector_type:
            return "insect"
        return "unknown"


class AddgeneClient:
    """Client for interacting with Addgene."""
    
    BASE_URL = "https://www.addgene.org"
    API_BASE_URL = "https://api.addgene.org"  # Official API (requires auth)
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Addgene client.

        Args:
            api_token: Optional API token for official API access.
                      If not provided, uses web scraping.
        """
        self.api_token = api_token or os.environ.get("ADDGENE_API_TOKEN")
        self.use_official_api = bool(self.api_token)

        # Use a persistent session so cookies (e.g., from visiting a plasmid
        # page) carry over to subsequent requests (e.g., GenBank file download).
        if HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            })
        else:
            self._session = None

        if self.use_official_api:
            logger.info("Using official Addgene API")
        else:
            logger.info("Using web scraping (no API token provided)")
    
    def _make_request(self, url: str, headers: Optional[Dict] = None) -> str:
        """Make an HTTP GET request using the persistent session."""
        headers = headers or {}

        if HAS_REQUESTS and self._session is not None:
            response = self._session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        else:
            headers.setdefault("User-Agent", "PlasmidLibrary/1.0")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode('utf-8')
    
    def _fetch_json(self, url: str) -> Dict:
        """Fetch JSON from URL."""
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        text = self._make_request(url, headers)
        return json.loads(text)
    
    def get_plasmid(self, addgene_id: str) -> Optional[AddgenePlasmid]:
        """
        Fetch a plasmid by Addgene ID.
        
        Args:
            addgene_id: The Addgene catalog number (e.g., "50005", "12260")
        
        Returns:
            AddgenePlasmid object or None if not found
        """
        # Clean the ID
        addgene_id = re.sub(r'[^\d]', '', str(addgene_id))
        
        if self.use_official_api:
            return self._get_plasmid_api(addgene_id)
        else:
            return self._get_plasmid_scrape(addgene_id)
    
    def _get_plasmid_api(self, addgene_id: str) -> Optional[AddgenePlasmid]:
        """Fetch plasmid using official API."""
        try:
            url = f"{self.API_BASE_URL}/v1/plasmids/{addgene_id}"
            data = self._fetch_json(url)
            return self._parse_api_response(data)
        except Exception as e:
            logger.error(f"API error fetching plasmid {addgene_id}: {e}")
            return None
    
    def _get_plasmid_scrape(self, addgene_id: str) -> Optional[AddgenePlasmid]:
        """Fetch plasmid by scraping the web page."""
        try:
            url = f"{self.BASE_URL}/{addgene_id}/"
            html = self._make_request(url)
            return self._parse_plasmid_page(addgene_id, html)
        except Exception as e:
            logger.error(f"Error scraping plasmid {addgene_id}: {e}")
            return None
    
    def _parse_plasmid_page(self, addgene_id: str, html: str) -> Optional[AddgenePlasmid]:
        """Parse plasmid information from HTML page."""
        plasmid = AddgenePlasmid(addgene_id=addgene_id)
        plasmid.url = f"{self.BASE_URL}/{addgene_id}/"
        
        # Extract name from title
        title_match = re.search(r'<title>Addgene:\s*([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            plasmid.name = title_match.group(1).strip()
        
        # Extract description
        desc_match = re.search(r'<meta name="description" content="([^"]+)"', html)
        if desc_match:
            plasmid.description = desc_match.group(1).strip()
        
        # Try to find size
        size_match = re.search(r'(\d{3,6})\s*bp', html)
        if size_match:
            plasmid.size_bp = int(size_match.group(1))
        
        # Look for resistance markers
        if re.search(r'ampicillin|amp\s*resistance', html, re.IGNORECASE):
            plasmid.bacterial_resistance = "Ampicillin"
        elif re.search(r'kanamycin|kan\s*resistance', html, re.IGNORECASE):
            plasmid.bacterial_resistance = "Kanamycin"
        
        # Look for mammalian selection
        if re.search(r'puromycin|puro\s*resistance', html, re.IGNORECASE):
            plasmid.mammalian_selection = "Puromycin"
        elif re.search(r'neomycin|neo\s*resistance|g418|geneticin', html, re.IGNORECASE):
            plasmid.mammalian_selection = "Neomycin/G418"
        
        # Look for promoter
        promoter_match = re.search(r'promoter[:\s]+(\w+)', html, re.IGNORECASE)
        if promoter_match:
            plasmid.promoter = promoter_match.group(1)
        
        # GenBank file URL
        gb_match = re.search(r'href="([^"]+\.gb[^"]*)"', html)
        if gb_match:
            plasmid.genbank_file_url = urljoin(self.BASE_URL, gb_match.group(1))
        
        # SnapGene file URL
        snap_match = re.search(r'href="([^"]+\.dna[^"]*)"', html)
        if snap_match:
            plasmid.snapgene_file_url = urljoin(self.BASE_URL, snap_match.group(1))
        
        return plasmid
    
    def _parse_api_response(self, data: Dict) -> AddgenePlasmid:
        """Parse official API response into AddgenePlasmid."""
        return AddgenePlasmid(
            addgene_id=str(data.get("id", "")),
            name=data.get("name"),
            description=data.get("description"),
            size_bp=data.get("size"),
            backbone=data.get("backbone_name"),
            promoter=data.get("promoter"),
            bacterial_resistance=data.get("bacterial_resistance"),
            mammalian_selection=data.get("selectable_markers"),
            vector_type=data.get("vector_type"),
            species=data.get("species"),
            gene_insert=data.get("gene_insert"),
            depositor=data.get("depositor_name"),
            article_title=data.get("article_title"),
            pubmed_id=data.get("pubmed_id"),
            sequence=data.get("sequence"),
            url=f"{self.BASE_URL}/{data.get('id')}/"
        )
    
    def get_sequence(self, addgene_id: str) -> Optional[str]:
        """
        Fetch the DNA sequence for a plasmid.

        Args:
            addgene_id: The Addgene catalog number

        Returns:
            DNA sequence string or None if not available
        """
        sequence, _, _ = self.get_genbank_data(addgene_id)
        return sequence
    
    def _extract_sequence_from_genbank(self, content: str) -> Optional[str]:
        """Extract DNA sequence from GenBank format content."""
        # Find ORIGIN section
        origin_match = re.search(r'ORIGIN\s*\n(.*?)(?://|\Z)', content, re.DOTALL)
        if not origin_match:
            return None

        origin_section = origin_match.group(1)

        # Remove numbers and whitespace, keep only nucleotides
        sequence = re.sub(r'[^atcgATCGnN]', '', origin_section)
        sequence = sequence.upper()

        if len(sequence) > 100:  # Sanity check
            return sequence

        return None

    @staticmethod
    def _parse_genbank_location(location_str: str) -> tuple[int, int]:
        """Parse a GenBank location string and return (start, end) as 0-based.

        Handles simple (100..200), complement(100..200),
        and join(100..200,201..300) formats. Returns the overall span.
        """
        # Strip complement() wrapper
        inner = location_str.strip()
        if inner.startswith("complement(") and inner.endswith(")"):
            inner = inner[len("complement("):-1]

        # Strip join() wrapper
        if inner.startswith("join(") and inner.endswith(")"):
            inner = inner[len("join("):-1]

        # Collect all coordinate pairs from possibly comma-separated ranges
        coords: list[int] = []
        for part in inner.split(","):
            part = part.strip()
            m = re.search(r'(\d+)\.\.(\d+)', part)
            if m:
                coords.extend([int(m.group(1)), int(m.group(2))])

        if not coords:
            # Single position like "100"
            m = re.search(r'(\d+)', location_str)
            if m:
                pos = int(m.group(1))
                return pos - 1, pos
            return 0, 0

        # Convert from 1-based GenBank to 0-based
        return min(coords) - 1, max(coords)

    def _parse_genbank_features(self, content: str) -> tuple[List[Dict], Optional[Dict]]:
        """Parse feature annotations from GenBank content.

        Returns:
            (features, mcs_position) where features is a list of dicts
            matching the backbone library format, and mcs_position is a dict
            with 'start', 'end', 'description' keys (or None).
        """
        # Types we care about for the backbone library
        RELEVANT_TYPES = {
            "promoter", "CDS", "polyA_signal", "rep_origin",
            "misc_feature", "regulatory", "enhancer",
        }

        features: List[Dict] = []
        mcs_position: Optional[Dict] = None

        # Find the FEATURES section
        feat_match = re.search(
            r'^FEATURES\s+Location/Qualifiers\s*\n(.*?)(?=^ORIGIN|\Z)',
            content, re.MULTILINE | re.DOTALL,
        )
        if not feat_match:
            return features, mcs_position

        feat_text = feat_match.group(1)

        # Split into individual feature blocks.
        # Each feature starts with 5 spaces + feature_key + spaces + location
        blocks = re.split(r'\n(?=     \S)', feat_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Parse the header line: "     feature_key    location"
            header_match = re.match(r'(\S+)\s+([\S]+)', block)
            if not header_match:
                continue
            feat_type = header_match.group(1)
            location_str = header_match.group(2)

            if feat_type not in RELEVANT_TYPES:
                continue

            start, end = self._parse_genbank_location(location_str)

            # Extract qualifiers
            label = ""
            note = ""
            gene = ""
            product = ""
            for qual_match in re.finditer(
                r'/(\w+)="((?:[^"\\]|"")*)"',
                block, re.DOTALL,
            ):
                key = qual_match.group(1)
                val = qual_match.group(2).replace("\n", " ").strip()
                # Collapse internal whitespace
                val = re.sub(r'\s+', ' ', val)
                if key == "label":
                    label = val
                elif key == "note":
                    note = val
                elif key == "gene":
                    gene = val
                elif key == "product":
                    product = val

            name = label or gene or product or feat_type

            # Detect MCS
            is_mcs = False
            if feat_type == "misc_feature":
                combined = (label + " " + note).lower()
                if "multiple cloning site" in combined or "mcs" in combined.split():
                    is_mcs = True

            if is_mcs:
                mcs_position = {
                    "start": start,
                    "end": end,
                    "description": note or f"Multiple cloning site ({label})",
                }
                # Also add as a feature
                features.append({
                    "name": name,
                    "type": "misc_feature",
                    "start": start,
                    "end": end,
                })
            else:
                features.append({
                    "name": name,
                    "type": feat_type,
                    "start": start,
                    "end": end,
                })

        return features, mcs_position

    def get_genbank_data(
        self, addgene_id: str,
    ) -> tuple[Optional[str], List[Dict], Optional[Dict]]:
        """Fetch GenBank file and extract sequence + features + MCS position.

        Returns:
            (sequence, features, mcs_position)
        """
        addgene_id = re.sub(r'[^\d]', '', str(addgene_id))

        genbank_content = self._fetch_genbank_content(addgene_id)
        if not genbank_content:
            return None, [], None

        sequence = self._extract_sequence_from_genbank(genbank_content)
        features, mcs_position = self._parse_genbank_features(genbank_content)
        return sequence, features, mcs_position

    def _fetch_genbank_content(self, addgene_id: str) -> Optional[str]:
        """Fetch the raw GenBank file content for a plasmid."""
        # Visit the main plasmid page first to establish session cookies
        try:
            self._make_request(f"{self.BASE_URL}/{addgene_id}/")
        except Exception:
            pass

        # Try the sequences page for GenBank links
        try:
            seq_url = f"{self.BASE_URL}/{addgene_id}/sequences/"
            html = self._make_request(seq_url)
            gb_links = re.findall(
                r'href="([^"]+(?:genbank|\.gb)[^"]*)"', html, re.IGNORECASE,
            )
            for link in gb_links:
                full_url = urljoin(self.BASE_URL, link)
                try:
                    content = self._make_request(full_url)
                    if "ORIGIN" in content:
                        return content
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Could not fetch sequences page for {addgene_id}: {e}")

        return None
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search Addgene for plasmids.
        
        Args:
            query: Search term (plasmid name, gene, etc.)
            limit: Maximum number of results
        
        Returns:
            List of search results with basic info
        """
        if self.use_official_api:
            return self._search_api(query, limit)
        else:
            return self._search_scrape(query, limit)
    
    def _search_api(self, query: str, limit: int) -> List[Dict]:
        """Search using official API."""
        try:
            url = f"{self.API_BASE_URL}/v1/plasmids/search?q={quote(query)}&limit={limit}"
            data = self._fetch_json(url)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"API search error: {e}")
            return []
    
    def _search_scrape(self, query: str, limit: int) -> List[Dict]:
        """Search by scraping the search page."""
        try:
            url = f"{self.BASE_URL}/search/catalog/plasmids/?q={quote(query)}"
            html = self._make_request(url)
            
            # Extract plasmid IDs and names from search results
            results = []
            pattern = r'href="/(\d+)/"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)
            
            for addgene_id, name in matches[:limit]:
                results.append({
                    "addgene_id": addgene_id,
                    "name": name.strip(),
                    "url": f"{self.BASE_URL}/{addgene_id}/"
                })
            
            return results
        except Exception as e:
            logger.error(f"Search scrape error: {e}")
            return []


class AddgeneLibraryIntegration:
    """Integration layer between Addgene and the local plasmid library."""
    
    def __init__(self, library_path: Path, api_token: Optional[str] = None):
        """
        Initialize the integration.
        
        Args:
            library_path: Path to the library directory containing backbones.json
            api_token: Optional Addgene API token
        """
        self.library_path = Path(library_path)
        self.backbones_file = self.library_path / "backbones.json"
        self.client = AddgeneClient(api_token)
    
    def _load_backbones(self) -> Dict:
        """Load the current backbones library."""
        with open(self.backbones_file, 'r') as f:
            return json.load(f)
    
    def _save_backbones(self, data: Dict):
        """Save the backbones library."""
        with open(self.backbones_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_plasmid(self, addgene_id: str, include_sequence: bool = True) -> Optional[Dict]:
        """
        Import a plasmid from Addgene into the local library.

        Args:
            addgene_id: Addgene catalog number
            include_sequence: Whether to fetch and include the sequence

        Returns:
            The imported backbone dict, or None if failed
        """
        # Fetch from Addgene
        plasmid = self.client.get_plasmid(addgene_id)
        if not plasmid:
            logger.error(f"Could not fetch plasmid {addgene_id} from Addgene")
            return None

        # Get sequence and features from GenBank file
        if include_sequence:
            sequence, features, mcs_position = self.client.get_genbank_data(addgene_id)
            if sequence:
                plasmid.sequence = sequence
                plasmid.sequence_source = "addgene"
            if features:
                plasmid.parsed_features = features
            if mcs_position:
                plasmid.mcs_position = mcs_position
        
        # Convert to backbone format
        backbone = plasmid.to_backbone_dict()
        
        # Load current library
        data = self._load_backbones()
        
        # Check if already exists
        existing_ids = {bb["id"] for bb in data["backbones"]}
        existing_addgene_ids = {bb.get("addgene_id") for bb in data["backbones"]}
        
        if addgene_id in existing_addgene_ids:
            # Update existing entry
            for i, bb in enumerate(data["backbones"]):
                if bb.get("addgene_id") == addgene_id:
                    data["backbones"][i] = backbone
                    logger.info(f"Updated existing entry for Addgene #{addgene_id}")
                    break
        else:
            # Add new entry
            data["backbones"].append(backbone)
            logger.info(f"Added new entry for Addgene #{addgene_id}")
        
        # Save
        self._save_backbones(data)
        
        return backbone
    
    def update_sequences_from_addgene(self) -> Dict[str, str]:
        """
        Update sequences for all backbones that have Addgene IDs but no sequences.
        
        Returns:
            Dict mapping backbone ID to status ('updated', 'failed', 'skipped')
        """
        data = self._load_backbones()
        results = {}
        
        for backbone in data["backbones"]:
            bb_id = backbone["id"]
            addgene_id = backbone.get("addgene_id")
            
            # Skip if no Addgene ID or already has sequence
            if not addgene_id:
                results[bb_id] = "no_addgene_id"
                continue
            
            if backbone.get("sequence"):
                results[bb_id] = "already_has_sequence"
                continue
            
            # Try to fetch sequence
            logger.info(f"Fetching sequence for {bb_id} (Addgene #{addgene_id})")
            sequence = self.client.get_sequence(addgene_id)
            
            if sequence:
                backbone["sequence"] = sequence
                backbone["sequence_source"] = "addgene"
                backbone["size_bp"] = len(sequence)
                results[bb_id] = "updated"
                logger.info(f"  ✓ Got {len(sequence)} bp sequence")
            else:
                results[bb_id] = "sequence_not_available"
                logger.warning(f"  ✗ Could not get sequence")
        
        # Save updates
        self._save_backbones(data)
        
        return results


# Convenience functions for use in MCP tools
def search_addgene(query: str, limit: int = 10) -> List[Dict]:
    """Search Addgene for plasmids."""
    client = AddgeneClient()
    return client.search(query, limit)


def get_addgene_plasmid(addgene_id: str) -> Optional[AddgenePlasmid]:
    """Fetch a plasmid from Addgene."""
    client = AddgeneClient()
    return client.get_plasmid(addgene_id)


def get_addgene_sequence(addgene_id: str) -> Optional[str]:
    """Fetch a plasmid sequence from Addgene."""
    client = AddgeneClient()
    return client.get_sequence(addgene_id)


def fetch_addgene_backbone(addgene_id: str) -> Optional[Dict]:
    """Fetch a plasmid from Addgene and return it as a backbone library dict.

    Fetches metadata, sequence, and GenBank feature annotations (promoters,
    resistance genes, MCS, polyA signals, etc.) in one call.

    Returns:
        Backbone dict in library format (same shape as entries in
        backbones.json), or None on failure.
    """
    client = AddgeneClient()
    plasmid = client.get_plasmid(addgene_id)
    if not plasmid:
        return None

    sequence, features, mcs_position = client.get_genbank_data(addgene_id)
    if sequence:
        plasmid.sequence = sequence
        plasmid.sequence_source = "addgene"
        plasmid.size_bp = len(sequence)
    if features:
        plasmid.parsed_features = features
    if mcs_position:
        plasmid.mcs_position = mcs_position

    return plasmid.to_backbone_dict()


if __name__ == "__main__":
    # Test the integration
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        addgene_id = sys.argv[1]
        print(f"\nFetching Addgene #{addgene_id}...")
        
        client = AddgeneClient()
        plasmid = client.get_plasmid(addgene_id)
        
        if plasmid:
            print(f"\nName: {plasmid.name}")
            print(f"Description: {plasmid.description}")
            print(f"Size: {plasmid.size_bp} bp")
            print(f"Resistance: {plasmid.bacterial_resistance}")
            print(f"URL: {plasmid.url}")
            
            print("\nFetching sequence...")
            seq = client.get_sequence(addgene_id)
            if seq:
                print(f"Sequence: {len(seq)} bp")
                print(f"First 100 bp: {seq[:100]}...")
            else:
                print("Sequence not available")
        else:
            print("Plasmid not found")
    else:
        print("Usage: python addgene_integration.py <addgene_id>")
        print("\nExample: python addgene_integration.py 50005")
