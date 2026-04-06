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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    depositor_comments: Optional[str] = None
    size_bp: Optional[int] = None
    backbone: Optional[str] = None
    backbone_origin: Optional[str] = None
    promoter: Optional[str] = None
    bacterial_resistance: Optional[str] = None
    mammalian_selection: Optional[str] = None
    vector_type: Optional[str] = None
    species: Optional[str] = None
    gene_insert: Optional[str] = None
    pubmed_id: Optional[str] = None
    sequence: Optional[str] = None
    sequence_source: Optional[str] = None
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
            "depositor_comments": self.depositor_comments,
            "size_bp": self.size_bp,
            "source": "Addgene",
            "organism": self._infer_organism(),
            "backbone_origin": self.backbone_origin,
            "promoter": self.promoter,
            "bacterial_resistance": self.bacterial_resistance,
            "mammalian_selection": self.mammalian_selection,
            "mcs_position": self.mcs_position,
            "features": self.parsed_features or [],
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
    API_BASE_URL = "https://api.developers.addgene.org"  # Official API (requires auth)
    
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
            headers["Authorization"] = f"Token {self.api_token}"
        
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
            #https://api.developers.addgene.org/catalog/plasmid-with-sequences
            url = f"{self.API_BASE_URL}/catalog/plasmid-with-sequences/{addgene_id}"
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
        cloning = data.get("cloning") or {}
        return AddgenePlasmid(
            addgene_id=str(data.get("id", "")),
            name=data.get("name"),
            description=data.get("description"),
            depositor_comments=data.get("depositor_comments"),
            size_bp=cloning.get("backbone_size"),
            backbone=cloning.get("backbone"),
            backbone_origin=cloning.get("backbone_origin"),
            promoter=data.get("promoter"),
            bacterial_resistance=data.get("bacterial_resistance"),
            mammalian_selection=data.get("resistance_markers"),
            vector_type=cloning.get("vector_types"),
            species=data.get("growth_strain"),
            gene_insert=data.get("inserts")[0].get("name") if data.get("inserts") else None,
            pubmed_id=data.get("article", {}).get("pubmed_id"),
            sequence=data.get("sequences", {}).get("public_addgene_full_sequences", [{}])[0].get("sequence"),
            url=data.get("article", {}).get("url"),
        )
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search Addgene for plasmids.
        
        Args:
            query: Search term (plasmid name, gene, etc.)
            limit: Maximum number of results
        
        Returns:
            List of search results with basic info
        """
        return self._search_scrape(query, limit)
    

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
    
    def import_plasmid(self, addgene_id: str) -> Optional[Dict]:
        """
        Import a plasmid from Addgene into the local library.

        Args:
            addgene_id: Addgene catalog number

        Returns:
            The imported backbone dict, or None if failed
        """
        plasmid = self.client.get_plasmid(addgene_id)
        if not plasmid:
            logger.error(f"Could not fetch plasmid {addgene_id} from Addgene")
            return None

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
    

# Convenience functions for use in MCP tools
def search_addgene(query: str, limit: int = 100) -> List[Dict]:
    """Search Addgene for plasmids."""
    client = AddgeneClient()
    return client.search(query, limit)


def fetch_addgene_sequence_with_metadata(addgene_id: str) -> Optional[AddgenePlasmid]:
    """Fetch a plasmid from Addgene and return the full AddgenePlasmid dataclass,
    including sequence and all available metadata fields (name, description,
    promoter, resistance markers, vector type, etc.)."""
    client = AddgeneClient()
    return client.get_plasmid(addgene_id)


def fetch_addgene_sequence(addgene_id: str) -> Optional[str]:
    """Fetch only the DNA sequence string for an Addgene plasmid.
    Use this when you need the raw sequence and don't need metadata."""
    plasmid = fetch_addgene_sequence_with_metadata(addgene_id)
    return plasmid.sequence if plasmid else None


def fetch_addgene_sequence_with_metadata_dict(addgene_id: str) -> Optional[Dict]:
    """Fetch a plasmid from Addgene and return it as a backbone library dict
    (same shape as entries in backbones.json). Use this when importing a plasmid
    directly into the library or passing it to the assembly engine."""
    client = AddgeneClient()
    plasmid = client.get_plasmid(addgene_id)
    if not plasmid:
        return None
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
            
            if plasmid.sequence:
                print(f"\nSequence: {len(plasmid.sequence)} bp")
                print(f"First 100 bp: {plasmid.sequence[:100]}...")
            else:
                print("Sequence not available")
        else:
            print("Plasmid not found")
    else:
        print("Usage: python addgene_integration.py <addgene_id>")
        print("\nExample: python addgene_integration.py 50005")
