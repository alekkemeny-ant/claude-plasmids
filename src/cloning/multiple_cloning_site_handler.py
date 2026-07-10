from typing import Optional
import re
import logging
from src.utils.sequence_utils import reverse_complement

logger = logging.getLogger(__name__)

# Common MCS recognition patterns (restriction sites commonly found in MCS)
COMMON_MCS_PATTERNS = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "KpnI": "GGTACC",
    "XbaI": "TCTAGA",
    "SalI": "GTCGAC",
    "PstI": "CTGCAG",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "NheI": "GCTAGC",
    "SmaI": "CCCGGG",
    "ApaI": "GGGCCC",
}

class MCSHandler:
    """Handles finding and inserting genes into plasmid MCS (Multiple Cloning Site)."""
    @staticmethod
    def find_mcs_sites(backbone_seq: str) -> list:
        """
        Find common restriction sites in the backbone that likely define the MCS.

        Args:
            backbone_seq: Backbone sequence string

        Returns:
            List of dicts: {name, position, end_position, pattern}
        """
        sites = []
        backbone_upper = backbone_seq.upper()

        for site_name, pattern in COMMON_MCS_PATTERNS.items():
            matches = re.finditer(pattern, backbone_upper)
            for match in matches:
                sites.append({
                    "name": site_name,
                    "position": match.start(),
                    "end_position": match.end(),
                    "pattern": pattern
                })

        # Sort by position
        sites.sort(key=lambda x: x["position"])
        return sites

    @staticmethod
    def find_mcs_boundaries(backbone_seq: str, max_gap: int = 40) -> Optional[tuple]:
        """
        Find the MCS boundaries by identifying the densest cluster of
        restriction sites. An MCS is characterized by many unique sites
        packed into a short region. Sites separated by more than max_gap bp
        (measured from end of one site to start of the next) are considered
        outside the cluster.

        Args:
            backbone_seq: Backbone sequence string
            max_gap: Maximum bp gap between the end of one site and the
                        start of the next to be considered part of the same
                        cluster. Default 40 bp — real MCS sites are typically
                        0-30 bp apart.

        Returns:
            Tuple of (start_position, end_position) of the best cluster,
            or None if fewer than 3 sites found.
        """
        sites = MCSHandler.find_mcs_sites(backbone_seq)

        if len(sites) < 3:
            logger.warning("Could not find enough restriction sites for MCS")
            return None

        # Group sites into clusters based on max_gap between consecutive sites
        # Gap is measured from end_position of previous site to position of next
        clusters = []
        current_cluster = [sites[0]]

        for i in range(1, len(sites)):
            gap = sites[i]["position"] - current_cluster[-1]["end_position"]
            if gap <= max_gap:
                current_cluster.append(sites[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sites[i]]
        clusters.append(current_cluster)

        # Pick the cluster with the most unique enzyme sites
        best_cluster = max(clusters, key=lambda c: len(set(s["name"] for s in c)))

        if len(best_cluster) < 3:
            logger.warning("No dense restriction site cluster found")
            return None

        start = best_cluster[0]["position"]
        end = best_cluster[-1]["end_position"]
        return (start, end)

    @staticmethod
    def detect_mcs_direction(mcs_bounds: tuple, features: Optional[list[dict]] = None) -> str:
        """
        Detect whether the MCS is in forward or reverse orientation by
        looking at the nearest promoter relative to the MCS cluster.

        Forward: promoter is upstream (lower bp) of MCS → insert at MCS start
        Reverse: promoter is downstream (higher bp) of MCS → insert at MCS end

        Args:
            mcs_bounds: Tuple of (start_position, end_position) from find_mcs_boundaries.
            features: List of feature dicts with name, type, start, end keys.

        Returns:
            "forward" or "reverse"
        """
        if not features:
            return "forward"

        mcs_center = (mcs_bounds[0] + mcs_bounds[1]) / 2

        # Find promoter features
        promoters = []
        for feat in features:
            feat_type = feat.get("type", "").lower()
            feat_name = feat.get("name", "").lower()
            # Skip bacterial resistance promoters (e.g. bla)
            if "bla" in feat_name or "resistance" in feat_name:
                continue
            if feat_type == "promoter" or "promoter" in feat_name:
                promoters.append(feat)

        if not promoters:
            return "forward"

        # Find the promoter closest to the MCS
        closest_promoter = min(
            promoters,
            key=lambda p: min(abs(p["start"] - mcs_center), abs(p["end"] - mcs_center))
        )

        # If the promoter center is at a higher bp position than the MCS center,
        # the expression cassette runs in reverse
        promoter_center = (closest_promoter["start"] + closest_promoter["end"]) / 2
        if promoter_center > mcs_center:
            return "reverse"

        return "forward"

    @staticmethod
    def insert_gene_at_mcs(
        backbone_seq: str,
        gene_seq: str,
        insertion_point: Optional[int] = None,
        features: Optional[list[dict]] = None,
    ) -> dict:
        """
        Insert gene sequence into plasmid at MCS or specified position.
        Automatically detects MCS direction from features to place the
        insert at the correct end of the MCS (closest to the promoter).

        Args:
            backbone_seq: Backbone sequence string
            gene_seq: Gene sequence to insert
            insertion_point: Optional specific position to insert. If None, use MCS detection.
            features: Optional list of backbone feature dicts (for direction detection).

        Returns:
            Dictionary with:
            - final_sequence: The resulting construct
            - insertion_position: Where the gene was inserted
            - method: How the insertion was performed
            - direction: "forward" or "reverse"
            - mcs_sites: All restriction sites found
        """
        if not backbone_seq or not gene_seq:
            return {
                "final_sequence": None,
                "insertion_position": None,
                "method": "error",
                "direction": None,
                "error": "Missing backbone or gene sequence"
            }

        direction = "forward"

        # Try to find MCS boundaries
        if insertion_point is None:
            mcs_bounds = MCSHandler.find_mcs_boundaries(backbone_seq)
            if mcs_bounds:
                direction = MCSHandler.detect_mcs_direction(mcs_bounds, features)
                if direction == "reverse":
                    insertion_point = mcs_bounds[1]  # End of cluster (closer to promoter)
                else:
                    insertion_point = mcs_bounds[0]  # Start of cluster (closer to promoter)
                method = "mcs"
                logger.info(f"MCS detected ({direction}): inserting at position {insertion_point}")
            else:
                # Fallback: try to find promoter and insert after it
                promoter_match = re.search(r"CMV|SV40|EF1A|UBC", backbone_seq.upper())
                if promoter_match:
                    insertion_point = promoter_match.end() + 100  # Insert 100bp after promoter start
                    method = "after_promoter"
                    logger.info(f"MCS not found, inserting after promoter at position {insertion_point}")
                else:
                    # Default: concatenate
                    insertion_point = len(backbone_seq)
                    method = "concatenation"
                    logger.warning("Could not find MCS or promoter, using concatenation")
        else:
            method = "custom_position"

        # Insert the gene
        if insertion_point < 0 or insertion_point > len(backbone_seq):
            insertion_point = len(backbone_seq)

        if direction == "reverse":
            gene_seq = reverse_complement(gene_seq)

        final_sequence = backbone_seq[:insertion_point] + gene_seq + backbone_seq[insertion_point:]

        return {
            "final_sequence": final_sequence,
            "insertion_position": insertion_point,
            "method": method,
            "direction": direction,
            "mcs_sites": MCSHandler.find_mcs_sites(backbone_seq)
        }

    @staticmethod
    def resolve_insertion_point(
        backbone: dict,
        backbone_seq: str,
        ) -> tuple[Optional[int], bool]:
        """Return (insertion_position, reverse_complement_insert).

        First tries the pre-stored mcs_position in the backbone dict.
        If that is absent, runs MCSHandler on the raw sequence to detect
        the MCS cluster and promoter direction automatically.
        Returns (None, False) if no position can be determined.
        """
        pos = MCSHandler.find_mcs_insertion_point(backbone)
        if pos is not None:
            return pos, False

        bounds = MCSHandler.find_mcs_boundaries(backbone_seq)
        if bounds is None:
            return None, False

        features = backbone.get("features")
        direction = MCSHandler.detect_mcs_direction(bounds, features)
        if direction == "reverse":
            return bounds[1], True  # end of MCS cluster, RC insert
        else:
            return bounds[0], False  # start of MCS cluster, no RC

    @staticmethod
    def find_mcs_insertion_point(backbone: dict) -> Optional[int]:
        """
        Determine the best insertion point within the MCS of a backbone.

        For expression vectors, the insert should go at the start of the MCS
        (immediately downstream of the promoter). Returns a 0-based position.

        Args:
            backbone: Backbone dict from the library (must have mcs_position).

        Returns:
            Insertion position (0-based), or None if no MCS info available.
        """
        mcs = backbone.get("mcs_position")
        if not mcs:
            return None

        # Default: insert at the start of the MCS
        # This places the insert immediately downstream of the promoter/5' UTR
        return mcs["start"]
