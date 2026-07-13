"""
Code relating to export functionality
"""
from typing import Optional
from src.cloning.assembler import AssemblyResult
from src.utils.genbank_utils import format_as_genbank
from src.utils.fasta_utils import format_as_fasta

def export_construct(
    result: AssemblyResult,
    output_format: str,
    construct_name: str = "construct",
    backbone_name: str = "",
    insert_name: str = "",
    reverse_complement_insert: bool = False,
    insert_length: int = 0,
    backbone_features: Optional[list[dict]] = None,
) -> str:
    """
    Export an assembled construct in the requested format.

    Args:
        result: A successful AssemblyResult.
        output_format: One of "fasta", "genbank", "raw".
        construct_name: Name for the output record.
        backbone_name: Name of backbone (for GenBank annotation).
        insert_name: Name of insert (for GenBank annotation).
        insert_length: Length of insert (for GenBank annotation).
        backbone_features: Original backbone features (for GenBank annotation).
        reverse_complement_insert: bool = False
    Returns:
        Formatted sequence string.

    Raises:
        ValueError: If result is not successful or format is unknown.
    """
    if not result.success or not result.sequence:
        raise ValueError("Cannot export a failed assembly result")

    fmt = output_format.lower().strip()

    if fmt == "raw":
        return result.sequence

    elif fmt == "fasta":
        desc = f"{insert_name} in {backbone_name}, {result.total_size_bp} bp"
        return format_as_fasta(result.sequence, construct_name, desc)

    elif fmt in ("genbank", "gb"):
        return format_as_genbank(
            sequence=result.sequence,
            name=construct_name,
            backbone_name=backbone_name,
            insert_name=insert_name,
            insert_position=result.insert_position or 0,
            insert_length=insert_length,
            features=backbone_features,
            reverse_complement_insert=reverse_complement_insert,
        )

    else:
        raise ValueError(f"Unknown output format: {output_format!r}. Use 'raw', 'fasta', or 'genbank'.")
