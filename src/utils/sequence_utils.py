import re


def clean_sequence(sequence: str) -> str:
    """Remove whitespace and normalize to uppercase."""
    return re.sub(r'\s', '', sequence.upper())


def validate_dna(sequence: str) -> tuple[bool, list[str]]:
    """Validate that a string is valid DNA. Returns (is_valid, errors)."""
    errors = []
    if not sequence:
        errors.append("Sequence is empty")
        return False, errors

    invalid_chars = set(sequence) - set('ATCGN')
    if invalid_chars:
        errors.append(f"Invalid characters in sequence: {sorted(invalid_chars)}")
        return False, errors

    return True, []


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(sequence))
