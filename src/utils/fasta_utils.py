def format_as_fasta(sequence: str, name: str, description: str = "") -> str:
    """Format a sequence as FASTA with 80-character line wrapping."""
    header = f">{name}"
    if description:
        header += f" {description}"

    lines = [header]
    for i in range(0, len(sequence), 80):
        lines.append(sequence[i:i + 80])

    return "\n".join(lines) + "\n"