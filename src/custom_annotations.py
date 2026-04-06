#!/usr/bin/env python3
"""
Custom Annotation Database — Bring Your Own Annotations (BYOA)

Extends pLannotate with user-provided sequence annotations from GenBank files
placed in $PLASMID_USER_LIBRARY/annotations/. Features annotated in those files
are extracted into a local BLAST database and merged into pLannotate results,
so lab-private or recently-published sequences are recognised during plasmid
annotation and region extraction.

Directory layout:
    $PLASMID_USER_LIBRARY/
        annotations/            ← GenBank files with annotated features
            my_promoter.gb
            new_fluorescent_protein.gb
            ...

Each GenBank file may contain one or more feature annotations. Features with
a /label (or /gene or /product) qualifier are extracted. The feature type and
label become the annotation identity, e.g. CDS::mCerulean3, promoter::myPromoter.

The BLAST database is rebuilt automatically when source files change (MD5
manifest). If BLAST is not installed or no annotation files exist the module
silently returns None from query_custom_db() — pLannotate-only behaviour is
preserved.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GENBANK_EXTENSIONS = (".gb", ".gbk", ".genbank")
_FEAT_SKIP = {"source", "primer_bind"}

# Module-level state — initialised once by setup_custom_annotations()
_annotation_dir: Optional[Path] = None
_blast_db_dir: Optional[Path] = None
_fasta_path: Optional[Path] = None
_manifest_path: Optional[Path] = None
_db_ready: bool = False
_setup_done: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_custom_annotations() -> None:
    """Scan $PLASMID_USER_LIBRARY/annotations/ and (re)build the BLAST DB.

    Idempotent — safe to call multiple times; rebuilds only when files change.
    Silently skips if the directory is absent or BLAST is not installed.
    """
    global _annotation_dir, _blast_db_dir, _fasta_path, _manifest_path
    global _db_ready, _setup_done

    if _setup_done:
        return
    _setup_done = True

    raw = os.environ.get("PLASMID_USER_LIBRARY")
    if not raw:
        return
    user_lib = Path(raw).expanduser()
    if not user_lib.is_dir():
        return

    ann_dir = user_lib / "annotations"
    if not ann_dir.is_dir():
        return

    gb_files = _scan_genbank_files(ann_dir)
    if not gb_files:
        logger.debug("custom_annotations: no GenBank files in %s", ann_dir)
        return

    db_dir = ann_dir / ".blast_db"
    fasta = db_dir / "custom_features.fasta"
    manifest = db_dir / "manifest.json"

    _annotation_dir = ann_dir
    _blast_db_dir = db_dir
    _fasta_path = fasta
    _manifest_path = manifest

    if not _blast_available():
        logger.warning(
            "custom_annotations: makeblastdb/blastn not found — "
            "custom annotations disabled. Install BLAST+ (conda install -c bioconda blast)."
        )
        return

    if _needs_rebuild(gb_files, db_dir, manifest):
        db_dir.mkdir(exist_ok=True)
        count = _extract_features_to_fasta(gb_files, fasta)
        if count == 0:
            logger.info("custom_annotations: no annotated features found in GenBank files")
            return
        if not _run_makeblastdb(fasta, db_dir):
            return
        _write_manifest(gb_files, manifest)
        logger.info(
            "custom_annotations: built BLAST DB with %d features from %d file(s)",
            count, len(gb_files),
        )
    else:
        logger.debug("custom_annotations: BLAST DB is up to date")

    _db_ready = True


def query_custom_db(sequence: str):
    """Run blastn against the custom annotation DB.

    Returns a pLannotate-compatible pandas DataFrame, or None if the custom
    DB is not available or returns no hits.
    """
    if not _db_ready or _blast_db_dir is None:
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    db_prefix = str(_blast_db_dir / "custom_features")
    rows = _run_blastn(sequence, db_prefix)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df if not df.empty else None


def merge_annotation_results(base_df, custom_df):
    """Merge custom annotation rows into a pLannotate DataFrame.

    Custom rows win on overlap: if a custom hit fully covers a base_df feature
    at >= equal identity, the base_df row is dropped. Otherwise custom rows
    are appended. Returns a new DataFrame (base_df is not modified).
    """
    try:
        import pandas as pd
    except ImportError:
        return base_df

    if custom_df is None or custom_df.empty:
        return base_df

    # Ensure all expected columns exist in custom_df (fill missing with defaults)
    ref_cols = list(base_df.columns) if (base_df is not None and not base_df.empty) else [
        "Feature", "Type", "qstart", "qend", "sframe",
        "pident", "percmatch", "db", "Description",
    ]
    custom_df = custom_df.copy()
    for col in ref_cols:
        if col not in custom_df.columns:
            custom_df[col] = "" if col in ("Feature", "Type", "db", "Description", "qseq") else 0

    if base_df is None or base_df.empty:
        return custom_df[ref_cols]

    # Drop base rows fully covered by a higher- or equal-identity custom row
    rows_to_drop = set()
    for _, crow in custom_df.iterrows():
        cs = int(crow["qstart"])
        ce = int(crow["qend"])
        ci = float(crow["pident"])
        for idx, brow in base_df.iterrows():
            bs = int(brow["qstart"])
            be = int(brow["qend"])
            if be <= bs:
                continue  # origin-spanning base row — skip overlap logic
            overlap = max(0, min(ce, be) - max(cs, bs))
            base_span = be - bs
            if base_span > 0 and (overlap / base_span) >= 0.9 and ci >= float(brow["pident"]):
                rows_to_drop.add(idx)

    filtered_base = base_df.drop(index=list(rows_to_drop)) if rows_to_drop else base_df
    return pd.concat([filtered_base, custom_df[ref_cols]], ignore_index=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _scan_genbank_files(directory: Path) -> list:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in GENBANK_EXTENSIONS and p.is_file()
    )


def _blast_available() -> bool:
    try:
        subprocess.run(
            ["makeblastdb", "-version"],
            check=True, capture_output=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _needs_rebuild(gb_files: list, db_dir: Path, manifest: Path) -> bool:
    if not manifest.exists():
        return True
    # Check that at least one BLAST DB index file exists
    if not list(db_dir.glob("custom_features.n*")):
        return True
    try:
        stored = json.loads(manifest.read_text())
        current = _file_hashes(gb_files)
        return current != stored
    except Exception:
        return True


def _file_hashes(files: list) -> dict:
    return {str(f): hashlib.md5(Path(f).read_bytes()).hexdigest() for f in files}


def _write_manifest(gb_files: list, manifest: Path) -> None:
    manifest.write_text(json.dumps(_file_hashes(gb_files), indent=2))


def _extract_features_to_fasta(gb_files: list, out_fasta: Path) -> int:
    """Extract annotated features from GenBank files into a FASTA file.

    Each feature is written as:
        >type::label
        ATGCATGC...

    Features without a /label, /gene, or /product qualifier are skipped.
    Returns the number of feature sequences written.
    """
    try:
        from Bio import SeqIO
    except ImportError:
        logger.error("custom_annotations: BioPython not available — cannot extract features")
        return 0

    count = 0
    with out_fasta.open("w") as fh:
        for gb_path in gb_files:
            try:
                records = list(SeqIO.parse(str(gb_path), "genbank"))
            except Exception as e:
                logger.warning("custom_annotations: could not parse %s: %s", gb_path.name, e)
                continue
            for record in records:
                for feat in record.features:
                    if feat.type in _FEAT_SKIP:
                        continue
                    label = (
                        feat.qualifiers.get("label", [""])[0]
                        or feat.qualifiers.get("gene", [""])[0]
                        or feat.qualifiers.get("product", [""])[0]
                    ).strip()
                    if not label:
                        continue
                    try:
                        feat_seq = str(feat.extract(record.seq)).upper()
                    except Exception:
                        continue
                    if len(feat_seq) < 10:
                        continue
                    # Encode type::label — sanitise spaces/colons for BLAST ID safety
                    safe_label = label.replace(" ", "_").replace("::", "__").replace(":", "_")
                    feat_id = f"{feat.type}::{safe_label}"
                    fh.write(f">{feat_id}\n{feat_seq}\n")
                    count += 1
    return count


def _run_makeblastdb(fasta: Path, db_dir: Path) -> bool:
    db_prefix = str(db_dir / "custom_features")
    try:
        result = subprocess.run(
            [
                "makeblastdb",
                "-in", str(fasta),
                "-dbtype", "nucl",
                "-out", db_prefix,
                "-title", "custom_annotations",
            ],
            capture_output=True, text=True, check=True,
        )
        logger.debug("makeblastdb: %s", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error("custom_annotations: makeblastdb failed: %s", e.stderr)
        return False


def _run_blastn(sequence: str, db_prefix: str) -> list:
    """Run blastn against the custom DB; return pLannotate-compatible row dicts."""
    outfmt = (
        "6 qseqid sseqid pident length qstart qend sstart send sstrand slen qlen evalue bitscore"
    )
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
            tmp.write(f">query\n{sequence}\n")
            query_file = tmp.name

        result = subprocess.run(
            [
                "blastn",
                "-query", query_file,
                "-db", db_prefix,
                "-outfmt", outfmt,
                "-perc_identity", "80",
                "-word_size", "11",
                "-dust", "no",
                "-soft_masking", "false",
            ],
            capture_output=True, text=True, check=True,
        )
        os.unlink(query_file)
    except FileNotFoundError:
        logger.debug("custom_annotations: blastn not found")
        return []
    except subprocess.CalledProcessError as e:
        logger.debug("custom_annotations: blastn failed: %s", e.stderr)
        return []

    rows = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 13:
            continue
        try:
            (_, sseqid, pident, length, qstart, qend,
             _sstart, _send, sstrand, slen, _qlen, evalue, bitscore) = parts

            pident = float(pident)
            length = int(length)
            qstart = int(qstart) - 1   # blastn is 1-based; convert to 0-based
            qend = int(qend)            # exclusive end after 0-based conversion
            slen = int(slen)

            # Parse feature type and label from FASTA ID (format: "type::label")
            if "::" in sseqid:
                feat_type, feat_label = sseqid.split("::", 1)
                feat_label = feat_label.replace("_", " ")
            else:
                feat_type = "misc_feature"
                feat_label = sseqid

            strand = 1 if sstrand == "plus" else -1
            percmatch = round(length / slen * 100, 1) if slen > 0 else 0.0

            rows.append({
                "Feature": feat_label,
                "Type": feat_type,
                "qstart": qstart,
                "qend": qend,
                "sframe": strand,
                "pident": pident,
                "percmatch": percmatch,
                "db": "custom_db",
                "Description": "",
                "qseq": sequence[qstart:qend],
                "score": float(bitscore),
                "evalue": float(evalue),
            })
        except (ValueError, IndexError):
            continue
    return rows
