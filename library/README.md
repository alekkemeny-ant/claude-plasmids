# Plasmid Library

This directory contains the built-in backbone and insert libraries (`backbones.json`, `inserts.json`). You can extend the agent with your own parts without editing these files by using the **Bring Your Own Library (BYOL)** feature.

---

## Bring Your Own Library (BYOL)

BYOL lets you load your own plasmid sequences into the agent from a local directory of GenBank files. Sequences you provide are kept entirely separate from the built-in library and are never written back to disk by the agent.

### 1. Set up your library directory

Create a directory anywhere on your machine with this layout:

```
my_library/
├── backbones/
│   ├── pMyVector.gb
│   └── pAnotherBackbone.gbk
├── inserts/
│   ├── MyPromoter.gb
│   └── MyORF.gbk
├── backbones_description.csv      ← optional metadata
└── inserts_description.csv        ← optional metadata
```

Accepted file extensions: `.gb`, `.gbk`, `.genbank`

### 2. Point the agent at your library

Set the environment variable before launching the app:

```bash
export PLASMID_USER_LIBRARY=/path/to/my_library
python app/app.py
```

Or add it to your `.env` file:

```
PLASMID_USER_LIBRARY=/path/to/my_library
```

### 3. Use your parts

Your entries appear in the library with a `user:` ID prefix (e.g. `user:pMyVector`). You can refer to them by that ID or by their LOCUS name. The agent will find them automatically when you search or ask for them by name.

---

## How sequences are inferred from GenBank files

The agent reads the following directly from each `.gb` file:

| Inferred field | Source |
|---|---|
| ID | `LOCUS` name (filename stem used as fallback) |
| Sequence | `ORIGIN` section |
| Size | Declared in `LOCUS` line |
| Features | `FEATURES` table (promoters, CDS, MCS, etc.) |
| MCS position | First feature annotated as `misc_feature` with "MCS" in label/note |
| Topology | `LOCUS` line — `circular` or `linear` |

**Part-in-vector detection:** If an insert file is declared `circular` in its LOCUS line, it is automatically treated as a `part_in_vector` (a Golden Gate carrier plasmid). The full sequence is stored as `plasmid_sequence` for use by the Golden Gate assembly tool, which will excise the insert at the Type IIS sites. Linear insert files are stored as plain sequences.

---

## Enriching metadata with CSV files

GenBank LOCUS names are often cryptic (e.g. `pLAB_V0027`). The optional CSV files let you supply human-readable names, aliases, assembly enzyme information, overhang sequences, and more.

### `inserts_description.csv`

| Column | Library field | Notes |
|---|---|---|
| `id` | Match key | Must match the GenBank LOCUS name exactly |
| `Description` | `name` + `aliases` | Full value becomes the display name; tokens split on `-` become aliases |
| `TypeIIS cutsite` | `assembly_enzyme` | e.g. `Esp3I`, `BsaI`, `BbsI` |
| `Overhang L` | `overhang_left` | 4-nt left overhang after Type IIS digestion |
| `Overhang R` | `overhang_right` | 4-nt right overhang after Type IIS digestion |
| `Size` | `insert_size_bp` | Size of the **excised insert** — stored separately from `size_bp` (the full carrier vector size from GenBank) |
| `Selection` | `bacterial_resistance` | e.g. `AmpR`, `KanR` |
| `Category` | `category` | See [Insert categories](#insert-categories) below |

**Example `inserts_description.csv`:**

```
id	Description	TypeIIS cutsite	Overhang L	Overhang R	Size	Selection	Category
LAB_Part001	Kozak-GFP-Variant-Alpha	Esp3I	CACC	CTGG	732	AmpR	part_in_vector
LAB_Part002	SV40-NLS-Peptide	Esp3I	AACG	GTTT	42	AmpR	epitope_tag
LAB_Part003	Custom-Luciferase-Codon-Optimized	BsaI	AATG	GCTT	1653	KanR	reporter
LAB_Part004	mTurquoise2-FP	Esp3I	CTGG	ATCC	720	AmpR	fluorescent_protein
LAB_Part005	GAPDH-Reference-Gene	BbsI	ACCG	GTTT	1008	AmpR	gene
```

---

### `backbones_description.csv`

| Column | Library field | Notes |
|---|---|---|
| `ID` | Match key | Must match the GenBank LOCUS name exactly |
| `E coli strain` | `ecoli_strain` | Propagation strain, e.g. `DH5alpha`, `Stabl3` |
| `Antibiotic resistance` | `bacterial_resistance` | e.g. `AmpR`, `KanR` |
| `Neg selection marker` | `description` | Included in auto-generated description |
| `Assembly enzyme` | `assembly_enzyme` | e.g. `Esp3I`, `BbsI` |
| `Overhang pair 1` | `overhang_left` / `overhang_right` | Format: `XXXX-YYYY` (split on `-`) |
| `Next step enzyme` | `next_step_enzyme` | Enzyme for the downstream cloning step |
| `Overhang pair 2` | `overhang_left_2` / `overhang_right_2` | Second cloning window, same `XXXX-YYYY` format |
| `Downstream` | `description` | Included in auto-generated description |
| `Mammalian selection marker` | `mammalian_selection` | e.g. `PuroR`, `BlastR`, `NeoR` |

**Example `backbones_description.csv`:**

```
ID	E coli strain	Antibiotic resistance	Neg selection marker	Assembly enzyme	Overhang pair 1	Next step enzyme	Overhang pair 2	Downstream	Mammalian selection marker
pLAB_V001	DH5alpha	AmpR	mCherry	Esp3I	CACC-CTGG			pExpression	PuroR
pLAB_V002	Stabl3	KanR	ccdB	BbsI	ACCG-GTTT	gRNA insertion	AACG-GTTT	pCRISPR	BlastR
pLAB_V003	DH10B	AmpR	LacZ	BsaI	AATG-GCTT			pReporter	NeoR
```

Both comma-separated (`.csv`) and tab-separated files are accepted — the format is detected automatically.

---

## Insert categories

Set the `Category` column in `inserts_description.csv` (or rely on automatic detection for `part_in_vector`) to one of the following:

| Category | Description | Example use |
|---|---|---|
| `gene` | A protein-coding sequence (CDS) | Human TP53, mouse MyD88 |
| `fluorescent_protein` | An engineered fluorescent protein | EGFP, mCherry, mScarlet |
| `reporter` | A reporter gene (non-fluorescent) | Firefly luciferase, Renilla luciferase |
| `epitope_tag` | A short peptide tag fused to a protein | FLAG, HA, His6, Myc |
| `part_in_vector` | A Golden Gate insert carried in a circular plasmid | Modular promoter/ORF/terminator parts |

If no category is specified and the file is linear, the entry is loaded without a category. The agent will still use it but won't apply category-specific logic (e.g. linker handling for epitope tags).

---

## Tips

- **LOCUS name = CSV id.** The `id`/`ID` column in both CSVs must exactly match the LOCUS name declared in the corresponding `.gb` file, not the filename. Check the first line of your GenBank file: `LOCUS  <name>  ...`
- **CSV rows with no matching file are skipped** with a warning in the logs. This lets you maintain a master CSV across your whole collection and only place the files you want to use in the directory.
- **Files with no CSV row load fine** — they just use the LOCUS name as the display name and have no enriched metadata.
- **The built-in library is never modified.** Your entries are loaded at runtime and namespaced under `user:`. Removing files from your directory removes them from the agent instantly on next restart. Restart the app to load the library if the app is already running.

