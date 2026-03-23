# PR Review Drafts — #9 and #10

**NOT POSTED. Approve before posting to GitHub.**

---

## PR #9 — Mialy's additions (commits `5fd7648..e42d033`)

### Summary

The CSV metadata overlay and library inspection UI are well-built. The Golden Gate compat layer has one real bug and one footgun, both in `_apply_insert_csv_meta`. Tests are good but have a coverage gap on the exact codepath the GG compat exists for.

---

### P1 — Overhang key mismatch with PR #10's assembler

`src/user_library.py:147,151` writes `overhang_left` / `overhang_right`. PR #10's `assemble_golden_gate` reads `overhang_l` / `overhang_r` (`src/tools.py:1239-1240` → `src/assembler.py:679-680`).

The assembler falls back to auto-detecting overhangs from cut sites, so nothing crashes — but user-provided overhang hints are silently ignored. This matters: `_excise_insert` uses those hints to pick the *correct* flanking sites when a carrier vector has extra enzyme sites (e.g., a stray Esp3I in AmpR). Without the hint, it uses leftmost/rightmost by position, which can excise the wrong fragment.

Fix: rename to `overhang_l` / `overhang_r` in `_apply_insert_csv_meta` and `_apply_backbone_csv_meta`, plus the README tables at lines 78-79 and 106-108.

---

### P1 — CSV Category clobbers auto-detected `part_in_vector`

`_parse_file_to_entry` sets `category="part_in_vector"` at line 269 for circular inserts. Then `_apply_insert_csv_meta` at line 168 unconditionally overwrites `entry["category"]` from the CSV `Category` column.

So if a user has a circular `.gb` file and their CSV says `Category=insert`, the auto-detected `part_in_vector` is lost and the golden gate tool won't find it. The README at line 127 says "or rely on automatic detection for `part_in_vector`" — but that only works if the CSV row leaves `Category` blank.

Two options:
- Make `part_in_vector` sticky (skip CSV overwrite when already set to `part_in_vector`)
- Or document that CSV `Category` wins and auto-detection is only a fallback when the column is empty

I'd lean toward the first — a circular insert file is unambiguously a carrier plasmid regardless of what the CSV says.

---

### P2 — No test for the circular-insert → `part_in_vector` path

The fixture insert `myGene.gbk` is declared `linear` in its LOCUS line, so `_parse_file_to_entry` lines 265-269 are never exercised. Add a second fixture with a `circular` LOCUS and assert that it gets `plasmid_sequence` (not `sequence`) and `category == "part_in_vector"`.

---

### What's good

- **CSV overlay architecture is clean.** GenBank is the source of truth; CSV enriches. The decoupling means users can drop files in and iterate on metadata separately.
- **Frontend escapes everything.** `escapeHtml()` on every user-controlled string in `_ulDetailRows` and `_ulBuildEntries`, and the `eid` sanitization (`replace(/[^a-zA-Z0-9_-]/g, '_')`) makes the DOM IDs and onclick args safe. No XSS from CSV-sourced data.
- **Sequence stripping in the API endpoint** — only metadata crosses the wire to the frontend, not the full sequences. Good instinct for a panel that will eventually show dozens of entries.
- **`test_builtin_loader_never_includes_user`** — nice. This is the exact invariant that prevents user entries from getting accidentally written into `backbones.json` via the Addgene cache path.
- **README is excellent.** Clear structure, concrete examples, covers both CSV schemas with column-by-column tables.
- **CLI smoke test passed** — `--list-library` prints enriched metadata correctly.

---

### Nits

- `_load_backbone_csv` at line 107 just calls `_load_insert_csv` — the indirection through a second function name doesn't buy anything since both paths hit line 304's conditional anyway. Could inline.
- `csv.Sniffer().sniff(text[:4096])` can raise `csv.Error` on degenerate input (single-column or very short files). The `except Exception` at line 97 catches it, but the result is a silent `{}` with only a log warning. Probably fine for now but worth a mention in the README that malformed CSVs fail silently.

---

---

## PR #10 — Golden Gate Assembly

### Summary

The assembler math is solid and well-tested. Main issue is branch hygiene: this is stacked on PR #7 (not #9 as Slack suggested), carries 15 unrelated commits, and will conflict with #9 on 5 files.

---

### P1 — Branch carries PR #7's commits; needs rebase

`git log origin/main..HEAD` shows 21 commits but only the top 6 are Mialy's golden gate work. The bottom 15 are yours from what looks like PR #7's branch (confidence scoring, mutations, FPbase, Phase-2 features). `gh pr diff` correctly shows only 1,593 lines of actual delta because GitHub does a merge-diff, but merging this as-is would pull in all the #7 commits too.

Rebase onto `main` after #7 merges, or cherry-pick the 6 golden gate commits onto a fresh branch from main.

---

### P2 — `junction_overhangs` can misreport in the fallback case

`src/assembler.py:775-780` builds `junctions` by appending each part's `right_oh`. But the *assembled sequence* at each internal junction actually contains the *next* part's `left_oh` (line 778: `assembled += left_oh + insert_body`).

When overhang chaining succeeds these are identical by construction. But when chaining fails and falls back to the given part order (line 760), `part[i].right_oh` may not equal `part[i+1].left_oh` — and `junction_overhangs` will report values that aren't actually in the sequence.

Low severity since the fallback already emits a warning, but it makes the diagnostics misleading exactly when you'd want them most. Consider building `junctions` from what actually goes into `assembled`.

---

### P2 — Duplicate left overhangs silently drop parts

`src/assembler.py:744`: `oh_map = {item[1]: item for item in excised}` is last-wins if two parts share a left overhang. The dropped part triggers the generic "Unmatched parts" warning at line 754, but the root cause (overhang collision) isn't surfaced.

Duplicate overhangs are a design error on the user's side, but a specific error like `"Parts {a!r} and {b!r} have the same left overhang {oh!r} — assembly is ambiguous"` would save debugging time.

---

### P2 — Circular wrap-around excision untested

`_excise_insert` at lines 606-608 handles the case where the insert spans the plasmid origin (`body_end < body_start` → concatenate tail + head). I don't see a test for it in `TestExciseInsert`. Worth adding since origin-spanning inserts do exist in real part libraries and the math there is subtle.

---

### What's good

- **Orientation-agnostic cut math.** The `min()`/`max()` trick on `cut_top`/`cut_bottom` (lines 604-605, 735-738) handles both Scenario A (standard FWD-left backbones) and Scenario B (Allen Institute REV-left backbones) without any strand-specific branching. Elegant.
- **Overhang-matching site selection** in `_excise_insert` is the right design. Real carrier vectors often have extra enzyme sites in the resistance cassette; picking by position would excise garbage. Picking by expected overhang finds the correct flanking sites.
- **`GoldenGateResult` dataclass** — good return contract. The `assembly_order` and `junction_overhangs` fields make the assembly auditable, which matters for a tool that's supposed to never generate DNA.
- **35 tests covering all 4 enzymes, both backbone orientations, multi-part chaining, and error paths.** `test_scenario_b_backbone_assembly` specifically exercises the Allen Institute vector orientation — nice.
- **Test fixture isolation** (`eef94e0`) — moving mock constructs out of `library/*.json` into `tests/fixtures/` is the right call. Don't want test data showing up in user-facing searches.

---

### Nits

- `src/assembler.py:618` — the `dropout_sequence` parameter is annotated `# noqa: ARG001 — kept for documentation`. If it's never going to be used, consider dropping it and documenting dropout behavior in the docstring instead. Unused params tend to confuse callers about whether they should pass something.

---

---

## Merge order & topology

**What Slack said:** "#10 is built directly on top of #9."
**What git says:** #10 branches from `b63a05d` (pre-main, carrying #7's commits). #9 branches from `347231f` (current main). They share zero commits beyond their common ancestor.

**Conflicts:** 8+ merge conflict markers across `app/app.py`, `app/system_prompt.md`, `evals/run_agent_evals.py`, `src/library.py`, `src/tools.py`.

**Recommended order:**

1. **Merge #9 first.** It's based on current main, tests pass, and the P1s are 3-line fixes (rename two keys, swap line order).
2. **After #7 merges** (or in parallel): rebase #10 onto main. The #7 commits will fall away, leaving the 6 golden gate commits.
3. **Resolve #10's conflicts with #9** during that rebase — mostly additive (both add blocks to `app.py`, both add to `tools.py`), should be mechanical.
4. **Verify the overhang key fix in #9 actually flows through** after both are merged — add one integration test that loads a BYOL circular insert with CSV overhang hints and asserts they reach `_excise_insert`.

The overhang key rename should happen in **#9**, not #10 — `overhang_l`/`overhang_r` is already the established name in #10's assembler docstring, tool schema, and app.py, so #9 is the odd one out.
