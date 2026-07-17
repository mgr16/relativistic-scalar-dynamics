# R3 deposit checklist — 3.3.0 paper draft

Prepared 2026-07-16. This checklist does not authorize a tag, push, DOI
registration, arXiv upload, or journal submission; Marco performs every
external action.

## Human metadata gate

1. Confirm the author name and supply the full affiliation.
2. Decide whether to include ORCID and email; provide exact values if yes.
3. Select and complete one variant from
   `paper/ACKNOWLEDGMENTS_VARIANTS.md`, including verified funding and compute
   acknowledgments, or confirm that acknowledgments are omitted.
4. Confirm repository license, archival license, conflicts/funding statements,
   and the public repository URL.

## Finalize the archival manuscript

5. Replace the fixed draft `\date` in `paper/main.tex` with the intended
   submission date.
6. Insert the confirmed byline metadata and acknowledgment text.
7. Regenerate numbers/macros/figures, compile with the pinned Tectonic bundle
   and `SOURCE_DATE_EPOCH`, inspect every PDF page, run all `--check` commands
   and the complete test suite, then regenerate the SHA-256 manifest.
8. Repeat the clean-clone reproduction drill and archive its successful
   transcript.

## External sequence — Marco only

9. Commit the reviewed release tree and create tag `v3.3.0-paper`.
10. Push the commit/tag and create the Zenodo release; record the DOI.
11. Replace `[REPOSITORY-URL-AT-DEPOSIT]` in Data availability with the DOI
    and public repository URL.
12. Recompile, repeat visual QA/checks, regenerate the manifest, and commit/tag
    whatever final archival update policy requires.
13. Submit to arXiv with primary category `gr-qc` and choose any cross-list only
    after author review.
14. Submit the same archival manuscript and required metadata to PRD.

## Pending decisions owned by Marco

- affiliation;
- ORCID/email inclusion and exact values;
- acknowledgments variant and filled names/resources/funding;
- release date and version bump policy for `pyproject.toml`;
- public repository URL and archival license;
- creation/push of `v3.3.0-paper`;
- Zenodo DOI, arXiv cross-list, and PRD submission.
