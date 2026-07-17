# Acknowledgments variants for Marco

These are release-time choices, not manuscript metadata inferred by the
pipeline. Replace every bracketed token with verified information or delete
the corresponding sentence. Marco should select one variant and confirm the
wording against the journal policy in force at submission.

## Variant A — without AI-assistance disclosure

```latex
\begin{acknowledgments}
The author thanks [NAMES] for [SPECIFIC CONTRIBUTIONS].
Computational resources were provided by [RESOURCE OR INSTITUTION].
This work was supported by [FUNDER AND GRANT NUMBER].
\end{acknowledgments}
```

## Variant B — with AI-assistance disclosure

```latex
\begin{acknowledgments}
The author thanks [NAMES] for [SPECIFIC CONTRIBUTIONS].
Computational resources were provided by [RESOURCE OR INSTITUTION].
This work was supported by [FUNDER AND GRANT NUMBER].
OpenAI Codex was used as a software and editorial assistant for code
generation, test construction, literature triage, and language revision.
All scientific choices, source verification, and responsibility for the
manuscript remain with the author.
\end{acknowledgments}
```

If there are no named contributors, dedicated compute allocation, or funding,
delete those sentences rather than inserting generic claims.
