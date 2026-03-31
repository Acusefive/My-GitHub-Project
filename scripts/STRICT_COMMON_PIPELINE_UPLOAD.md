# Strict Common Pipeline Upload Guide

This document lists what should be committed for the strict common pipeline,
and what should stay local.

## Commit These Files

Core entrypoints:

- `scripts/run_common_cognitive_pipeline_strict.py`
- `scripts/validate_common_pipeline_strict.py`

Core package:

- `scripts/common_pipeline_strict/__init__.py`
- `scripts/common_pipeline_strict/constants.py`
- `scripts/common_pipeline_strict/io_utils.py`
- `scripts/common_pipeline_strict/models.py`
- `scripts/common_pipeline_strict/stage32.py`
- `scripts/common_pipeline_strict/stage34.py`
- `scripts/common_pipeline_strict/validation.py`

Support files:

- `requirements-strict-common-pipeline.txt`
- `scripts/STRICT_COMMON_PIPELINE_UPLOAD.md`

## Do Not Commit

Local data and generated outputs:

- `datalocal/`
- `cachelocal/`
- `out/`
- `__pycache__/`

Generated artifacts:

- `*.pkl`
- `*.npy`
- `*.npz`
- `*.pt`
- `*.pth`

Temporary review and extraction files:

- extracted txt/docx copies
- preview outputs
- local audit scratch files unless they are intended as project docs

## Existing Files Not Required For This Upload

These are not part of the strict pipeline implementation itself:

- `scripts/preprocess_embeddings_cognitive_rag.py`
- `AKT-master/main.py`
- baseline-specific code

Keep them out of the strict-pipeline commit unless you intentionally changed
them for a separate reason.

## Recommended Commit Scope

If you want one clean commit for this work, stage only:

```bash
git add .gitignore
git add requirements-strict-common-pipeline.txt
git add scripts/run_common_cognitive_pipeline_strict.py
git add scripts/validate_common_pipeline_strict.py
git add scripts/STRICT_COMMON_PIPELINE_UPLOAD.md
git add scripts/common_pipeline_strict
```

## Smoke Commands Before Push

```bash
python scripts/run_common_cognitive_pipeline_strict.py --smoke --dry_run --preview_limit 10
python scripts/validate_common_pipeline_strict.py --smoke
```

## Full Run Commands

```bash
python scripts/run_common_cognitive_pipeline_strict.py
python scripts/validate_common_pipeline_strict.py
```
