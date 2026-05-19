# Asset Policy

Large local assets are documented here and are not forcibly copied into this release draft.

## Required Assets

The checks below were performed against the read-only original repository at `D:\Python深度学习\smi-editor-main`. This release draft does not copy `smi_editor.pt`, `pytorch_model.bin`, or the full `text_encoder_name_or_path` directory.

| asset | exists locally | original local path | public upload recommendation | expected release placement when supplied |
|---|---:|---|---|---|
| `smi_editor.pt` | yes | `D:\Python深度学习\smi-editor-main\smi_editor.pt` | Do not commit to GitHub. Publish only as an external/release asset if redistribution rights and storage policy allow it. | `assets\smi_editor.pt` |
| `smi_dict_token.txt` | yes | `D:\Python深度学习\smi-editor-main\smi_dict_token.txt` | Small text asset; can be included only if license/redistribution terms allow it. | `assets\smi_dict_token.txt` |
| `text_encoder_name_or_path\config.json` | yes | `D:\Python深度学习\smi-editor-main\text_encoder_name_or_path\config.json` | Small metadata file; can be included only if license/redistribution terms allow it. | `assets\text_encoder_name_or_path\config.json` |
| `text_encoder_name_or_path\pytorch_model.bin` | yes | `D:\Python深度学习\smi-editor-main\text_encoder_name_or_path\pytorch_model.bin` | Do not commit to GitHub. Publish only as an external/release asset if redistribution rights and storage policy allow it. | `assets\text_encoder_name_or_path\pytorch_model.bin` |
| `text_encoder_name_or_path\vocab.txt` | yes | `D:\Python深度学习\smi-editor-main\text_encoder_name_or_path\vocab.txt` | Text vocabulary asset; can be included only if license/redistribution terms allow it. | `assets\text_encoder_name_or_path\vocab.txt` |

If these assets are not bundled directly, users should place them at the release-relative paths listed above before running configurations that require the SMI-Editor backbone or local text encoder.

## MoleculeNet LMDB Splits

The current release draft assumes materialized MoleculeNet LMDB splits. GitHub packaging should not directly upload the LMDB data files. A later release pass can choose between download scripts and scaffold split-generation scripts. The checks below only record local existence and do not copy LMDB files.

| dataset | `train.lmdb` | `valid.lmdb` | `test.lmdb` | original local data directory |
|---|---:|---:|---:|---|
| BACE | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\bace\data` |
| BBBP | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\bbbp\data` |
| MUV | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\muv\data` |
| SIDER | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\sider\data` |
| Tox21 | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\tox21\data` |
| ToxCast | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\toxcast\data` |
| ClinTox | yes | yes | yes | `D:\Python深度学习\smi-editor-main\tasks\clintox\data` |

## Dataset sources and preparation

This study uses seven public MoleculeNet datasets:

- BACE
- BBBP
- MUV
- SIDER
- Tox21
- ToxCast
- ClinTox

Original data sources:

- MoleculeNet official website: https://moleculenet.org/
- DeepChem MoleculeNet loaders: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
- DeepChem / MoleculeNet GitHub reference: https://github.com/deepchem/moleculenet

This release does not directly upload LMDB files. The LMDB files used by this project are materialized preprocessed `train` / `valid` / `test` splits, not the original public raw datasets. The GitHub repository keeps code, configs, final summaries, data-download notes, and reproducibility notes only.

Users should download the raw datasets from MoleculeNet or DeepChem, then generate or place the LMDB splits under the release data directory. In placeholder form, each dataset should use `data/<dataset>/train.lmdb`, `data/<dataset>/valid.lmdb`, and `data/<dataset>/test.lmdb`.

```text
data/
├── bace/
│   ├── train.lmdb
│   ├── valid.lmdb
│   └── test.lmdb
├── bbbp/
├── muv/
├── sider/
├── tox21/
├── toxcast/
└── clintox/
```

A formal release can choose one of two packaging paths:

- Provide `scripts/download_moleculenet.py` and `scripts/build_lmdb_splits.py`.
- Provide materialized LMDB splits through Zenodo or Figshare and cite the DOI from the GitHub README.

## Checkpoint Status

The seven final task checkpoints are currently unavailable in the evidence packages and mapped local original-repo locations. They were likely removed after cleanpt cleanup.

This release draft therefore targets training reproducibility first and does not currently promise direct checkpoint-based evaluate-only reproduction.
