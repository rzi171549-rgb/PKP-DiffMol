# PKP-DiffMol

Official reproducibility release for PKP-DiffMol, a physicochemical knowledge-prompt encoding and quality-controlled latent diffusion framework for molecular property prediction.

## Overview

Molecular property prediction requires representations that combine fragment-level SMILES semantics with explicit physicochemical evidence. PKP-DiffMol addresses this need by integrating learned molecular sequence representations with numerical and semantic physicochemical knowledge.

PKP-DiffMol builds upon the pretrained SMI-EDITOR encoder and enhances it using numerical physicochemical priors, descriptor-derived chemical knowledge prompts, hierarchical physicochemical knowledge fusion, and quality-controlled conditional latent diffusion. The framework constructs a chemically enriched fused latent space and uses a Mahalanobis-distance quality gate to filter generated synthetic molecular latent representations.

This repository preserves the source-of-truth runtime implementations used for the reported seven-dataset MoleculeNet classification experiments under scaffold splitting.

## Key Features

- SMI-Editor-based fragment-level molecular representation encoder
- Numerical Physicochemical Prior Branch based on RDKit descriptors
- Descriptor-derived physicochemical knowledge prompt encoding
- Hierarchical physicochemical knowledge fusion
- Quality-controlled conditional latent diffusion with Mahalanobis-distance quality gate
- Final configurations and main-table results for seven MoleculeNet classification datasets

## Main Results

The main table results are provided in `results/main_table_results.csv`.

| Dataset | Seed | ROC-AUC (%) |
|---|---:|---:|
| BACE | 350 | 87.21 |
| BBBP | 53 | 78.38 |
| Tox21 | 0 | 78.52 |
| SIDER | 10 | 66.92 |
| MUV | 134 | 84.49 |
| ClinTox | 21 | 99.81 |
| ToxCast | 2 | 68.40 |
| Mean | - | 80.53 |

Across the seven MoleculeNet tasks, PKP-DiffMol achieves a mean ROC-AUC of 80.53%, improves over SMI-EDITOR by 2.73 percentage points, and obtains the best result on six of seven datasets, with large gains on BACE, MUV, and SIDER.

## Repository Structure

```text
PKP-DiffMol/
|-- README.md
|-- assets/
|-- configs/final/
|-- results/main_table_results.csv
|-- scripts/
|-- pkpdiffmol/runtime_legacy/
|-- third_party/
|-- environment.yml
`-- requirements.txt
```

`pkpdiffmol/runtime_legacy/` contains the source-of-truth implementations used for reproduction. `scripts/train.py` maps clean implementation keys in `configs/final/*.yaml` to `runtime_legacy` training entries. Historical filenames inside `runtime_legacy` are preserved for exact reproduction compatibility.

## Dependencies

PKP-DiffMol uses PyTorch, RDKit, transformers, scikit-learn, and the preserved SMI-Editor/fairseq runtime components under `pkpdiffmol/runtime_legacy/`. See `requirements.txt` and `environment.yml` for the recorded environment dependencies.

## Installation

```bash
conda create -n pkpdiffmol python=3.9
conda activate pkpdiffmol
pip install -r requirements.txt
```

Alternatively, if using the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate pkpdiffmol
```

Some legacy components depend on SMI-Editor/fairseq runtime code preserved under `pkpdiffmol/runtime_legacy/`.

## Data and Assets

This repository does not include large datasets, LMDB splits, pretrained weights, or final task checkpoints.

Required assets:

```text
assets/smi_editor.pt
assets/smi_dict_token.txt
assets/text_encoder_name_or_path/config.json
assets/text_encoder_name_or_path/pytorch_model.bin
assets/text_encoder_name_or_path/vocab.txt
```

Required data layout:

```text
data/<dataset>/train.lmdb
data/<dataset>/valid.lmdb
data/<dataset>/test.lmdb
```

The seven datasets are BACE, BBBP, MUV, SIDER, Tox21, ToxCast, and ClinTox.

Dataset sources:

- MoleculeNet official website: https://moleculenet.org/
- DeepChem MoleculeNet loaders: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
- DeepChem / MoleculeNet GitHub reference: https://github.com/deepchem/moleculenet

Users should download the raw datasets from MoleculeNet or DeepChem, then build or place the processed LMDB splits under the data layout shown above.

## Reproduction

Dry-run a final configuration without launching training:

```bash
python scripts/train.py --config configs/final/bace.yaml --dry-run
```

Show the main-table results:

```bash
python scripts/evaluate.py
```

Run full training after preparing the required assets and LMDB data:

```bash
python scripts/train.py --config configs/final/bace.yaml
```

Inspect all seven final commands:

```bash
bash scripts/reproduce_main_table.sh
```

Dry-run resolves the final config and prints the corresponding runtime command without launching training. Full training requires the external assets and LMDB splits listed above. Evaluate-only from final checkpoints is not provided because final task checkpoints are not included.

## Method Components

PKP-DiffMol consists of the following components:

- Fragment-level molecular representation encoding: a pretrained SMI-EDITOR encoder provides fragment-level SMILES representations used as the molecular backbone.
- Multi-perspective physicochemical knowledge encoding: RDKit descriptors are encoded as continuous numerical priors, and the same descriptor evidence is transformed into descriptor-derived physicochemical knowledge prompts using a Transformer-based text encoder.
- Hierarchical physicochemical knowledge fusion: numerical and semantic physicochemical representations are fused at the descriptor level, then integrated with fragment-level molecular representations to construct a chemically enriched fused latent representation.
- Quality-controlled conditional latent diffusion: a conditional latent diffusion module models and generates synthetic molecular latent representations, with timestep and label conditioning in the diffusion process.
- Downstream molecular property prediction: filtered real and synthetic fused latent representations support downstream MoleculeNet classification with ROC-AUC evaluation.

## Scope of This Release

This release is intended for reproducibility and code inspection of the reported PKP-DiffMol experiments.

Included:

- Method-related runtime code
- Final configs
- Main-table results
- Reproduction wrappers
- Dataset and asset preparation instructions
- Third-party license records

Excluded:

- LMDB datasets
- Pretrained model weights
- Final task checkpoints
- Raw logs
- Raw summaries
- Exploratory seed sweeps
- Failed tuning runs
- Server-specific paths

## Third-Party Acknowledgements

The root LICENSE applies to PKP-DiffMol-specific code. Third-party components preserved for reproducibility follow their original licenses under `third_party/`.

This release preserves SMI-Editor-related runtime code for reproducibility. The SMI-Editor code package includes an MIT License; see `third_party/smi-editor/LICENSE`.

This release includes fairseq-derived code under `pkpdiffmol/runtime_legacy/original_repo/fairseq`. The fairseq-derived code follows the MIT License; see `third_party/fairseq/LICENSE`.

RDKit and transformers are external dependencies and are not vendored. MoleculeNet and DeepChem are used as dataset sources; processed LMDB data are not included.

## Citation

If you use this repository, please cite:

```bibtex
@article{liu2026pkpdiffmol,
  title={PKP-DiffMol: A Physicochemical Knowledge-Prompt Encoding and Latent Diffusion Framework for Molecular Property Prediction},
  author={Liu, Ruizi and Yuan, Tongtong and Guo, Molin},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={To be updated upon publication}
}
```

Please also cite SMI-Editor:

```bibtex
@inproceedings{zheng2025smieditor,
  title={SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision},
  author={Zheng, Kangjie and Liang, Siyue and Yang, Junwei and Feng, Bin and Liu, Zequn and Ju, Wei and Xiao, Zhiping and Zhang, Ming},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
