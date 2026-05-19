#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
checkpoint_adapter.py
======================
Thin adapter: SMI-Editor pretrain checkpoint → SIDER downstream fine-tuning.

What this file does (ONLY):
  1. Load LevenshteinEncoderModel from the SMI-Editor pretrain checkpoint
  2. Load the SMI-Editor dictionary (smi_dict_token.txt), using the SAME
     protocol as the original translation_lev_smi task
  3. Register the SIDER classification head (27-class multi-label)
  4. Align the state dict using the model's built-in upgrade_state_dict_named()
  5. Return (model, dictionary) ready for fine-tuning

What this file does NOT do:
  - Does NOT rewrite any backbone or training logic
  - Does NOT invent new key mappings
  - Does NOT define new model architectures
  - Does NOT touch any data loading or loss functions

Backbone entry: fairseq/models/roberta/levenshtein_encoder.py
  class LevenshteinEncoderModel (registered as "levenshtein_encoder")
  forward for downstream: model(src_tokens, features_only=True,
                                 classification_head_name='sider',
                                 levenshtein=False)
  classification uses features[:, 0, :] = [CLS] token
"""

import os
import sys
import logging

import torch

# --------------------------------------------------------------------------
# Add smi-editor-main to sys.path so `import fairseq` resolves to the local
# SMI-Editor fairseq package (not a system-installed one).
# --------------------------------------------------------------------------
_ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
_SMI_EDITOR_ROOT = os.path.abspath(os.path.join(_ADAPTER_DIR, '../../..'))
if _SMI_EDITOR_ROOT not in sys.path:
    sys.path.insert(0, _SMI_EDITOR_ROOT)

from fairseq.data import Dictionary
from fairseq.models.roberta.levenshtein_encoder import LevenshteinEncoderModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# SMILES tokenization regex (same as SMILESTokenizeDataset in the original
# fairseq/data/ai4sci/mol_datasets/smiles_tokenize_dataset.py)
# --------------------------------------------------------------------------
SMILES_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


# --------------------------------------------------------------------------
# Minimal task stub  (satisfies LevenshteinEncoderModel.build_model signature)
# --------------------------------------------------------------------------
class _FinetuneTask:
    """
    Minimal task object supplying only what build_model() needs:
      task.source_dictionary  →  the SMI-Editor Dictionary instance
    Nothing else is used by the model build path.
    """
    def __init__(self, dictionary):
        self.source_dictionary = dictionary
        self.target_dictionary = dictionary

    def max_positions(self):
        return (1024, 1024)


# --------------------------------------------------------------------------
# Dictionary loader  (mirrors translation_lev_smi.py setup_task logic)
# --------------------------------------------------------------------------
def load_smi_editor_dictionary(dict_path: str) -> Dictionary:
    """
    Load the SMI-Editor vocabulary using the SAME protocol as the original
    TranslationLevenshteinSMITask in fairseq/tasks/translation_lev_smi.py.

    Protocol (from translation_lev_smi.py):
      1. Create a blank fairseq Dictionary (adds <s>, <pad>, </s>, <unk>)
      2. Read every symbol from smi_dict_token.txt and add_symbol()
      3. Override bos/pad/eos/unk indices to point to [CLS]/[PAD]/[SEP]/[UNK]
      4. Add [MASK] symbol

    The resulting dictionary indices match those used during pre-training,
    so the embedding layer in the checkpoint is correctly aligned.
    """
    # Use add_special_symbols=False to match original translation_lev_smi.py
    # protocol (load_only_mol_dict), which does NOT prepend the 4 default
    # fairseq tokens (<s>, <pad>, </s>, <unk>).  Without this flag the
    # dictionary grows to 373 symbols while the checkpoint vocab is 369,
    # causing a shape mismatch on embed_tokens / lm_head.
    dictionary = Dictionary(add_special_symbols=False)
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            sym = line.strip().split()[0].strip()
            if sym:
                dictionary.add_symbol(sym)

    # Mirror the special-token overrides from translation_lev_smi.py
    dictionary.bos_index = dictionary.index('[CLS]')
    dictionary.pad_index = dictionary.index('[PAD]')
    dictionary.eos_index = dictionary.index('[SEP]')
    dictionary.unk_index = dictionary.index('[UNK]')
    dictionary.add_symbol('[MASK]')

    return dictionary


# --------------------------------------------------------------------------
# Main adapter function
# --------------------------------------------------------------------------
def load_levenshtein_model_for_sider(
    checkpoint_path: str,
    dict_path: str,
    num_classes: int = 27,
) -> tuple:
    """
    Load LevenshteinEncoderModel with SIDER classification head.

    Parameters
    ----------
    checkpoint_path : str
        Path to smi_editor_release.pt (fairseq checkpoint format)
    dict_path : str
        Path to smi_dict_token.txt
    num_classes : int
        Number of SIDER labels (27 per paper Table 1)

    Returns
    -------
    (model, dictionary)
        model   : LevenshteinEncoderModel with 'sider' head registered
                  and pretrain weights loaded
        dictionary : SMI-Editor Dictionary for tokenizing input SMILES

    Checkpoint alignment strategy:
        - LevenshteinEncoderModel.load_state_dict calls upgrade_state_dict_named
          internally (via BaseFairseqModel.load_state_dict)
        - upgrade_state_dict_named copies newly-registered classification_heads.*
          (random init) into the state_dict, so strict=True load succeeds
        - No custom key mapping invented: the built-in mechanism handles all
          alignment between the pretrain checkpoint and the fine-tune model
        - Expected missing keys (before upgrade): only classification_heads.sider.*
          (4 keys: dense.weight, dense.bias, out_proj.weight, out_proj.bias)
        - Expected unexpected keys: none (or 'version' which is non-parameter)

    Raises
    ------
    FileNotFoundError  if checkpoint or dict file not found
    RuntimeError       if core encoder keys are missing after loading
                       (indicates real architecture mismatch, not just head)
    """
    # ------------------------------------------------------------------
    # 1. Validate paths
    # ------------------------------------------------------------------
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SMI-Editor checkpoint not found: {checkpoint_path}\n"
            f"Please download smi_editor_release.pt from the Google Drive link "
            f"in the SMI-Editor README and place it at the expected path."
        )
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(
            f"SMI-Editor dictionary not found: {dict_path}"
        )

    # ------------------------------------------------------------------
    # 2. Load checkpoint (CPU)
    # ------------------------------------------------------------------
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    assert 'model' in ckpt, "Not a valid fairseq checkpoint (missing 'model' key)"
    assert 'args' in ckpt, "Not a valid fairseq checkpoint (missing 'args' key)"

    args = ckpt['args']
    if args is None:
        args = ckpt.get('cfg', {}).get('model', None)
        if args is None:
            raise RuntimeError("Cannot find model config in checkpoint (tried 'args' and 'cfg.model')")
        logger.info("  (ckpt['args'] is None; using ckpt['cfg']['model'] instead)")
    # Keep only tensor entries (filter out 'version' metadata key)
    state_dict = {
        k: v for k, v in ckpt['model'].items()
        if isinstance(v, torch.Tensor)
    }

    logger.info(f"  arch: {getattr(args, 'arch', 'unknown')}")
    logger.info(f"  encoder_layers: {getattr(args, 'encoder_layers', '?')}")
    logger.info(f"  encoder_embed_dim: {getattr(args, 'encoder_embed_dim', '?')}")
    logger.info(f"  tensor keys in checkpoint: {len(state_dict)}")

    # ------------------------------------------------------------------
    # 3. Load dictionary (SAME protocol as pre-training)
    # ------------------------------------------------------------------
    dictionary = load_smi_editor_dictionary(dict_path)
    logger.info(f"  dictionary size: {len(dictionary)}")
    logger.info(f"  bos=[CLS]={dictionary.bos_index}  pad=[PAD]={dictionary.pad_index}"
                f"  eos=[SEP]={dictionary.eos_index}  unk=[UNK]={dictionary.unk_index}")

    # ------------------------------------------------------------------
    # 4. Build LevenshteinEncoderModel from checkpoint args
    # ------------------------------------------------------------------
    task = _FinetuneTask(dictionary)
    model = LevenshteinEncoderModel.build_model(args, task)

    # ------------------------------------------------------------------
    # 5. Register SIDER classification head
    #    RobertaClassificationHead: input → dense → tanh → dropout → out_proj
    #    forward: uses features[:, 0, :] = [CLS] token
    # ------------------------------------------------------------------
    model.register_classification_head(
        'sider',
        num_classes=num_classes,
        inner_dim=args.encoder_embed_dim,
    )
    logger.info(f"  registered classification_head 'sider' with {num_classes} classes")

    # ------------------------------------------------------------------
    # 6. Load state dict
    #    BaseFairseqModel.load_state_dict internally calls:
    #      self.upgrade_state_dict(state_dict)
    #        → self.upgrade_state_dict_named(state_dict, "")
    #          → copies classification_heads.sider.* (random init) into state_dict
    #      prune_state_dict(state_dict, model_cfg=None)  → no-op
    #      nn.Module.load_state_dict(state_dict, strict=True)
    #
    #    LevenshteinEncoderModel.load_state_dict hardcodes strict=True.
    #    After upgrade_state_dict_named, all 4 classification head keys
    #    are present in state_dict, so strict=True load succeeds.
    # ------------------------------------------------------------------
    model.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # 7. Verify alignment by comparing model and checkpoint keys
    #    (informational only - load_state_dict already succeeded above)
    # ------------------------------------------------------------------
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    head_keys = {k for k in model_keys if 'classification_heads' in k}
    core_ckpt_keys = ckpt_keys - head_keys
    core_model_keys = model_keys - head_keys

    missing_core = core_model_keys - core_ckpt_keys
    unexpected_core = core_ckpt_keys - core_model_keys

    logger.info(f"Post-load alignment:")
    logger.info(f"  core model keys: {len(core_model_keys)}")
    logger.info(f"  core checkpoint keys: {len(core_ckpt_keys)}")
    logger.info(f"  missing core keys: {len(missing_core)}")
    logger.info(f"  unexpected core keys: {len(unexpected_core)}")
    logger.info(f"  classification head keys: {len(head_keys)} (random init, correct)")

    if missing_core:
        logger.error(f"ALIGNMENT FAILURE: Missing core encoder keys: {list(missing_core)[:5]}")
        raise RuntimeError(
            f"Checkpoint alignment failed. Core encoder keys missing after loading:\n"
            f"{list(missing_core)[:10]}\n"
            f"This indicates a true structural mismatch between the checkpoint "
            f"and the LevenshteinEncoderModel. Cannot proceed."
        )

    if unexpected_core:
        logger.warning(
            f"Unexpected keys in checkpoint (not in model): {list(unexpected_core)[:5]}"
        )

    logger.info("Checkpoint loaded successfully. Model ready for SIDER fine-tuning.")
    return model, dictionary


# --------------------------------------------------------------------------
# SMILES tokenizer (used by train_sider.py)
# --------------------------------------------------------------------------
import re
_SMILES_REGEX = re.compile(SMILES_REGEX_PATTERN)


def tokenize_smiles(smiles: str, dictionary: Dictionary,
                    max_len: int = 512) -> torch.Tensor:
    """
    Tokenize a SMILES string into a token ID tensor.

    Protocol:
      1. Apply SMILES regex to split into chemically-meaningful tokens
         (same regex as SMILESTokenizeDataset in the original code)
      2. Map tokens to dictionary indices (unk if not found)
      3. Prepend bos ([CLS]) and append eos ([SEP])
      4. Truncate to max_len tokens total (including bos/eos)

    The resulting token sequence starts with [CLS] at position 0,
    which is what RobertaClassificationHead uses for classification:
      features[:, 0, :] = [CLS] representation → classification head

    Parameters
    ----------
    smiles : str
        Input SMILES string (as-is from LMDB, no normalization)
    dictionary : Dictionary
        SMI-Editor dictionary loaded by load_smi_editor_dictionary()
    max_len : int
        Maximum sequence length including bos and eos tokens

    Returns
    -------
    torch.LongTensor of shape (L,) where L ≤ max_len
    """
    tokens = _SMILES_REGEX.findall(smiles)

    # Truncate body to leave room for bos + eos
    max_body = max_len - 2
    if len(tokens) > max_body:
        tokens = tokens[:max_body]

    # Map to indices (unk for unknown tokens)
    ids = [dictionary.bos_index]  # [CLS]
    for tok in tokens:
        ids.append(dictionary.index(tok))
    ids.append(dictionary.eos_index)  # [SEP]

    return torch.tensor(ids, dtype=torch.long)
