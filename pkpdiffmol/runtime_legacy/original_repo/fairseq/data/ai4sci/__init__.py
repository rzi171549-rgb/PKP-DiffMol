from .mol_datasets import (
    LMDBDataset,
    KeyDataset,
    TokenizeDataset,
    SMILESDropNewCombDataset,
    SMILESTokenizeDataset,
)

from .protein_datasets import (
    KeyTokenizeDataset
)

# Import Uni-Mol 3D dataset classes from unimol.data
# These are needed by fairseq/data/__init__.py but only used for 3D pre-training
import sys
import os
_unimol_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Uni-Mol-main/unimol'))
if _unimol_path not in sys.path:
    sys.path.insert(0, _unimol_path)

from unimol.data import (
    Add2DConformerDataset,
    ConformerSampleDataset,
    AtomTypeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    MaskPointsDataset,
    EdgeTypeDataset,
    DistanceDataset,
    RightPadDatasetCoord,
    RightPadDatasetCross2D,
)

from unicore.data import (
    FromNumpyDataset,
    RightPadDataset2D,
)

__all__ = []
