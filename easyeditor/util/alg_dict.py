from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor, MendPerRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..dataset import ZsreDataset, CounterFactDataset, CaptionDataset, VQADataset, PersonalityDataset, SafetyDataset
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'GRACE': apply_grace_to_model,
    'WISE': apply_wise_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
}

ALG_MULTIMODAL_DICT = {
    'MEND': MendMultimodalRewriteExecutor().apply_to_model,
}

PER_ALG_DICT = {
    "MEND": MendPerRewriteExecutor().apply_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}

PER_DS_DICT = {
    "personalityEdit": PersonalityDataset
}
Safety_DS_DICT ={
    "safeEdit": SafetyDataset
}