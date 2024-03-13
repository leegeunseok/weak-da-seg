###########################################################################
# Created by: Yi-Hsuan Tsai, NEC Labs America, 2019
###########################################################################

from .gta5 import GTA5Segmentation
from .cityscapes import CityscapesSegmentation
from .synthia import SYNTHIADataSet
from .gaps_s import GAPSSSegmentation
from .gaps_t import GAPSTSegmentation
from .crack500_s import CRACK500SSegmentation
from .crack500_t import CRACK500TSegmentation
from .rissbilder_s import RissbilderSSegmentation
from .rissbilder_t import RissbilderTSegmentation
from .road_s import RoadSSegmentation
from .road_t import RoadTSegmentation
from .generalcon_s import GeneralConSSegmentation
from .generalcon_t import GeneralConTSegmentation
from .express_s import ExpressSSegmentation
from .express_t import ExpressTSegmentation
from .deepcrack_s import DeepCrackSSegmentation
from .deepcrack_t import DeepCrackTSegmentation
from .deepnon_s import DeepNonSSegmentation
from .deepnon_t import DeepNonTSegmentation
from .kaggle_s import KaggleSSegmentation
from .kaggle_t import KaggleTSegmentation
from .yang_s import YangSSegmentation
from .yang_t import YangTSegmentation
from .cfd_s import CFDSSegmentation
from .cfd_t import CFDTSegmentation
from .crack500non_s import CRACK500NonSSegmentation
from .crack500non_t import CRACK500NonTSegmentation
from .crack500_305_s import CRACK500305SSegmentation
from .crack500_305_t import CRACK500305TSegmentation
from .crack500_605_s import CRACK500605SSegmentation
from .crack500_605_t import CRACK500605TSegmentation
from .crack500_905_s import CRACK500905SSegmentation
from .crack500_905_t import CRACK500905TSegmentation

datasets = {
    'gta5': GTA5Segmentation,
    'cityscapes': CityscapesSegmentation,
    'synthia': SYNTHIADataSet,
    'gaps_s': GAPSSSegmentation,
    'gaps_t': GAPSTSegmentation,
    'crack500_s': CRACK500SSegmentation,
    'crack500_t': CRACK500TSegmentation,
    'rissbilder_s': RissbilderSSegmentation,
    'rissbilder_t': RissbilderTSegmentation,
    'road_s': RoadSSegmentation,
    'road_t': RoadTSegmentation,
    'generalcon_s': GeneralConSSegmentation,
    'generalcon_t': GeneralConTSegmentation,
    'express_s': ExpressSSegmentation,
    'express_t': ExpressTSegmentation,
    'deepcrack_s': DeepCrackSSegmentation,
    'deepcrack_t': DeepCrackTSegmentation,
    'deepnon_s': DeepNonSSegmentation,
    'deepnon_t': DeepNonTSegmentation,
    'kaggle_s': KaggleSSegmentation,
    'kaggle_t': KaggleTSegmentation,
    'yang_s': YangSSegmentation,
    'yang_t': YangTSegmentation,
    'cfd_s': CFDSSegmentation,
    'cfd_t': CFDTSegmentation,
    'crack500non_s': CRACK500NonSSegmentation,
    'crack500non_t': CRACK500NonTSegmentation,
    'crack500_305_s': CRACK500305SSegmentation,
    'crack500_305_t': CRACK500305TSegmentation,
    'crack500_605_s': CRACK500605SSegmentation,
    'crack500_605_t': CRACK500605TSegmentation,
    'crack500_905_s': CRACK500905SSegmentation,
    'crack500_905_t': CRACK500905TSegmentation,
}


def get_dataset(name, **kwargs):
    return datasets[name.lower()](name, **kwargs)
