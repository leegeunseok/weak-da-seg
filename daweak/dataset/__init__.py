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
    'generalcon_t': GeneralConTSegmentation
}


def get_dataset(name, **kwargs):
    return datasets[name.lower()](name, **kwargs)
