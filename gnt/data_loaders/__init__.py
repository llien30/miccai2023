from .deepvoxels import *
from .google_scanned_objects import *
from .ibrnet_collected import *
from .llff import *
from .llff_med import *
from .llff_render import *
from .llff_test import *
from .nerf_synthetic import *
from .nerf_synthetic_render import *
from .nmr_dataset import *
from .realestate import *
from .shiny import *
from .shiny_render import *
from .spaces_dataset import *

dataset_dict = {
    "spaces": SpacesFreeDataset,
    "google_scanned": GoogleScannedDataset,
    "realestate": RealEstateDataset,
    "deepvoxels": DeepVoxelsDataset,
    "nerf_synthetic": NerfSyntheticDataset,
    "llff": LLFFDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "llff_test": LLFFTestDataset,
    "shiny": ShinyDataset,
    "llff_render": LLFFRenderDataset,
    "shiny_render": ShinyRenderDataset,
    "nerf_synthetic_render": NerfSyntheticRenderDataset,
    "nmr": NMRDataset,
    "llff_med": LLFFMedDataset,
}
