#from .lenet import *
#from .vggnet import *
#from .resnet import *
from .wide_resnet import *
from .wide_resnet_batchensemble import *
#from .vggnet_ensemble import *
#from .wide_resnet_ensemble2 import *
#from .bayesianwide_resnet_ensemble import *
#from .wide_resnet_ensemble2_div import *
#from .wide_resnet_ensemble2_latent import *

from .wide_resnet_ensemble_LPBNN import *

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
