from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset  # 确保你导入了 BaseSegDataset 这个具体的基类

@DATASETS.register_module()
class MyCustomDataset(BaseSegDataset):
    # 定义类别和颜色调色板
    CLASSES = ('background', 'cell')  # 根据你的数据集调整类别
    PALETTE = [[0, 0, 0], [255, 255, 255]]  # 定义每个类别的颜色（用于可视化）

    def __init__(self, **kwargs):
        # 调用父类构造函数，并传递必要的参数
        super(MyCustomDataset, self).__init__(img_suffix='.tif', seg_map_suffix='.png', **kwargs)
