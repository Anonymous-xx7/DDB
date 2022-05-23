from . import CustomDataset, CityscapesDataset, DATASETS


@DATASETS.register_module()
class Synthia(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(Synthia, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
