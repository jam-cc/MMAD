from .mvtec import MVTecDataset, DatasetSplit  # NOCA:unused-import(used)


class BTADDataset(MVTecDataset):
    def __init__(self,
                 **kwargs):
        super(BTADDataset, self).__init__(**kwargs)

    def set_normal_class(self):
        self.normal_class = "ok"
