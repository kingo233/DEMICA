from torch.utils.data import Dataset

class StirlingDataset(Dataset):
    def __init__(self,config) -> None:
        super().__init__()