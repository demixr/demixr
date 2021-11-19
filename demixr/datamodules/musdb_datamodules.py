from pytorch_lightning import LightningDataModule

from pathlib import Path


class MusdbDataModule(LightningDataModule):
    """
    LightningDataModule for Musdb18-HQ dataset

    Load parameters from configs/datamodule/musdb_datamodule.yml
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)

        # Dataloader part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # audio part
        self.sample_rate = sample_rate
