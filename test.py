from datamodule.data_module import DataModule
from config import get_config
import time
cfg = get_config()  # however you load your config
dm = DataModule(cfg)
train_loader = dm.train_dataloader()

for i, batch in enumerate(train_loader):
    start = time.time()
    print(f"Batch {i} loaded with keys: {list(batch.keys())}")
    if i >= 2:
        break
    print(f"Loading sample {i} took {time.time() - start:.2f} seconds")
