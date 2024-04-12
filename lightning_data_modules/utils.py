import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split


_LIGHTNING_DATA_MODULES = {}
def register_lightning_datamodule(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _LIGHTNING_DATA_MODULES:
      raise ValueError(f'Already registered model with name: {local_name}')
    _LIGHTNING_DATA_MODULES[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_lightning_datamodule_by_name(name):
  return _LIGHTNING_DATA_MODULES[name]

def create_lightning_datamodule(config):
  datamodule = get_lightning_datamodule_by_name(config.data.datamodule)(config)
  return datamodule

