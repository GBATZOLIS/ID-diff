
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from SyntheticDataset import SyntheticDataModule
from SyntheticDataset import scatter_plot
from model_lightning import SdeGenerativeModel


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def main(argv):
    config = FLAGS.config
    data = SyntheticDataModule(config)
    data.setup()
    loader = data.train_dataloader()
    batch = next(iter(loader))
    scatter_plot(batch, save=True)

    model = SdeGenerativeModel(config)
    x = model.sample()
    scatter_plot(x, save=True)

if __name__ == "__main__":
  app.run(main)
