## Flows4Ad

Repository for Anomaly Detection with Normalizing Flows.

### Data preparation

Clone ADBench repository (somewhere):
```bash
git clone https://github.com/Minqi824/ADBench
```

Data is contained in `datasets` directory. Make a symbolic link to the root 
of the current project:
```bash
ln -s <ADBench_dir/datasets> datasets
```

Datasets are expected to be in `.npz` format with `X` and `y` keys.

### Experiment running 

Look at the example in `run_main.py`. In order to change some parameters of experiment, please refer to the `.yaml` config file.

For RealNVP it is necessary to choose either `channel_wise` or `checkerboard` split or both.
