"""
print_hydra_cfg.py - A helper script for printing hydra configuration

Helps with checking correct overrides are being used by printing the whole
DictConfig as YAML.
"""


import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
