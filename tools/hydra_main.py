# tools/hydra_main.py
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(config_path="../app_config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded config:\n", OmegaConf.to_yaml(cfg))

    # produce a tiny artifact to show config-driven run
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame([{
        "use_mlflow": bool(cfg.backend.use_mlflow),
        "model_stem": str(cfg.backend.model_stem),
        "tracking_uri": str(cfg.mlflow.tracking_uri),
        "run_id": str(cfg.mlflow.run_id),
    }]).to_csv(out_dir / "hydra_sample_payload.csv", index=False)
    print("Wrote artifacts/hydra_sample_payload.csv")

if __name__ == "__main__":
    main()