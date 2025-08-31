# utils/model_loader.py
import datetime as _dt
import pandas as pd
from omegaconf import OmegaConf

def load_yaml_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return OmegaConf.create(f.read())

# ---- sklearn compatibility shim (old pickles needing SimpleImputer.verbose) ----
def patch_imputer_verbose_class():
    try:
        from sklearn.impute import SimpleImputer
        if not hasattr(SimpleImputer, 'verbose'):
            SimpleImputer.verbose = 0
    except Exception:
        pass

def _patch_imputer_verbose_instance(obj):
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        return
    if isinstance(obj, SimpleImputer) and not hasattr(obj, 'verbose'):
        obj.verbose = 0
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except Exception:
        Pipeline = ColumnTransformer = None
    if Pipeline and isinstance(obj, Pipeline):
        for _, step in getattr(obj, 'steps', []):
            _patch_imputer_verbose_instance(step)
    if ColumnTransformer and isinstance(obj, ColumnTransformer):
        for _, transformer, _ in getattr(obj, 'transformers', []):
            if transformer not in (None, 'drop'):
                _patch_imputer_verbose_instance(transformer)
    if isinstance(obj, (list, tuple)):
        for it in obj:
            _patch_imputer_verbose_instance(it)
    elif hasattr(obj, '__dict__'):
        for v in obj.__dict__.values():
            _patch_imputer_verbose_instance(v)

# ---- preprocessing like in training ----
_NUMERIC_LIKE = {
    'rooms','bedroom2','bathroom','car','distance',
    'land_size','building_area','year_built','property_count',
    'latitude','longitude'
}

def preprocess_like_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'postcode' in df.columns:
        df['postcode'] = df['postcode'].astype(str)
    for c in _NUMERIC_LIKE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'property_age' not in df.columns and 'year_built' in df.columns:
        if 'date' in df.columns:
            year_from_date = df['date'].dt.year
        else:
            year_from_date = pd.Series(_dt.datetime.now().year, index=df.index)
        df['property_age'] = pd.to_numeric(year_from_date, errors='coerce') - pd.to_numeric(df['year_built'], errors='coerce')
    return df

# ---- model wrappers ----
class LocalPyCaretModel:
    def __init__(self, model_stem: str):
        from pycaret.regression import load_model
        patch_imputer_verbose_class()
        self.model = load_model(model_stem)
        _patch_imputer_verbose_instance(self.model)
    def predict(self, df: pd.DataFrame):
        from pycaret.regression import predict_model
        out = predict_model(self.model, data=df)
        if 'Label' in out.columns:
            return out['Label']
        return out.iloc[:, -1]

class MLflowModel:
    def __init__(self, tracking_uri: str, run_id: str):
        import mlflow
        from mlflow.pyfunc import load_model as mlflow_load_model
        patch_imputer_verbose_class()
        mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow_load_model(f'runs:/{run_id}/model')
    def predict(self, df: pd.DataFrame):
        preds = self.model.predict(df)
        try:
            import pandas as pd
            return pd.Series(preds)
        except Exception:
            return preds

def load_model_from_cfg(cfg, override_use_mlflow=None, override_run_id=None):
    use_mlflow = bool(override_use_mlflow) if override_use_mlflow is not None else bool(cfg.backend.use_mlflow)
    if use_mlflow:
        run_id = override_run_id if override_run_id else str(cfg.mlflow.run_id or '')
        if not run_id:
            raise ValueError('USE_MLFLOW is True but no run_id provided.')
        return MLflowModel(str(cfg.mlflow.tracking_uri), run_id)
    return LocalPyCaretModel(str(cfg.backend.model_stem))
