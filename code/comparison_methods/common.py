from dataclasses import dataclass
from pathlib import Path
import numpy as np
from mf_train_baseline.mf_utils import assert_indices_match, load_split_block
from sklearn.preprocessing import StandardScaler

from mf_train_baseline.mf_train import FPCA, fpca_propagate_var_to_y


@dataclass
class MFDatasetBundle:
    root: Path
    wavelengths: np.ndarray
    idx_wavelength: np.ndarray
    hf: dict
    lf_paired: dict
    lf_unpaired: dict

def _block(root, name):
    out = {}
    for split in ("train", "val", "test"):
        x, y, t, idx = load_split_block(root, name, split)
        out[split] = {"x": x, "y": y, "t": t, "idx": idx}
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{name}/{split}: x/y count mismatch")
    return out

def load_bundle(root):
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(f"Dataset root does not exist: {root}")
    wl = np.load(root / "wavelengths.npy").astype(np.float32).reshape(-1)
    iw = np.load(root / "idx_wavelength.npy").astype(np.int64).reshape(-1)
    if wl.size != iw.size: raise ValueError("wavelength/index size mismatch")
    hf, lp, lu = _block(root, "hf"), _block(root, "lf_paired"), _block(root, "lf_unpaired")
    for s in ("train", "val", "test"):
        assert_indices_match(hf[s]["idx"], lp[s]["idx"], f"hf/{s}", f"lf_paired/{s}")
    return MFDatasetBundle(root, wl, iw, hf, lp, lu)



class HFPCATransform:
    """The submitted HF-train spectral scaler -> FPCA -> latent scaler protocol."""

    def __init__(
        self,
        var_ratio=0.999,
        max_dim=64,
        dim=0,
        ridge=1e-8,
        random_state=None,
    ):
        self.var_ratio = var_ratio
        self.max_dim = max_dim
        self.dim = dim
        self.ridge = ridge
        self.random_state = random_state

    def fit(self, y_hf):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True).fit(y_hf)
        y_scaled = self.scaler_y.transform(y_hf).astype(np.float32)
        self.fpca = FPCA(
            n_components=self.dim,
            var_ratio=self.var_ratio,
            max_components=self.max_dim,
            ridge=self.ridge,
            random_state=self.random_state,
        )
        z_hf = self.fpca.fit_transform(y_scaled).astype(np.float32)
        self.scaler_z = StandardScaler(with_mean=True, with_std=True).fit(z_hf)
        return self

    @property
    def effective_dim(self):
        return int(self.fpca.n_components_)

    def transform(self, y):
        y_scaled = self.scaler_y.transform(y).astype(np.float32)
        return self.scaler_z.transform(self.fpca.transform(y_scaled)).astype(np.float32)

    def inverse(self, z):
        z_unscaled = self.scaler_z.inverse_transform(z).astype(np.float32)
        y_scaled = self.fpca.inverse_transform(z_unscaled)
        return self.scaler_y.inverse_transform(y_scaled).astype(np.float32)

    def variance_to_y(self, var):
        var_unscaled = np.asarray(var) * np.square(self.scaler_z.scale_)[None, :]
        return fpca_propagate_var_to_y(var_unscaled, self.fpca, self.scaler_y)
