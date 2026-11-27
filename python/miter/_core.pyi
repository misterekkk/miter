from enum import Enum

import numpy as np
from numpy.typing import NDArray

def hello() -> str: ...
def cpp_version() -> str: ...
def has_openmp() -> bool: ...
def openmp_max_threads() -> int: ...
def lignum_linked() -> bool: ...
def lignum_smoke_hash(text: str) -> int: ...
def array_shape(arr: NDArray[np.float64]) -> tuple[int, int]: ...

class Objective(Enum):
    Regression = ...
    Binary = ...
    Multiclass = ...

class GrowPolicy(Enum):
    Leafwise = ...
    Symmetric = ...

CTR_COMPLEXITY_AUTO: int

class QuantizerMode(Enum):
    Linear = ...
    Quantile = ...
    Auto = ...

class Params:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    grow_policy: GrowPolicy
    min_data_in_leaf: int
    min_sum_hessian_in_leaf: float
    lambda_l2: float
    min_gain_to_split: float
    max_bin: int
    max_cat_to_onehot: int
    min_data_per_group: int
    max_categories: int
    cat_smooth: float
    cat_l2: float
    max_cat_threshold: int
    m_mult: int
    k_proj: int
    oblique_max_depth: int
    early_stopping_rounds: int
    time_limit: float
    n_jobs: int
    random_state: int
    feature_fraction: float
    bagging_fraction: float
    extra_trees: bool
    oblique_min_distinct: int
    oblique_sample_rows: int
    efb: bool
    oblique_binary_penalty: float
    oblique_gain_margin: float
    max_ctr_complexity: int
    ctr_prior_weight: float
    ctr_n_permutations: int
    ctr_max_borders: int
    ctr_model_size_reg: float
    quantizer_mode: QuantizerMode
    def __init__(self) -> None: ...

class Model:
    miter_version: str
    objective: Objective
    init_score: float
    n_classes: int
    @property
    def n_ctr_tables(self) -> int: ...
    @property
    def n_ctr_projections(self) -> int: ...
    @property
    def n_quantile_features(self) -> int: ...
    @property
    def init_scores(self) -> NDArray[np.float64]: ...
    learning_rate: float
    n_features: int
    best_iteration: int
    params: Params
    feature_names: list[str]
    @property
    def is_categorical(self) -> list[bool]: ...
    def feature_importances_split(self) -> NDArray[np.uint64]: ...
    def feature_importances_gain(self) -> NDArray[np.float64]: ...

class TrainResult:
    model: Model
    train_loss: list[float]
    valid_loss: list[float]
    def final_train_scores(self) -> NDArray[np.float64]: ...

class TransformedData:
    n_rows: int
    n_features: int
    n_num_features: int
    def codes(self) -> NDArray[np.int16]: ...
    def unsigned_codes(self) -> NDArray[np.uint16]: ...
    def bins(self) -> NDArray[np.uint8]: ...

class Dataset:
    @property
    def n_rows(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_num_features(self) -> int: ...
    @property
    def n_cat_features(self) -> int: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def is_categorical(self) -> list[bool]: ...
    def codes(self) -> NDArray[np.int16]: ...
    def unsigned_codes(self) -> NDArray[np.uint16]: ...
    def bins(self) -> NDArray[np.uint8]: ...
    def quantizer_params(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]: ...
    def bin_edges(self, feature_idx: int) -> NDArray[np.uint16]: ...
    def num_bins_for_feature(self, feature_idx: int) -> int: ...
    def cat_categories(self, feature_idx: int) -> list[str]: ...
    def cat_other_code(self, feature_idx: int) -> int: ...
    def cat_missing_code(self, feature_idx: int) -> int: ...
    def transform_numeric_f64(self, arr: NDArray[np.float64], n_jobs: int = 1) -> TransformedData: ...
    def transform_numeric_f32(self, arr: NDArray[np.float32], n_jobs: int = 1) -> TransformedData: ...
    def transform_mixed(
        self,
        n_rows: int,
        num_cols: list[NDArray[np.float64] | NDArray[np.float32]],
        cat_vocabs: list[list[str]],
        cat_indices: list[NDArray[np.int32]],
        n_jobs: int = 1,
    ) -> TransformedData: ...

def fit_numeric_f64(
    arr: NDArray[np.float64],
    max_bin: int = 255,
    feature_names: list[str] = ...,
    n_jobs: int = 1,
    quantizer_mode: QuantizerMode = ...,
) -> Dataset: ...
def fit_numeric_f32(
    arr: NDArray[np.float32],
    max_bin: int = 255,
    feature_names: list[str] = ...,
    n_jobs: int = 1,
    quantizer_mode: QuantizerMode = ...,
) -> Dataset: ...
def fit_mixed(
    n_rows: int,
    is_cat: list[bool],
    feature_names: list[str],
    num_cols: list[NDArray[np.float64] | NDArray[np.float32]],
    cat_vocabs: list[list[str]],
    cat_indices: list[NDArray[np.int32]],
    max_bin: int = 255,
    min_data_per_group: int = 100,
    max_categories: int = 253,
    n_jobs: int = 1,
    quantizer_mode: QuantizerMode = ...,
) -> Dataset: ...
def fit_booster(
    ds: Dataset,
    objective: Objective,
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
    valid_ds: TransformedData | None = None,
    y_valid: NDArray[np.float64] | None = None,
    valid_sample_weight: NDArray[np.float64] | None = None,
    params: Params = ...,
) -> TrainResult: ...
def fit_axis_booster(
    ds: Dataset,
    objective: Objective,
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
    valid_ds: TransformedData | None = None,
    y_valid: NDArray[np.float64] | None = None,
    valid_sample_weight: NDArray[np.float64] | None = None,
    params: Params = ...,
) -> TrainResult: ...
def predict_raw_f64(
    model: Model, arr: NDArray[np.float64], num_iteration: int = 0
) -> NDArray[np.float64]: ...
def predict_raw_f32(
    model: Model, arr: NDArray[np.float32], num_iteration: int = 0
) -> NDArray[np.float64]: ...
def predict_proba_f64(
    model: Model, arr: NDArray[np.float64], num_iteration: int = 0
) -> NDArray[np.float64]: ...
def predict_proba_f32(
    model: Model, arr: NDArray[np.float32], num_iteration: int = 0
) -> NDArray[np.float64]: ...
def predict_raw_mixed(
    model: Model,
    n_rows: int,
    num_cols: list[NDArray[np.float64] | NDArray[np.float32]],
    cat_vocabs: list[list[str]],
    cat_indices: list[NDArray[np.int32]],
    num_iteration: int = 0,
) -> NDArray[np.float64]: ...
def predict_proba_mixed(
    model: Model,
    n_rows: int,
    num_cols: list[NDArray[np.float64] | NDArray[np.float32]],
    cat_vocabs: list[list[str]],
    cat_indices: list[NDArray[np.int32]],
    num_iteration: int = 0,
) -> NDArray[np.float64]: ...
class LignumPredictor:
    """A miter model built into lignum once; raw scores equal miter's evaluator bit for bit."""

    def __init__(self, model: Model, num_iteration: int = 0) -> None: ...
    @property
    def binned(self) -> bool: ...
    @property
    def n_oblique(self) -> int: ...
    @property
    def n_ctr_tables(self) -> int: ...
    def save(self, path: str) -> None: ...
    def predict_raw_f64(self, arr: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_raw_f32(self, arr: NDArray[np.float32]) -> NDArray[np.float64]: ...
    def predict_raw_mixed(
        self,
        n_rows: int,
        num_cols: list[NDArray[np.float64] | NDArray[np.float32]],
        cat_vocabs: list[list[str]],
        cat_indices: list[NDArray[np.int32]],
    ) -> NDArray[np.float64]: ...

def proba_from_raw(raw: NDArray[np.float64]) -> NDArray[np.float64]: ...
def proba_softmax_from_raw(raw: NDArray[np.float64]) -> NDArray[np.float64]: ...
def model_to_json(model: Model) -> str: ...
def model_from_json(json_text: str) -> Model: ...
def save_model_json(model: Model, path: str) -> None: ...
def load_model_json(path: str) -> Model: ...
