import logging
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any

# Lưu trữ Run ID và Log Path theo từng dataset và pipeline mode ('notebook' vs 'experiment')
_ACTIVE_RUN_IDS: Dict[str, str] = {}
_LOG_PATHS: Dict[str, Path] = {}
_ACTIVE_PIPELINE_MODE: Dict[str, str] = {}


def get_project_root() -> Path:
    """
    Purpose:
        Locate the root directory of the Lab_TimeSeries project dynamically
        using pathlib.Path without relying on hardcoded absolute paths.

    Parameters:
        None.

    Returns:
        Path: The resolved project root directory containing 'src/' and 'data/'.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


def normalize_dataset_key(dataset_name: str) -> str:
    """
    Purpose:
        Normalize a human-readable dataset name or identifier into a canonical
        folder key ('dataset1' or 'dataset2') for consistent log and report routing.

    Parameters:
        dataset_name (str): Name or key of the dataset (e.g., 'dataset1',
            'Monthly Retail Sales', 'dataset2', 'AirPassengers').

    Returns:
        str: Canonical dataset key ('dataset1', 'dataset2', or a sanitized slug).
    """
    raw = str(dataset_name or "dataset1").strip().lower()
    if any(k in raw for k in ["dataset1", "dataset_1", "dataset01", "retail", "a10", "sales"]):
        return "dataset1"
    if any(k in raw for k in ["dataset2", "dataset_2", "dataset02", "air", "passenger"]):
        return "dataset2"
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return sanitized or "dataset1"


def get_run_id(dataset_name: str = "dataset1", force_new: bool = False) -> str:
    """
    Purpose:
        Retrieve or generate a unique Run ID for tracking an execution session
        of a specific dataset across logs and model summary reports.

    Parameters:
        dataset_name (str): Target dataset identifier.
        force_new (bool): If True, generate a fresh Run ID even if one exists.

    Returns:
        str: A human-readable unique Run ID string (e.g., 'dataset1_20261112_004000_a1b2c3').
    """
    ds_key = normalize_dataset_key(dataset_name)
    if force_new or ds_key not in _ACTIVE_RUN_IDS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        _ACTIVE_RUN_IDS[ds_key] = f"{ds_key}_{timestamp}_{short_uuid}"
    return _ACTIVE_RUN_IDS[ds_key]


def get_log_file_path(
    dataset_name: str = "dataset1",
    base_dir: Optional[Path] = None,
    pipeline_mode: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    """
    Purpose:
        Resolve and create the target log file path for a specific dataset and pipeline:
        - Notebook pipeline: logs/lab02/<dataset_key>/lab02_<dataset_key>.log
        - Experiment pipeline: logs/lab02/experiments/<dataset_key>/experiment_<RUN_ID>.log
          (never overwrites previous experiment runs).

    Parameters:
        dataset_name (str): Target dataset identifier ('dataset1' or 'dataset2').
        base_dir (Optional[Path]): Custom project root directory if provided.
        pipeline_mode (Optional[str]): 'notebook' or 'experiment' (uses active mode if None).
        run_id (Optional[str]): Explicit Run ID for experiment log file naming.

    Returns:
        Path: Absolute path to the log file.
    """
    ds_key = normalize_dataset_key(dataset_name)
    mode = pipeline_mode or _ACTIVE_PIPELINE_MODE.get(ds_key, "notebook")
    root = Path(base_dir).resolve() if base_dir is not None else get_project_root()

    if mode == "experiment":
        active_rid = run_id or get_run_id(ds_key, force_new=False)
        log_dir = root / "logs" / "lab02" / "experiments" / ds_key
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"experiment_{active_rid}.log"
    else:
        log_dir = root / "logs" / "lab02" / ds_key
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"lab02_{ds_key}.log"

    _LOG_PATHS[ds_key] = log_file
    return log_file


def setup_lab02_logger(
    dataset_name: str = "dataset1",
    module_name: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = False,
    base_dir: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    pipeline_mode: Optional[str] = None,
    force_new_run: bool = False,
) -> logging.Logger:
    """
    Purpose:
        Configure and return a centralized Python logger for a LAB02 dataset
        supporting both the interactive Notebook Pipeline and the Automated
        Experiment Pipeline (with non-overwriting per-run log files), preventing
        duplicate handlers on repeated calls.

    Parameters:
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        module_name (Optional[str]): Optional submodule name to append to the logger hierarchy.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        console_output (bool): Whether to also emit log records to stdout.
        base_dir (Optional[Path]): Optional custom root path for log directory creation.
        max_bytes (int): Maximum size in bytes before log rotation (default 5 MB).
        backup_count (int): Number of rotated backup log files to retain.
        pipeline_mode (Optional[str]): 'notebook' or 'experiment'. If None, preserves current active mode.
        force_new_run (bool): If True, generate a fresh RUN_ID and bind a new experiment log file.

    Returns:
        logging.Logger: Configured logger instance ready for structured logging.
    """
    ds_key = normalize_dataset_key(dataset_name)
    if pipeline_mode is not None:
        _ACTIVE_PIPELINE_MODE[ds_key] = pipeline_mode
    active_mode = _ACTIVE_PIPELINE_MODE.get(ds_key, "notebook")

    run_id = get_run_id(ds_key, force_new=force_new_run)
    log_file = get_log_file_path(ds_key, base_dir=base_dir, pipeline_mode=active_mode, run_id=run_id)

    base_logger_name = f"lab02.{ds_key}"
    base_logger = logging.getLogger(base_logger_name)
    base_logger.setLevel(level)
    base_logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_target_file_handler = False
    for handler in list(base_logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            if Path(handler.baseFilename).resolve() == log_file.resolve():
                has_target_file_handler = True
            else:
                # Giữ lại file handler của notebook nếu đang chạy song song hoặc đóng handler cũ khác đường dẫn
                base_logger.removeHandler(handler)
                handler.close()

    if not has_target_file_handler:
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        base_logger.addHandler(file_handler)
        base_logger.info("=" * 88)
        base_logger.info(
            f"[SYSTEM] Initialized LAB02 Logger | pipeline={active_mode.upper()} | "
            f"dataset={ds_key} | run_id={run_id} | log_path={log_file}"
        )

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in base_logger.handlers
    )
    if console_output and not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        base_logger.addHandler(console_handler)
    elif not console_output and has_console_handler:
        for h in list(base_logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
                base_logger.removeHandler(h)

    if module_name:
        clean_mod = module_name.replace("src.", "").strip(".")
        child_logger = logging.getLogger(f"{base_logger_name}.{clean_mod}")
        child_logger.setLevel(level)
        return child_logger

    return base_logger


def setup_experiment_logger(
    dataset_name: str = "dataset1",
    run_id: Optional[str] = None,
    console_output: bool = False,
    base_dir: Optional[Path] = None,
) -> tuple[logging.Logger, str, Path]:
    """
    Purpose:
        Initialize a dedicated non-overwriting experiment logger at:
        logs/lab02/experiments/<dataset_key>/experiment_<RUN_ID>.log.

    Parameters:
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        run_id (Optional[str]): Optional custom Run ID; generates a fresh unique Run ID if None.
        console_output (bool): Whether to mirror logs to stdout.
        base_dir (Optional[Path]): Optional custom project root directory.

    Returns:
        tuple[logging.Logger, str, Path]: (logger, run_id, log_file_path).
    """
    ds_key = normalize_dataset_key(dataset_name)
    if run_id is not None:
        _ACTIVE_RUN_IDS[ds_key] = run_id
        force_new = False
    else:
        force_new = True

    logger = setup_lab02_logger(
        dataset_name=ds_key,
        module_name="experiment_runner",
        console_output=console_output,
        base_dir=base_dir,
        pipeline_mode="experiment",
        force_new_run=force_new,
    )
    active_rid = get_run_id(ds_key, force_new=False)
    log_path = get_log_file_path(ds_key, base_dir=base_dir, pipeline_mode="experiment", run_id=active_rid)
    return logger, active_rid, log_path


def get_lab02_logger(dataset_name: str = "dataset1", module_name: Optional[str] = None) -> logging.Logger:
    """
    Purpose:
        Convenience helper for any LAB02 module to obtain the active dataset logger
        (automatically routing to either the Notebook log or the Experiment log).

    Parameters:
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        module_name (Optional[str]): Name of the calling module.

    Returns:
        logging.Logger: Active logger for the specified dataset and module.
    """
    return setup_lab02_logger(dataset_name=dataset_name, module_name=module_name)


def log_data_loading(
    dataset_name: str,
    file_path: str | Path,
    n_obs: int,
    start_date: Any,
    end_date: Any,
    frequency: str,
    columns: list,
    missing_values: int,
    duplicate_timestamps: int = 0,
    is_monotonic: bool = True,
) -> None:
    """
    Purpose:
        Log structured metadata during the DATA LOADING phase without dumping raw arrays.
    """
    logger = get_lab02_logger(dataset_name, "data_loading")
    run_id = get_run_id(dataset_name)
    logger.info(
        f"[DATA_LOADING] run_id='{run_id}' | dataset='{dataset_name}' | path='{file_path}' | "
        f"n_obs={n_obs} | range=['{start_date}' -> '{end_date}'] | freq='{frequency}' | "
        f"columns={columns} | missing_values={missing_values} | "
        f"duplicate_timestamps={duplicate_timestamps} | chronological_order={is_monotonic}"
    )


def log_preprocessing(
    dataset_name: str,
    train_size: int,
    test_size: int,
    train_start: Any,
    train_end: Any,
    test_start: Any,
    test_end: Any,
    split_ratio: str = "80/20",
    differencing_info: str = "None",
    transformations: str = "None",
    output_dir: Optional[str | Path] = None,
) -> None:
    """
    Purpose:
        Log structured metadata during the DATA PREPROCESSING & TRAIN/TEST SPLIT phase.
    """
    logger = get_lab02_logger(dataset_name, "preprocessing")
    run_id = get_run_id(dataset_name)
    logger.info(
        f"[PREPROCESSING] run_id='{run_id}' | dataset='{dataset_name}' | split_ratio='{split_ratio}' | "
        f"train_obs={train_size} ('{train_start}' -> '{train_end}') | "
        f"test_obs={test_size} ('{test_start}' -> '{test_end}') | "
        f"differencing='{differencing_info}' | transformations='{transformations}' | "
        f"saved_to='{output_dir}'"
    )


def log_statistical_analysis(
    dataset_name: str,
    analysis_type: str,
    series_name: str,
    details: Dict[str, Any],
) -> None:
    """
    Purpose:
        Log structured results from STATISTICAL ANALYSIS steps (ADF, ACF/PACF).
    """
    logger = get_lab02_logger(dataset_name, "statistical_analysis")
    run_id = get_run_id(dataset_name)
    kv_str = " | ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"[{analysis_type}] run_id='{run_id}' | series='{series_name}' | {kv_str}")


def log_model_training(
    dataset_name: str,
    model_name: str,
    model_family: str,
    order: tuple,
    seasonal_order: Optional[tuple],
    start_time: str,
    end_time: str,
    duration_sec: float,
    aic: float,
    bic: float,
    llf: float,
    converged: bool,
    warnings_list: list,
    parameters_summary: Optional[Dict[str, float]] = None,
) -> None:
    """
    Purpose:
        Log structured telemetry during the MODEL TRAINING phase, including
        Run ID, model family, configuration, estimated parameters, execution timing,
        AIC/BIC, Log-Likelihood, convergence status, and warnings.
    """
    logger = get_lab02_logger(dataset_name, f"training.{model_family.lower()}")
    run_id = get_run_id(dataset_name)
    param_str = f" | params={parameters_summary}" if parameters_summary else ""
    msg = (
        f"[MODEL_TRAINING] run_id='{run_id}' | dataset='{dataset_name}' | model='{model_name}' | "
        f"family='{model_family}' | order={order} | seasonal_order={seasonal_order} | "
        f"start='{start_time}' | end='{end_time}' | duration={duration_sec:.4f}s | "
        f"AIC={aic:.4f} | BIC={bic:.4f} | LogLikelihood={llf:.4f} | "
        f"converged={converged} | warnings={len(warnings_list)}{param_str}"
    )
    if not converged or warnings_list:
        logger.warning(f"{msg} | warning_details={warnings_list}")
    else:
        logger.info(msg)


def log_model_evaluation(
    dataset_name: str,
    model_name: str,
    mae: float,
    rmse: float,
    mape: float,
    r2: float,
    resid_mean: float,
    resid_std: float,
    lb_pvalue: float,
    jb_pvalue: Optional[float] = None,
    het_pvalue: Optional[float] = None,
    extra_notes: str = "",
    output_locations: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Purpose:
        Log structured metrics from the MODEL EVALUATION and RESIDUAL DIAGNOSTICS phases.
    """
    logger = get_lab02_logger(dataset_name, "evaluation")
    run_id = get_run_id(dataset_name)
    jb_str = f"{jb_pvalue:.4f}" if jb_pvalue is not None else "N/A"
    het_str = f"{het_pvalue:.4f}" if het_pvalue is not None else "N/A"
    out_str = f" | outputs={output_locations}" if output_locations else ""
    logger.info(
        f"[MODEL_EVALUATION] run_id='{run_id}' | dataset='{dataset_name}' | model='{model_name}' | "
        f"MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.4f}% | R2={r2:.4f} | "
        f"resid_mean={resid_mean:.4f} | resid_std={resid_std:.4f} | LjungBox_p={lb_pvalue:.4f} | "
        f"JarqueBera_p={jb_str} | Heteroskedasticity_p={het_str} | notes='{extra_notes}'{out_str}"
    )


def log_system_status(
    dataset_name: str,
    status: str,
    output_paths: Dict[str, Any],
    message: str = "",
) -> None:
    """
    Purpose:
        Log overall SYSTEM execution status, Run ID, timestamp, and generated artifact paths.
    """
    logger = get_lab02_logger(dataset_name, "system")
    run_id = get_run_id(dataset_name)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_fn = logger.info if status.upper() == "SUCCESS" else logger.error
    log_fn(
        f"[SYSTEM_STATUS] run_id='{run_id}' | dataset='{dataset_name}' | timestamp='{ts}' | "
        f"status='{status}' | message='{message}' | outputs={output_paths}"
    )
