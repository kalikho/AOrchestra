"""Configuration for aorchestra."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class GAIAOrchestraConfig:
    """GAIA Orchestra configuration"""
    
    # Model
    main_model: str
    sub_models: List[str]
    
    # GAIA dataset
    dataset_path: Path
    attachments_dir: Path
    level_filter: List[int] | None = None
    max_tasks: int | None = None
    
    # Execution
    max_steps: int = 30
    max_attempts: int = 5
    max_concurrency: int = 1
    
    # Output
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_folder: Path = field(default_factory=lambda: Path("workspace/logs/trajectories"))
    timestamp: str | None = None
    
    @classmethod
    def load(cls, config_path: Path | str) -> "GAIAOrchestraConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        # Required fields
        main_model = raw.get("main_model")
        if not main_model:
            raise ValueError("main_model is required")
        
        sub_models = raw.get("sub_models")
        if not sub_models or not isinstance(sub_models, list):
            raise ValueError("sub_models must be a non-empty list")
        
        dataset_path = raw.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required")
        dataset_path = cls._resolve_path(dataset_path, config_path)
        
        attachments_dir = raw.get("attachments_dir")
        if not attachments_dir:
            raise ValueError("attachments_dir is required")
        attachments_dir = cls._resolve_path(attachments_dir, config_path)
        
        # Optional fields
        level_filter = raw.get("level_filter")
        max_tasks = raw.get("max_tasks")
        max_steps = int(raw.get("max_steps", 30))
        max_attempts = int(raw.get("max_attempts", 5))
        max_concurrency = int(raw.get("max_concurrency", 1))
        
        result_folder = cls._resolve_path(
            raw.get("result_folder", "workspace/logs"),
            config_path
        )
        trajectory_folder = cls._resolve_path(
            raw.get("trajectory_folder", "workspace/logs/trajectories"),
            config_path
        )
        
        return cls(
            main_model=str(main_model),
            sub_models=[str(m) for m in sub_models],
            dataset_path=dataset_path,
            attachments_dir=attachments_dir,
            level_filter=level_filter,
            max_tasks=max_tasks,
            max_steps=max_steps,
            max_attempts=max_attempts,
            max_concurrency=max_concurrency,
            result_folder=result_folder,
            trajectory_folder=trajectory_folder,
        )
    
    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        """Resolve path (relative to config file or project root)"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try relative to config file
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        # Try relative to project root
        PROJECT_ROOT = Path(__file__).parent.parent
        return (PROJECT_ROOT / path).resolve()


@dataclass
class TerminalBenchOrchestraConfig:
    """TerminalBench Orchestra configuration"""
    
    # Model
    main_model: str
    sub_models: List[str]
    
    # Tasks
    tasks_dir: Path
    max_tasks: int | None = None
    
    # Execution
    max_steps: int = 30
    max_attempts: int = 10
    max_concurrency: int = 1
    sandbox: str = "docker"  # docker | e2b | daytona
    docker_timeout: int = 600
    
    # Output
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_dir: Path | None = None
    csv_summary_path: Path | None = None
    timestamp: str | None = None
    
    # Sandbox API keys (for e2b/daytona)
    e2b_api_key: str | None = None
    daytona_api_key: str | None = None
    daytona_api_url: str | None = None
    daytona_target: str | None = None
    
    # Environment init
    env_init: dict[str, str] | None = None
    
    @classmethod
    def load(cls, config_path: Path | str) -> "TerminalBenchOrchestraConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        # Required fields
        main_model = raw.get("main_model")
        if not main_model:
            raise ValueError("main_model is required")
        
        sub_models = raw.get("sub_models")
        if not sub_models or not isinstance(sub_models, list):
            raise ValueError("sub_models must be a non-empty list")
        
        tasks_dir = raw.get("tasks_dir")
        if not tasks_dir:
            raise ValueError("tasks_dir is required")
        tasks_dir = cls._resolve_path(tasks_dir, config_path)
        
        # Optional fields
        max_tasks = raw.get("max_tasks")
        max_steps = int(raw.get("max_steps", 30))
        max_attempts = int(raw.get("max_attempts", 10))
        max_concurrency = int(raw.get("max_concurrency", 1))
        sandbox = str(raw.get("sandbox", "docker"))
        docker_timeout = int(raw.get("docker_timeout", 600))
        
        result_folder = cls._resolve_path(
            raw.get("result_folder", "workspace/logs"),
            config_path
        )
        
        trajectory_dir = raw.get("trajectory_dir")
        if trajectory_dir:
            trajectory_dir = cls._resolve_path(trajectory_dir, config_path)
        
        csv_summary_path = raw.get("csv_summary_path")
        if csv_summary_path:
            csv_summary_path = cls._resolve_path(csv_summary_path, config_path)
        
        return cls(
            main_model=str(main_model),
            sub_models=[str(m) for m in sub_models],
            tasks_dir=tasks_dir,
            max_tasks=max_tasks,
            max_steps=max_steps,
            max_attempts=max_attempts,
            max_concurrency=max_concurrency,
            sandbox=sandbox,
            docker_timeout=docker_timeout,
            result_folder=result_folder,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
            env_init=raw.get("env_init"),
            e2b_api_key=raw.get("e2b_api_key"),
            daytona_api_key=raw.get("daytona_api_key"),
            daytona_api_url=raw.get("daytona_api_url"),
            daytona_target=raw.get("daytona_target"),
        )
    
    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        """Resolve path (relative to config file or project root)"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try relative to config file
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        # Try relative to project root
        PROJECT_ROOT = Path(__file__).parent.parent
        return (PROJECT_ROOT / path).resolve()


@dataclass
class SWEBenchOrchestraConfig:
    """SWE-bench Orchestra configuration"""
    
    # Model
    main_model: str
    sub_models: List[str]
    
    # SWE-bench dataset settings
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    subset_seed: Optional[int] = None
    subset_sizes: Optional[Dict[str, int]] = None
    subset_role: Optional[str] = None
    
    # Instance ID file (JSON array)
    selected_ids_file: Optional[Path] = None
    
    # Task settings
    max_tasks: Optional[int] = None
    max_steps: int = 50  # SubAgent max steps
    max_attempts: int = 10  # MainAgent max attempts
    
    # Execution
    max_concurrency: int = 1
    docker_timeout: int = 1800
    
    # Output
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_dir: Optional[Path] = None
    csv_summary_path: Optional[Path] = None
    timestamp: Optional[str] = None
    
    # Environment initialization
    env_init: Optional[Dict[str, str]] = None
    
    # HuggingFace cache directory
    cache_dir: Optional[str] = None
    
    # ACI file view window size
    window_size: int = 100
    
    @classmethod
    def load(cls, config_path: Path | str) -> "SWEBenchOrchestraConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        # Required fields
        main_model = raw.get("main_model")
        if not main_model:
            raise ValueError("main_model is required")
        
        sub_models = raw.get("sub_models")
        if not sub_models or not isinstance(sub_models, list):
            raise ValueError("sub_models must be a non-empty list")
        
        # SWE-bench dataset settings
        dataset_name = raw.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        split = raw.get("split", "test")
        
        subset_seed = raw.get("subset_seed")
        if subset_seed is not None:
            subset_seed = int(subset_seed)
        
        subset_sizes = raw.get("subset_sizes")
        if subset_sizes:
            subset_sizes = {
                str(k): int(v)
                for k, v in subset_sizes.items()
                if v is not None
            }
        
        subset_role = raw.get("subset_role")
        
        # Optional paths
        result_folder = cls._resolve_path(
            raw.get("result_folder", "workspace/logs"),
            config_path
        )
        
        trajectory_dir = raw.get("trajectory_dir")
        if trajectory_dir:
            trajectory_dir = cls._resolve_path(trajectory_dir, config_path)
        
        csv_summary_path = raw.get("csv_summary_path")
        if csv_summary_path:
            csv_summary_path = cls._resolve_path(csv_summary_path, config_path)
        
        selected_ids_file = raw.get("selected_ids_file")
        if selected_ids_file:
            selected_ids_file = cls._resolve_path(selected_ids_file, config_path)
        
        return cls(
            main_model=str(main_model),
            sub_models=[str(m) for m in sub_models],
            dataset_name=str(dataset_name),
            split=str(split),
            subset_seed=subset_seed,
            subset_sizes=subset_sizes,
            subset_role=subset_role,
            selected_ids_file=selected_ids_file,
            max_tasks=raw.get("max_tasks"),
            max_steps=int(raw.get("max_steps", 50)),
            max_attempts=int(raw.get("max_attempts", 10)),
            max_concurrency=int(raw.get("max_concurrency", 1)),
            docker_timeout=int(raw.get("docker_timeout", 1800)),
            result_folder=result_folder,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
            env_init=raw.get("env_init"),
            cache_dir=raw.get("cache_dir"),
            window_size=int(raw.get("window_size", 100)),
        )
    
    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        """Resolve path (relative to config file or project root)"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try relative to config file
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        # Try relative to project root
        PROJECT_ROOT = Path(__file__).parent.parent
        return (PROJECT_ROOT / path).resolve()
