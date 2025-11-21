"""Configuration loading and management for lite-chat."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    contexts: list[int] = None  # Optional; used for Ollama/ctx-length


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    id: str
    type: str  # e.g., "ollama", "openai-compatible"
    api_url: str
    api_key: str = ""
    models: list[ModelConfig] = None
    streaming: bool = True
    headers: dict = None
    default_model: Optional[str] = None


@dataclass
class Config:
    """Application configuration."""
    api_url: str
    api_key: str
    conversations_dir: Path
    exports_dir: Optional[Path]
    enable_streaming: bool
    system_prompt: Optional[str]
    default_provider: str
    default_model: str
    default_temperature: float
    timeout: int  # API request timeout in seconds
    log_level: str  # Logging level: DEBUG, INFO, WARNING, ERROR
    log_file: Optional[Path]  # Path to log file (None = stderr)
    providers: list[ProviderConfig]

    def get_provider(self, provider_id: str) -> ProviderConfig:
        """Get provider configuration by id."""
        for p in self.providers:
            if p.id == provider_id:
                return p
        raise ValueError(f"Provider '{provider_id}' not found in configuration")

    def get_model(self, provider_id: str, name: str) -> ModelConfig:
        """Get model configuration by provider and name.

        Args:
            provider_id: Provider id
            name: Model name to look up

        Returns:
            ModelConfig for the specified model

        Raises:
            ValueError: If model is not found
        """
        provider = self.get_provider(provider_id)
        if provider.models:
            for model in provider.models:
                if model.name == name:
                    return model
        # If models not specified, allow dynamic model names
        return ModelConfig(name=name, contexts=None)

    def list_models(self, provider_id: str) -> list[ModelConfig]:
        provider = self.get_provider(provider_id)
        return provider.models or []


def _setup_logging(config: Config) -> None:
    """Configure logging based on config settings.

    Args:
        config: Configuration object with logging settings
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, config.log_level, logging.INFO)

    # Create formatter with timestamp, level, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Add appropriate handler based on config
    if config.log_file:
        # Ensure log directory exists
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(config.log_file)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(numeric_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log the configuration
    logging.info(f"Logging initialized: level={config.log_level}, file={config.log_file}")


def load_config() -> Config:
    """Load configuration from ~/.lite-chat/config.toml.

    Returns:
        Config object with loaded or default values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path.home() / ".lite-chat" / "config.toml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}\n"
            "Please create ~/.lite-chat/config.toml with your settings."
        )

    # Load TOML file
    with open(config_path, 'r') as f:
        data = toml.load(f)

    # Extract sections
    general_section = data.get('general', {})
    ollama_section = data.get('ollama', {})

    # Parse providers
    providers_config = data.get('providers', [])
    models_data_legacy = data.get('models', [])

    # Get general configuration values
    log_level = general_section.get('log_level', 'INFO').upper()
    log_file_str = general_section.get('log_file')
    log_file = Path(log_file_str).expanduser() if log_file_str else None

    exports_dir_str = general_section.get('exports_dir')
    exports_dir = Path(exports_dir_str).expanduser() if exports_dir_str else None
    if exports_dir:
        exports_dir.mkdir(parents=True, exist_ok=True)

    # Get Ollama configuration values with defaults
    api_url = ollama_section.get('api_url', 'http://localhost:11434/api')
    api_key = ollama_section.get('api_key', '')

    conversations_dir_str = general_section.get(
        'conversations_dir',
        str(Path.home() / '.lite-chat' / 'conversations')
    )
    conversations_dir = Path(conversations_dir_str).expanduser()

    # Ensure conversations directory exists
    conversations_dir.mkdir(parents=True, exist_ok=True)

    enable_streaming = bool(general_section.get('enable_streaming', False))

    system_prompt = general_section.get('system_prompt', ollama_section.get('system_prompt'))
    default_model = general_section.get('default_model') or ollama_section.get('default_model')
    default_temperature = general_section.get('default_temperature', ollama_section.get('default_temperature', 0.7))
    timeout = general_section.get('timeout', ollama_section.get('timeout', 1200))  # Default: 20 minutes

    default_provider = general_section.get('default_provider', 'ollama')

    providers: list[ProviderConfig] = []

    def parse_models_field(value) -> list[ModelConfig]:
        if value is None:
            return []
        if isinstance(value, str):
            names = [n.strip() for n in value.split(',') if n.strip()]
            return [ModelConfig(name=n, contexts=None) for n in names]
        if isinstance(value, list):
            result = []
            for entry in value:
                if isinstance(entry, dict):
                    name = entry.get('name')
                    if not name:
                        continue
                    contexts_val = entry.get('contexts')
                    contexts = None
                    if contexts_val:
                        try:
                            contexts = [int(c.strip()) for c in str(contexts_val).split(',')]
                        except Exception:
                            contexts = None
                    result.append(ModelConfig(name=name, contexts=contexts))
                elif isinstance(entry, str):
                    result.append(ModelConfig(name=entry.strip(), contexts=None))
            return result
        return []

    if providers_config:
        for entry in providers_config:
            pid = entry.get('id')
            ptype = entry.get('type')
            api_url = entry.get('api_url')
            if not pid or not ptype or not api_url:
                raise ValueError("Each provider must have id, type, and api_url")
            models = parse_models_field(entry.get('models'))
            providers.append(ProviderConfig(
                id=pid,
                type=ptype,
                api_url=api_url,
                api_key=entry.get('api_key', ''),
                models=models,
                streaming=entry.get('streaming', True),
                headers=entry.get('headers', {}),
                default_model=entry.get('default_model')
            ))
    else:
        # Legacy config: build a single ollama provider from [ollama] and [[models]]
        legacy_models = []
        for model_data in models_data_legacy:
            name = model_data.get('name')
            if not name:
                raise ValueError("Model missing 'name' field")

            contexts_str = model_data.get('contexts', '')
            if not contexts_str:
                raise ValueError(f"Model '{name}' missing 'contexts' field")

            try:
                contexts = [int(c.strip()) for c in contexts_str.split(',')]
            except ValueError:
                raise ValueError(f"Invalid contexts format for model '{name}': {contexts_str}")

            legacy_models.append(ModelConfig(name=name, contexts=contexts))

        providers.append(ProviderConfig(
            id='ollama',
            type='ollama',
            api_url=api_url,
            api_key=api_key,
            models=legacy_models,
            streaming=True,
            headers={},
            default_model=default_model
        ))
        default_provider = 'ollama'

    # Validate default provider exists
    provider_ids = [p.id for p in providers]
    if default_provider not in provider_ids:
        raise ValueError(f"default_provider '{default_provider}' not found in providers {provider_ids}")

    # If default_model not set, try provider default or first model
    if not default_model:
        try:
            default_provider_obj = next(p for p in providers if p.id == default_provider)
        except StopIteration:
            default_provider_obj = None
        if default_provider_obj:
            if default_provider_obj.default_model:
                default_model = default_provider_obj.default_model
            elif default_provider_obj.models:
                default_model = default_provider_obj.models[0].name

    # Validate default model exists in default provider if available
    try:
        default_provider_obj = next(p for p in providers if p.id == default_provider)
    except StopIteration:
        default_provider_obj = None
    if default_provider_obj and default_model:
        model_names = [m.name for m in default_provider_obj.models or []]
        if model_names and default_model not in model_names:
            raise ValueError(
                f"default_model '{default_model}' not found in provider '{default_provider}'. Options: {model_names}"
            )

    config = Config(
        api_url=api_url,
        api_key=api_key,
        conversations_dir=conversations_dir,
        exports_dir=exports_dir,
        enable_streaming=enable_streaming,
        system_prompt=system_prompt,
        default_provider=default_provider,
        default_model=default_model,
        default_temperature=default_temperature,
        timeout=timeout,
        log_level=log_level,
        log_file=log_file,
        providers=providers
    )

    # Configure logging based on settings
    _setup_logging(config)

    return config
