"""Configuration loading and management for lite-chat."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    contexts: list[int]


@dataclass
class Config:
    """Application configuration."""
    api_url: str
    api_key: str
    conversations_dir: Path
    system_prompt: Optional[str]
    default_model: str
    default_temperature: float
    models: list[ModelConfig]

    def get_model(self, name: str) -> ModelConfig:
        """Get model configuration by name.

        Args:
            name: Model name to look up

        Returns:
            ModelConfig for the specified model

        Raises:
            ValueError: If model is not found
        """
        for model in self.models:
            if model.name == name:
                return model
        raise ValueError(f"Model '{name}' not found in configuration")


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

    # Extract ollama section
    ollama_section = data.get('ollama', {})

    # Parse models
    models_data = data.get('models', [])
    if not models_data:
        raise ValueError("No models configured in config.toml")

    models = []
    for model_data in models_data:
        name = model_data.get('name')
        if not name:
            raise ValueError("Model missing 'name' field")

        contexts_str = model_data.get('contexts', '')
        if not contexts_str:
            raise ValueError(f"Model '{name}' missing 'contexts' field")

        # Parse comma-separated context lengths
        try:
            contexts = [int(c.strip()) for c in contexts_str.split(',')]
        except ValueError:
            raise ValueError(f"Invalid contexts format for model '{name}': {contexts_str}")

        models.append(ModelConfig(name=name, contexts=contexts))

    # Get configuration values with defaults
    api_url = ollama_section.get('api_url', 'http://localhost:11434/api')
    api_key = ollama_section.get('api_key', '')

    conversations_dir_str = ollama_section.get(
        'conversations_dir',
        str(Path.home() / '.lite-chat' / 'conversations')
    )
    conversations_dir = Path(conversations_dir_str).expanduser()

    # Ensure conversations directory exists
    conversations_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = ollama_section.get('system_prompt')
    default_model = ollama_section.get('default_model')
    default_temperature = ollama_section.get('default_temperature', 0.7)

    # Validate default_model
    if not default_model:
        raise ValueError("'default_model' must be specified in [ollama] section")

    # Check that default_model exists in models
    model_names = [m.name for m in models]
    if default_model not in model_names:
        raise ValueError(
            f"default_model '{default_model}' not found in configured models: {model_names}"
        )

    return Config(
        api_url=api_url,
        api_key=api_key,
        conversations_dir=conversations_dir,
        system_prompt=system_prompt,
        default_model=default_model,
        default_temperature=default_temperature,
        models=models
    )
