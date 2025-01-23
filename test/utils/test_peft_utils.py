import os
import pytest
from unittest.mock import patch, mock_open
import importlib.metadata

from transformers.utils.peft_utils import (
    find_adapter_config_file,
    check_peft_version,
    ADAPTER_CONFIG_NAME,
)


def test_find_adapter_config_file_local_dir(tmp_path):
    # Test with local directory containing adapter config
    config_path = tmp_path / ADAPTER_CONFIG_NAME
    config_path.write_text("{}")

    result = find_adapter_config_file(str(tmp_path))
    assert result == str(config_path)


def test_find_adapter_config_file_local_dir_no_config(tmp_path):
    # Test with local directory not containing adapter config
    result = find_adapter_config_file(str(tmp_path))
    assert result is None


def test_find_adapter_config_file_none_input():
    # Test with None input
    result = find_adapter_config_file(None)
    assert result is None


@patch("transformers.utils.peft_utils.cached_file")
def test_find_adapter_config_file_remote(mock_cached_file, tmp_path):
    # Test with remote model ID
    expected_path = str(tmp_path / ADAPTER_CONFIG_NAME)
    mock_cached_file.return_value = expected_path

    result = find_adapter_config_file("remote/model")
    assert result == expected_path

    mock_cached_file.assert_called_once_with(
        "remote/model",
        ADAPTER_CONFIG_NAME,
        cache_dir=None,
        force_download=False,
        resume_download=None,
        proxies=None,
        token=None,
        revision=None,
        local_files_only=False,
        subfolder="",
        _commit_hash=None,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


@patch("transformers.utils.peft_utils.is_peft_available")
def test_check_peft_version_not_installed(mock_is_peft_available):
    # Test when PEFT is not installed
    mock_is_peft_available.return_value = False

    with pytest.raises(ValueError, match="PEFT is not installed"):
        check_peft_version("0.1.0")


@patch("transformers.utils.peft_utils.is_peft_available")
@patch("importlib.metadata.version")
def test_check_peft_version_compatible(mock_version, mock_is_peft_available):
    # Test with compatible PEFT version
    mock_is_peft_available.return_value = True
    mock_version.return_value = "0.5.0"

    # Should not raise error
    check_peft_version("0.4.0")


@patch("transformers.utils.peft_utils.is_peft_available")
@patch("importlib.metadata.version")
def test_check_peft_version_incompatible(mock_version, mock_is_peft_available):
    # Test with incompatible PEFT version
    mock_is_peft_available.return_value = True
    mock_version.return_value = "0.3.0"

    with pytest.raises(ValueError, match="version of PEFT you are using is not compatible"):
        check_peft_version("0.4.0")


@patch("transformers.utils.peft_utils.cached_file")
def test_find_adapter_config_file_with_options(mock_cached_file, tmp_path):
    # Test with all optional parameters
    config_path = tmp_path / ADAPTER_CONFIG_NAME
    config_path.write_text("{}")
    mock_cached_file.return_value = str(config_path)

    result = find_adapter_config_file(
        "model_id",
        cache_dir="cache",
        force_download=True,
        resume_download=True,
        proxies={"http": "proxy"},
        token="token",
        revision="main",
        local_files_only=True,
        subfolder="subfolder",
        _commit_hash="hash"
    )

    assert result == str(config_path)
    mock_cached_file.assert_called_once_with(
        "model_id",
        ADAPTER_CONFIG_NAME,
        cache_dir="cache",
        force_download=True,
        resume_download=True,
        proxies={"http": "proxy"},
        token="token",
        revision="main",
        local_files_only=True,
        subfolder="subfolder",
        _commit_hash="hash",
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
