#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logging configuration for RSD.

This module provides a unified logging interface for the entire project,
replacing print() statements with proper logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "rsd",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure logging for RSD.
    
    Args:
        name: Logger name (default: "rsd")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Whether to log to console (default: True)
    
    Returns:
        Configured logger instance
    
    Examples:
        >>> logger = setup_logger("rsd", level=logging.DEBUG)
        >>> logger.info("Simulation started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers (pero respetar el nuevo nivel solicitado)
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger
    
    # Formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "rsd") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: "rsd")
    
    Returns:
        Logger instance
    
    Examples:
        >>> from rsd.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
    """
    return logging.getLogger(name)


def set_log_level(level: int, logger_name: str = "rsd") -> None:
    """
    Set logging level for a specific logger.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger to configure
    
    Examples:
        >>> import logging
        >>> from rsd.utils.logger import set_log_level
        >>> set_log_level(logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
