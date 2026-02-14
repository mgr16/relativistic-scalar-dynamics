#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logging configuration for PSYOP.

This module provides a unified logging interface for the entire project,
replacing print() statements with proper logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "psyop",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure logging for PSYOP.
    
    Args:
        name: Logger name (default: "psyop")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Whether to log to console (default: True)
    
    Returns:
        Configured logger instance
    
    Examples:
        >>> logger = setup_logger("psyop", level=logging.DEBUG)
        >>> logger.info("Simulation started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
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


def get_logger(name: str = "psyop") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: "psyop")
    
    Returns:
        Logger instance
    
    Examples:
        >>> from psyop.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
    """
    return logging.getLogger(name)


def set_log_level(level: int, logger_name: str = "psyop") -> None:
    """
    Set logging level for a specific logger.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger to configure
    
    Examples:
        >>> import logging
        >>> from psyop.utils.logger import set_log_level
        >>> set_log_level(logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
