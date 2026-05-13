"""
OpenRLVR Pipeline Logger - Concurrent-safe, minimal, visualized logging.

Design Philosophy:
- **Minimal**: Default INFO level shows only key milestones
- **Concurrent-safe**: Thread-local context for parallel execution
- **Visualized**: Clear hierarchy with colors and progress indicators
- **Zero-overhead**: Fast checks prevent unnecessary formatting
- **Pipeline-aware**: Specialized logging for task/setup/reward stages

Usage:
    # Initialize once at pipeline start
    from utils.logger import init_logger, LogLevel
    logger = init_logger(level=LogLevel.INFO, log_file=Path("logs/pipeline.log"))

    # Use in any module
    from utils.logger import info, debug, TaskContext

    with TaskContext("chrome_bookmarks", "chrome", "setup"):
        info("Starting setup generation")
        debug("Calling LLM with prompt length 2000")
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextvars import ContextVar
from enum import IntEnum


# Thread-safe context variables
_task_id: ContextVar[Optional[str]] = ContextVar('task_id', default=None)
_module_name: ContextVar[Optional[str]] = ContextVar('module_name', default=None)
_stage: ContextVar[Optional[str]] = ContextVar('stage', default=None)


class LogLevel(IntEnum):
    """
    Hierarchical log levels for controlling verbosity.

    SILENT: No output (for testing/benchmarking)
    ERROR: Only errors and failures
    WARN: Errors + warnings
    INFO: Key milestones + warnings + errors (DEFAULT - minimal but informative)
    DEBUG: Detailed steps for debugging
    TRACE: Everything including LLM prompts/responses
    """
    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3      # Default - shows key events only
    DEBUG = 4
    TRACE = 5


class PipelineLogger:
    """
    Thread-safe logger optimized for concurrent pipeline execution.

    Features:
    - Minimal output by default (INFO level)
    - Automatic task context tracking in concurrent scenarios
    - Progress visualization for batch operations
    - Specialized logging for pipeline stages (task/setup/reward)
    - File + console output with color support
    """

    # ANSI colors (minimal set)
    C = {
        'RESET': '\033[0m',
        'ERROR': '\033[91m',      # Red
        'WARN': '\033[93m',       # Yellow
        'INFO': '\033[92m',       # Green
        'DEBUG': '\033[94m',      # Blue
        'TRACE': '\033[90m',      # Gray
        'CONTEXT': '\033[96m',    # Cyan
        'DIM': '\033[2m',         # Dim
        'BOLD': '\033[1m',        # Bold
    }

    # Stage icons
    STAGE_ICONS = {
        'task': '📋',
        'setup': '⚙️',
        'reward': '🎯',
        'pipeline': '🔄',
    }

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        use_colors: bool = True,
        show_timestamp: bool = True,
        show_context: bool = True
    ):
        """
        Initialize pipeline logger.

        Args:
            level: Log level threshold
            log_file: Optional file path for persistent logging
            use_colors: Enable colored output (disable for CI/logs)
            show_timestamp: Show timestamps in output
            show_context: Show task context in concurrent execution
        """
        self.level = level
        self.log_file = log_file
        self.use_colors = use_colors
        self.show_timestamp = show_timestamp
        self.show_context = show_context

        # Thread-safe primitives
        self._file_lock = threading.Lock()
        self._timer_lock = threading.Lock()
        self._progress_lock = threading.Lock()

        # Task timing tracking
        self._task_timers: Dict[str, float] = {}

        # Progress tracking for batch operations
        self._progress_state: Dict[str, Dict[str, Any]] = {}

        # Initialize log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._write_header()

    def _write_header(self):
        """Write log file header"""
        with self._file_lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"OpenRLVR Pipeline Log - {datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n\n")

    def _should_log(self, level: LogLevel) -> bool:
        """Fast path: check if message should be logged"""
        return level <= self.level

    def _get_context_str(self) -> str:
        """Build context string from thread-local variables"""
        if not self.show_context:
            return ""

        parts = []

        # Stage (task/setup/reward)
        stage = _stage.get()
        if stage:
            icon = self.STAGE_ICONS.get(stage, '')
            parts.append(f"{icon}{stage}")

        # Module name
        module = _module_name.get()
        if module:
            parts.append(module)

        # Task ID (truncated)
        task_id = _task_id.get()
        if task_id:
            short_id = task_id[:20] + "..." if len(task_id) > 20 else task_id
            parts.append(short_id)

        if parts:
            ctx = "|".join(parts)
            if self.use_colors:
                return f"{self.C['CONTEXT']}[{ctx}]{self.C['RESET']}"
            else:
                return f"[{ctx}]"

        return ""

    def _format_message(
        self,
        level_name: str,
        message: str,
        color_key: Optional[str] = None
    ) -> str:
        """Format log message with context and styling"""
        parts = []

        # Timestamp
        if self.show_timestamp:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if self.use_colors:
                parts.append(f"{self.C['DIM']}[{ts}]{self.C['RESET']}")
            else:
                parts.append(f"[{ts}]")

        # Context (stage|module|task)
        ctx = self._get_context_str()
        if ctx:
            parts.append(ctx)

        # Level
        color = self.C.get(color_key, '') if self.use_colors and color_key else ''
        reset = self.C['RESET'] if self.use_colors else ''
        parts.append(f"{color}[{level_name:5s}]{reset}")

        # Message
        parts.append(message)

        return " ".join(parts)

    def _write(self, formatted_msg: str):
        """Write to console and file"""
        # Console
        print(formatted_msg)

        # File (strip ANSI codes)
        if self.log_file:
            with self._file_lock:
                try:
                    import re
                    clean = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', formatted_msg)
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(clean + '\n')
                except Exception as e:
                    print(f"[Logger] Failed to write to log file: {e}")

    # ==================== Core Logging Methods ====================

    def error(self, message: str):
        """Log error message (always visible unless SILENT)"""
        if self._should_log(LogLevel.ERROR):
            msg = self._format_message("ERROR", f"❌ {message}", "ERROR")
            self._write(msg)

    def warn(self, message: str):
        """Log warning message"""
        if self._should_log(LogLevel.WARN):
            msg = self._format_message("WARN", f"⚠️  {message}", "WARN")
            self._write(msg)

    def info(self, message: str):
        """Log info message (key milestones)"""
        if self._should_log(LogLevel.INFO):
            msg = self._format_message("INFO", message, "INFO")
            self._write(msg)

    def debug(self, message: str):
        """Log debug message (detailed steps)"""
        if self._should_log(LogLevel.DEBUG):
            msg = self._format_message("DEBUG", message, "DEBUG")
            self._write(msg)

    def trace(self, message: str):
        """Log trace message (verbose details)"""
        if self._should_log(LogLevel.TRACE):
            msg = self._format_message("TRACE", message, "TRACE")
            self._write(msg)

    # ==================== Task Context Management ====================

    def set_context(
        self,
        task_id: Optional[str] = None,
        module: Optional[str] = None,
        stage: Optional[str] = None
    ):
        """Set logging context for current thread"""
        if task_id is not None:
            _task_id.set(task_id)
        if module is not None:
            _module_name.set(module)
        if stage is not None:
            _stage.set(stage)

    def clear_context(self):
        """Clear logging context for current thread"""
        _task_id.set(None)
        _module_name.set(None)
        _stage.set(None)

    def task_start(self, task_id: str, module: str, stage: str):
        """Mark task start with automatic context and timing"""
        self.set_context(task_id=task_id, module=module, stage=stage)

        with self._timer_lock:
            self._task_timers[task_id] = time.time()

        stage_icon = self.STAGE_ICONS.get(stage, '🔹')
        self.info(f"{stage_icon} Start: {module}")
        self.debug(f"Task ID: {task_id}")

    def task_end(self, task_id: str, success: bool = True):
        """Mark task completion with timing"""
        duration = None
        with self._timer_lock:
            if task_id in self._task_timers:
                duration = time.time() - self._task_timers[task_id]
                del self._task_timers[task_id]

        dur_str = f" ({duration:.2f}s)" if duration else ""

        if success:
            self.info(f"✅ Complete{dur_str}")
        else:
            self.error(f"❌ Failed{dur_str}")

        self.clear_context()

    # ==================== Pipeline-Specific Logging ====================

    def llm_call(
        self,
        model: str,
        prompt_length: int,
        response_length: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """Log LLM API call details"""
        details = [f"model={model}", f"prompt={prompt_length}chars"]

        if response_length:
            details.append(f"response={response_length}chars")
        if temperature is not None:
            details.append(f"temp={temperature}")

        self.debug(f"🤖 LLM call: {', '.join(details)}")

    def llm_response(self, response_preview: str, max_length: int = 100):
        """Log LLM response preview (TRACE level only)"""
        if self._should_log(LogLevel.TRACE):
            preview = response_preview[:max_length]
            if len(response_preview) > max_length:
                preview += "..."
            self.trace(f"LLM response preview: {preview}")

    def parsing(self, format_type: str, success: bool, details: str = ""):
        """Log parsing operation result"""
        if success:
            msg = f"🔍 Parsed {format_type}"
            if details:
                msg += f": {details}"
            self.debug(msg)
        else:
            self.warn(f"Failed to parse {format_type}" + (f": {details}" if details else ""))

    def file_created(self, filepath: str, file_type: str = "file"):
        """Log file creation"""
        filename = Path(filepath).name
        self.info(f"📝 Created {file_type}: {filename}")

    def file_saved(self, filepath: str, size_bytes: Optional[int] = None):
        """Log file save with optional size"""
        filename = Path(filepath).name
        size_str = f" ({size_bytes} bytes)" if size_bytes else ""
        self.debug(f"💾 Saved: {filename}{size_str}")

    def config_generated(self, config_type: str, steps: int, details: str = ""):
        """Log configuration generation"""
        msg = f"⚙️  Generated {config_type} config ({steps} steps)"
        if details:
            msg += f": {details}"
        self.info(msg)

    def validation(self, item: str, passed: bool, details: str = ""):
        """Log validation result"""
        if passed:
            msg = f"✓ Validated {item}"
            if details:
                msg += f": {details}"
            self.debug(msg)
        else:
            msg = f"✗ Validation failed: {item}"
            if details:
                msg += f": {details}"
            self.warn(msg)

    # ==================== Progress Tracking ====================

    def progress_start(self, batch_id: str, total: int, description: str = ""):
        """Initialize progress tracking for batch operation"""
        with self._progress_lock:
            self._progress_state[batch_id] = {
                'total': total,
                'current': 0,
                'description': description,
                'start_time': time.time()
            }

        self.info(f"🔄 Starting batch: {description} ({total} items)")

    def progress_update(self, batch_id: str, increment: int = 1):
        """Update progress counter"""
        with self._progress_lock:
            if batch_id not in self._progress_state:
                return

            state = self._progress_state[batch_id]
            state['current'] += increment
            current = state['current']
            total = state['total']
            desc = state['description']

            # Calculate percentage and ETA
            pct = int(current / total * 100) if total > 0 else 0
            elapsed = time.time() - state['start_time']
            eta = (elapsed / current * (total - current)) if current > 0 else 0

            # Log progress (only at INFO level, and only for significant updates)
            if pct % 10 == 0 or current == total:
                eta_str = f", ETA {eta:.1f}s" if eta > 0 and current < total else ""
                self.info(f"Progress: {current}/{total} ({pct}%){eta_str} - {desc}")

    def progress_end(self, batch_id: str, success: bool = True):
        """Finalize progress tracking"""
        with self._progress_lock:
            if batch_id not in self._progress_state:
                return

            state = self._progress_state[batch_id]
            duration = time.time() - state['start_time']
            total = state['total']
            desc = state['description']

            del self._progress_state[batch_id]

        if success:
            self.info(f"✅ Batch complete: {desc} ({total} items, {duration:.2f}s)")
        else:
            self.error(f"❌ Batch failed: {desc}")


class TaskContext:
    """
    Context manager for automatic task tracking and timing.

    Usage:
        logger = get_logger()
        with TaskContext("chrome_bookmarks_123", "chrome", "setup"):
            logger.info("Doing work...")
            # Automatic timing and context cleanup
    """

    def __init__(
        self,
        task_id: str,
        module: str,
        stage: str,
        logger: Optional[PipelineLogger] = None
    ):
        self.task_id = task_id
        self.module = module
        self.stage = stage
        self.logger = logger or get_logger()
        self.success = False

    def __enter__(self):
        self.logger.task_start(self.task_id, self.module, self.stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        self.logger.task_end(self.task_id, self.success)
        return False  # Don't suppress exceptions


# ==================== Global Logger Singleton ====================

_global_logger: Optional[PipelineLogger] = None


def get_logger() -> PipelineLogger:
    """Get global logger instance (lazy initialization)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PipelineLogger()
    return _global_logger


def init_logger(
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[Path] = None,
    use_colors: bool = True,
    show_timestamp: bool = True,
    show_context: bool = True
) -> PipelineLogger:
    """
    Initialize global logger with configuration.

    Call this once at pipeline start, then use get_logger() or convenience
    functions (info, debug, etc.) throughout the codebase.
    """
    global _global_logger
    _global_logger = PipelineLogger(
        level=level,
        log_file=log_file,
        use_colors=use_colors,
        show_timestamp=show_timestamp,
        show_context=show_context
    )
    return _global_logger


# ==================== Convenience Functions ====================

def error(message: str):
    """Log error using global logger"""
    get_logger().error(message)


def warn(message: str):
    """Log warning using global logger"""
    get_logger().warn(message)


def info(message: str):
    """Log info using global logger"""
    get_logger().info(message)


def debug(message: str):
    """Log debug using global logger"""
    get_logger().debug(message)


def trace(message: str):
    """Log trace using global logger"""
    get_logger().trace(message)
