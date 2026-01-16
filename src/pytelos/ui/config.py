"""UI configuration constants.

Centralizes magic numbers and configuration values for the UI module.
"""


class LogLevel:
    """Log level constants with numeric values for comparison.

    Standard logging hierarchy: DEBUG < INFO < WARNING < ERROR
    Lower numeric value = more verbose (shows more messages).
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

    _names = {
        DEBUG: "DEBUG",
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
    }

    _from_string = {
        "debug": DEBUG,
        "info": INFO,
        "warning": WARNING,
        "error": ERROR,
    }

    @classmethod
    def name(cls, level: int) -> str:
        """Get the name for a log level."""
        return cls._names.get(level, "UNKNOWN")

    @classmethod
    def from_string(cls, level_str: str) -> int:
        """Convert string to log level. Returns DEBUG if invalid."""
        return cls._from_string.get(level_str.lower(), cls.DEBUG)


# Streaming configuration
STREAM_BUFFER_THRESHOLD = 50  # Characters before flushing stream buffer

# Input history configuration
INPUT_HISTORY_MAX_SIZE = 100  # Maximum entries in input history

# Output truncation limits
MAX_TOOL_RESULT_LENGTH = 2000  # Characters before truncating tool results
MAX_STDOUT_LENGTH = 10000  # Characters before truncating stdout

# Log panel configuration
LOG_TIMESTAMP_FORMAT = "%H:%M:%S"
LOG_MAX_MESSAGE_LENGTH = 500  # Characters before truncating log messages

# Metrics panel configuration
METRICS_TOKEN_SEPARATOR = "/"  # Separator between input/output tokens

# Chat display configuration
CHAT_MESSAGE_MAX_PREVIEW = 500  # Characters before truncating in preview

# Agent modes
AGENT_MODE_SIMPLE = "simple"  # Single-step RAG with DocumentAgent
AGENT_MODE_REASONING = "reasoning"  # Multi-step with ReasoningAgent
