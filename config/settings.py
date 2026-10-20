"""
Django settings for the Personalised RAG project.

Configuration is read from environment variables, which are loaded from
``.env.local`` (preferred, machine specific) and ``.env`` (shared defaults)
files in the project root.  No secret is ever hard coded here: the Groq API
key lives in the environment and is never returned to the browser.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_environment() -> None:
    """Load ``.env.local`` then ``.env`` into ``os.environ``.

    ``.env.local`` takes precedence because it holds machine specific values
    (for example the personal ``GROQ_API_KEY``).  Both files are optional so
    that a deployment can provide real environment variables instead.
    """
    env_file = BASE_DIR / ".env"
    local_env_file = BASE_DIR / ".env.local"

    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
    if local_env_file.exists():
        load_dotenv(dotenv_path=local_env_file, override=True)


_load_environment()


def env_str(name: str, default: str = "") -> str:
    """Return a stripped environment string."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean environment value."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Return an integer environment value, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Return a float environment value, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def env_path(name: str, default: str) -> Path:
    """Return a filesystem path environment value, resolved against BASE_DIR."""
    raw = env_str(name, default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def parse_path_list(name: str, default: list[str] | None = None) -> list[Path]:
    """Parse a list of filesystem roots.

    Accepted formats (in order of precedence):

    * a JSON array, e.g. ``["/home/me/docs", "/srv/shared"]``
    * a separator delimited string using the OS path separator
      (``:`` on macOS/Linux, ``;`` on Windows)
    * a single path
    """
    raw = env_str(name, "")
    if not raw:
        raw_values = list(default or [])
    elif raw.strip().startswith("["):
        try:
            parsed = json.loads(raw)
            raw_values = [str(item) for item in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            raw_values = []
    else:
        raw_values = [part for part in raw.split(os.pathsep) if part.strip()]

    roots: list[Path] = []
    for value in raw_values:
        candidate = Path(value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate not in roots:
            roots.append(candidate)
    return roots


# ---------------------------------------------------------------------------
# Core Django settings
# ---------------------------------------------------------------------------
SECRET_KEY = env_str("DJANGO_SECRET_KEY", "django-insecure-personalised-rag-local-development-key")
DEBUG = env_bool("DJANGO_DEBUG", True)

ALLOWED_HOSTS = [host.strip() for host in env_str("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rag_app",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = env_str("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"] if (BASE_DIR / "static").exists() else []
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Django REST framework
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "EXCEPTION_HANDLER": "rag_app.api.exceptions.api_exception_handler",
    "UNAUTHENTICATED_USER": None,
}

# ---------------------------------------------------------------------------
# Personalised RAG application settings
# ---------------------------------------------------------------------------

# -- LLM (Groq) -----------------------------------------------------------
GROQ_API_KEY = env_str("GROQ_API_KEY", "")
GROQ_MODEL = env_str("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TEMPERATURE = env_float("GROQ_TEMPERATURE", 0.2)
GROQ_MAX_TOKENS = env_int("GROQ_MAX_TOKENS", 1024)
GROQ_REQUEST_TIMEOUT = env_int("GROQ_REQUEST_TIMEOUT", 60)

# -- Embeddings (local, HuggingFace) --------------------------------------
EMBEDDING_MODEL = env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = env_str("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = env_int("EMBEDDING_BATCH_SIZE", 32)

# -- Vector store ----------------------------------------------------------
CHROMA_PERSIST_DIRECTORY = env_path("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
CHROMA_COLLECTION_PREFIX = env_str("CHROMA_COLLECTION_PREFIX", "kb")
CHROMA_SPACE = env_str("CHROMA_SPACE", "cosine")

# -- Sources ---------------------------------------------------------------
REPOSITORY_WORKSPACE = env_path("REPOSITORY_WORKSPACE", "./data/repositories")
ALLOWED_SOURCE_ROOTS = parse_path_list("ALLOWED_SOURCE_ROOTS", default=[str(Path.home())])
MAX_SOURCE_FILES = env_int("MAX_SOURCE_FILES", 5000)
MAX_FILE_SIZE_BYTES = env_int("MAX_FILE_SIZE_BYTES", 25 * 1024 * 1024)

IGNORED_DIRECTORIES = [
    name.strip()
    for name in env_str(
        "IGNORED_DIRECTORIES",
        ".git,.hg,.svn,.venv,venv,env,node_modules,__pycache__,.mypy_cache,.pytest_cache,"
        ".ruff_cache,.tox,.nox,.eggs,dist,build,out,target,site-packages,.idea,.vscode,"
        ".next,.nuxt,.gradle,.cache,coverage,htmlcov,.terraform,.parcel-cache",
    ).split(",")
    if name.strip()
]

SENSITIVE_FILE_NAMES = [
    name.strip().lower()
    for name in env_str(
        "SENSITIVE_FILE_NAMES",
        ".env,.env.local,.env.production,.env.development,.envrc,.netrc,.npmrc,.pypirc,"
        "credentials,credentials.json,secrets.json,secrets.yaml,secrets.yml,id_rsa,id_dsa,"
        "id_ecdsa,id_ed25519,known_hosts,keystore,master.key",
    ).split(",")
    if name.strip()
]

SENSITIVE_FILE_EXTENSIONS = [".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".asc", ".gpg"]

# -- Chunking --------------------------------------------------------------
CHUNK_SIZE = env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 150)

# -- Retrieval -------------------------------------------------------------
RETRIEVAL_K = env_int("RETRIEVAL_K", 5)
# Minimum cosine similarity (0..1) for a chunk to count as usable evidence.
# 0 disables the threshold and relies purely on "did we retrieve anything".
RETRIEVAL_SCORE_THRESHOLD = env_float("RETRIEVAL_SCORE_THRESHOLD", 0.15)
# Maximum number of conversations/messages returned by the list endpoints.
CONVERSATION_PAGE_SIZE = env_int("CONVERSATION_PAGE_SIZE", 50)
MAX_QUESTION_LENGTH = env_int("MAX_QUESTION_LENGTH", 2000)
MAX_CONVERSATION_TITLE_LENGTH = env_int("MAX_CONVERSATION_TITLE_LENGTH", 120)

# -- Ingestion -------------------------------------------------------------
INGESTION_MAX_RETRIES = env_int("INGESTION_MAX_RETRIES", 1)
INGESTION_RETRY_DELAY_SECONDS = env_float("INGESTION_RETRY_DELAY_SECONDS", 2.0)

# -- Logging ---------------------------------------------------------------
LOG_LEVEL = env_str("DJANGO_LOG_LEVEL", "INFO").upper()
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "rag_app": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}
