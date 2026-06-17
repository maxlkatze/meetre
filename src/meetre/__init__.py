"""meetre — record macOS audio and produce meeting transcripts with speaker association."""

__version__ = "0.1.0"


def _silence_dependency_noise() -> None:
    """Quiet the (non-actionable) warning/log spam from torch, torchaudio,
    speechbrain, pyannote and HuggingFace so the CLI stays readable.

    Runs at import, before those libraries are loaded, so env-var switches
    (e.g. the HF download progress bars) take effect.
    """
    import logging
    import os
    import warnings

    # No tqdm "Fetching N files" bars; no tokenizer fork warnings.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    for category in (UserWarning, DeprecationWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=category)
    # urllib3's NotOpenSSLWarning uses its own category; match by message.
    warnings.filterwarnings("ignore", message=".*OpenSSL.*")

    for name in ("speechbrain", "pyannote", "torchaudio", "lightning",
                 "pytorch_lightning", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


_silence_dependency_noise()
