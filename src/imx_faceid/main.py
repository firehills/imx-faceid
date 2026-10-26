"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from imx_faceid.config import AppConfig, ConfigurationError, load_config
from imx_faceid.inference import EmbeddingModel
from imx_faceid.service import FaceIdService
from imx_faceid.vector_store import VectorStore


def _log_level(debug: str) -> int:
    if debug == "none":
        return logging.CRITICAL + 1
    if debug == "error":
        return logging.ERROR
    if debug == "info":
        return logging.INFO
    return logging.DEBUG


async def _serve(config: AppConfig) -> None:
    model = EmbeddingModel(config.model)
    store = await asyncio.to_thread(
        VectorStore, config.qdrant, model.embedding_dim
    )
    service = FaceIdService(config, model, store)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    await service.run(stop)


def main() -> None:
    parser = argparse.ArgumentParser(description="NATS face identification service")
    parser.add_argument(
        "--config", default="config.yaml", help="YAML configuration path"
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        logging.basicConfig(
            level=_log_level(config.debug),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger(__name__).info("Debug mode: %s", config.debug)
        asyncio.run(_serve(config))
    except (ConfigurationError, OSError, RuntimeError, ValueError) as exc:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
