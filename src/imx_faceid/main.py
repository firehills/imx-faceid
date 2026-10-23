"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from imx_faceid.config import ConfigurationError, load_config
from imx_faceid.inference import EmbeddingModel
from imx_faceid.service import FaceIdService
from imx_faceid.vector_store import VectorStore


async def _serve(config_path: str) -> None:
    config = load_config(config_path)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_serve(args.config))
    except (ConfigurationError, OSError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
