from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from durian_leaf_disease.training.trainer import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện một model hoặc toàn bộ các model được cấu hình."
    )
    parser.add_argument(
        "--model",
        choices=["all", "mobilenet_v2", "efficientnet_b0", "resnet50"],
        default="all",
        help="Model cần huấn luyện, mặc định là all.",
    )
    args = parser.parse_args()
    main(model_names=None if args.model == "all" else [args.model])
