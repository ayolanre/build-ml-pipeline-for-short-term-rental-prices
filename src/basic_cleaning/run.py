#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact %s", args.input_artifact)
    download_root = os.path.join(
        "artifacts", args.input_artifact.replace(":", "_"))
    artifact_local_path = run.use_artifact(
        args.input_artifact).file(root=download_root)

    logger.info("Reading raw data")
    df = pd.read_csv(artifact_local_path)

    logger.info("Filtering price range between %s and %s",
                args.min_price, args.max_price)
    price_mask = df["price"].between(args.min_price, args.max_price)
    df = df.loc[price_mask].copy()

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Applying NYC geographic boundary filter")
    nyc_mask = df["longitude"].between(-74.25, -
                                       73.50) & df["latitude"].between(40.5, 41.2)
    df = df.loc[nyc_mask].copy()

    logger.info("Writing cleaned data to clean_sample.csv")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Uploading cleaned dataset to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified input artifact name",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Artifact type for the cleaned dataset",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned dataset artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted nightly price",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted nightly price",
        required=True,
    )

    parsed_args = parser.parse_args()

    go(parsed_args)
