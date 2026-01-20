#!/usr/bin/env python3
# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download and prepare public recommendation datasets for evaluation.

Supports major benchmark datasets used in academic research:
- MovieLens (100K, 1M, 10M, 25M)
- Amazon Reviews (multiple categories)
- Yelp Dataset
- Netflix Prize (if available)
- Criteo CTR
- MIND (Microsoft News)
"""

import os
import urllib.request
import zipfile
import gzip
import tarfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and prepare recommendation datasets."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_movielens(self, size: str = "1m") -> Path:
        """Download MovieLens dataset.

        Args:
            size: "100k", "1m", "10m", or "25m"

        Returns:
            Path to extracted dataset directory
        """

        urls = {
            "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
            "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        }

        if size not in urls:
            raise ValueError(f"Invalid size: {size}. Choose from {list(urls.keys())}")

        url = urls[size]
        dataset_name = f"movielens-{size}"
        output_dir = self.data_dir / dataset_name

        if output_dir.exists():
            logger.info(f"{dataset_name} already exists at {output_dir}")
            return output_dir

        logger.info(f"Downloading {dataset_name} from {url}")

        zip_path = self.data_dir / f"{dataset_name}.zip"
        self._download_file(url, zip_path)

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # Find extracted directory
        extracted_dir = self.data_dir / f"ml-{size}"
        if extracted_dir.exists():
            extracted_dir.rename(output_dir)

        zip_path.unlink()  # Remove zip file

        logger.info(f"Dataset ready at {output_dir}")
        return output_dir

    def download_amazon_reviews(self, category: str = "Books") -> Path:
        """Download Amazon Reviews dataset.

        Args:
            category: Product category (e.g., "Books", "Electronics", "Movies_and_TV")

        Returns:
            Path to downloaded dataset
        """

        base_url = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall"
        url = f"{base_url}/{category}_5.json.gz"

        dataset_name = f"amazon-{category.lower()}"
        output_dir = self.data_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{category}_5.json"

        if output_file.exists():
            logger.info(f"{dataset_name} already exists at {output_file}")
            return output_dir

        logger.info(f"Downloading Amazon {category} reviews from {url}")

        gz_path = output_dir / f"{category}_5.json.gz"
        self._download_file(url, gz_path)

        logger.info(f"Extracting {gz_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())

        gz_path.unlink()

        logger.info(f"Dataset ready at {output_file}")
        return output_dir

    def download_mind(self, size: str = "small") -> Path:
        """Download MIND (Microsoft News) dataset.

        Args:
            size: "small", "large", or "demo"

        Returns:
            Path to extracted dataset directory
        """

        urls = {
            "demo": "https://mind201910small.blob.core.windows.net/release/MINDdemo_train.zip",
            "small": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "large": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
        }

        if size not in urls:
            raise ValueError(f"Invalid size: {size}. Choose from {list(urls.keys())}")

        url = urls[size]
        dataset_name = f"mind-{size}"
        output_dir = self.data_dir / dataset_name

        if output_dir.exists():
            logger.info(f"{dataset_name} already exists at {output_dir}")
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading MIND {size} from {url}")

        zip_path = self.data_dir / f"{dataset_name}.zip"
        self._download_file(url, zip_path)

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        zip_path.unlink()

        logger.info(f"Dataset ready at {output_dir}")
        return output_dir

    def download_criteo(self, sample: bool = True) -> Path:
        """Download Criteo CTR dataset.

        Args:
            sample: If True, download sample (1GB). If False, download full dataset (11GB)

        Returns:
            Path to downloaded dataset
        """

        if sample:
            url = "https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"
            dataset_name = "criteo-sample"
        else:
            logger.warning("Full Criteo dataset is 11GB. This may take a while.")
            url = "https://labs.criteo.com/wp-content/uploads/2015/04/dac.tar.gz"
            dataset_name = "criteo-full"

        output_dir = self.data_dir / dataset_name

        if output_dir.exists():
            logger.info(f"{dataset_name} already exists at {output_dir}")
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading Criteo dataset from {url}")

        tar_path = self.data_dir / f"{dataset_name}.tar.gz"
        self._download_file(url, tar_path)

        logger.info(f"Extracting {tar_path}")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)

        tar_path.unlink()

        logger.info(f"Dataset ready at {output_dir}")
        return output_dir

    def download_yelp(self) -> Path:
        """Download Yelp dataset.

        Note: Requires manual download from Yelp dataset challenge page.
        This provides instructions.

        Returns:
            Path where dataset should be placed
        """

        dataset_name = "yelp"
        output_dir = self.data_dir / dataset_name

        if output_dir.exists():
            logger.info(f"{dataset_name} already exists at {output_dir}")
            return output_dir

        logger.info(
            "\nYelp Dataset requires manual download:\n"
            "1. Visit: https://www.yelp.com/dataset\n"
            "2. Download the dataset\n"
            "3. Extract to: {}\n".format(output_dir)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def list_datasets(self) -> List[str]:
        """List available datasets."""

        if not self.data_dir.exists():
            return []

        datasets = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(datasets)

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a downloaded dataset."""

        dataset_dir = self.data_dir / dataset_name

        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_dir}")

        info = {
            "name": dataset_name,
            "path": str(dataset_dir),
            "files": [],
            "size_mb": 0,
        }

        # List files and compute size
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                info["files"].append(file_path.name)
                info["size_mb"] += file_path.stat().st_size / (1024 * 1024)

        info["size_mb"] = round(info["size_mb"], 2)
        info["num_files"] = len(info["files"])

        return info

    def _download_file(self, url: str, output_path: Path):
        """Download a file with progress reporting."""

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
            print()  # New line after progress
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            raise


def main():
    """Download recommended datasets for evaluation."""

    downloader = DatasetDownloader()

    print("=" * 60)
    print("Phoenix V2 Dataset Downloader")
    print("=" * 60)
    print("\nDownloading recommended evaluation datasets...\n")

    # Download MovieLens 1M (standard benchmark)
    print("\n[1/4] MovieLens 1M")
    print("-" * 60)
    downloader.download_movielens("1m")

    # Download Amazon Books (standard benchmark)
    print("\n[2/4] Amazon Books Reviews")
    print("-" * 60)
    try:
        downloader.download_amazon_reviews("Books")
    except Exception as e:
        logger.warning(f"Failed to download Amazon Books: {e}")

    # Download MIND small (news recommendations)
    print("\n[3/4] MIND Small (News Recommendations)")
    print("-" * 60)
    try:
        downloader.download_mind("small")
    except Exception as e:
        logger.warning(f"Failed to download MIND: {e}")

    # Download Criteo sample (CTR prediction)
    print("\n[4/4] Criteo Sample (CTR Prediction)")
    print("-" * 60)
    try:
        downloader.download_criteo(sample=True)
    except Exception as e:
        logger.warning(f"Failed to download Criteo: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    datasets = downloader.list_datasets()
    if datasets:
        print(f"\nDownloaded {len(datasets)} datasets:")
        for dataset in datasets:
            info = downloader.get_dataset_info(dataset)
            print(f"  - {dataset}: {info['num_files']} files, {info['size_mb']} MB")
    else:
        print("\nNo datasets downloaded.")

    print("\nDatasets are ready for evaluation!")
    print(f"Location: {downloader.data_dir.absolute()}\n")


if __name__ == "__main__":
    main()
