import urllib.request
from pathlib import Path
from tqdm import tqdm

URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
OUTPUT = Path(__file__).parent / "food-101.tar.gz"


def download():
    print("Downloading Food-101 dataset...")

    progress = tqdm(unit="B", unit_scale=True, desc="Downloading")

    def reporthook(block_num, block_size, total_size):
        if progress.total is None:
            progress.total = total_size
        downloaded = block_num * block_size
        progress.update(downloaded - progress.n)

    urllib.request.urlretrieve(URL, OUTPUT, reporthook)

    progress.close()
    print("Download complete:", OUTPUT)


if __name__ == "__main__":
    download()