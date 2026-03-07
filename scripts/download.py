import urllib.request
from pathlib import Path
import tarfile
from tqdm import tqdm

URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVE = DATA_DIR / "food-101.tar.gz"

def download():
    print("Downloading Food-101 dataset...")

    progress = tqdm(unit="B", unit_scale=True, desc="Downloading")

    def reporthook(block_num, block_size, total_size):
        if progress.total is None:
            progress.total = total_size
        downloaded = block_num * block_size
        progress.update(downloaded - progress.n)

    urllib.request.urlretrieve(URL, ARCHIVE, reporthook)

    progress.close()
    print("Download complete:", ARCHIVE)

def extract():
    print("Extracting dataset...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=DATA_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    download()
    extract()