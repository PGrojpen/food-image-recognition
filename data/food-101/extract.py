import tarfile
from pathlib import Path

ARCHIVE = Path(__file__).parent / "food-101.tar.gz"
OUTPUT = Path(__file__).parent

def extract():
    print("Extracting dataset...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(path=OUTPUT)
    print("Extraction complete.")

if __name__ == "__main__":
    extract()