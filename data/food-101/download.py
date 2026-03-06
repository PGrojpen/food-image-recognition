import urllib.request
from pathlib import Path

URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
OUTPUT = Path(__file__).parent / "food-101.tar.gz"

def download():
    print("Downloading Food-101 dataset...")
    urllib.request.urlretrieve(URL, OUTPUT)
    print("Download complete:", OUTPUT)

if __name__ == "__main__":
    download()