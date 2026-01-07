"""
Download TREC Deep Learning Track datasets (2019 & 2020)
"""
import os
import urllib.request
import sys

def download_file(url, output_path):
    """Download a file from URL to output path."""
    try:
        print(f"  Downloading: {os.path.basename(output_path)}")
        urllib.request.urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  [OK] Downloaded: {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def download_trec_dl_2019():
    """Download TREC-DL 2019 dataset."""
    print("\n=== TREC Deep Learning 2019 ===")
    
    # Alternative URLs from GitHub and official sources
    files = {
        "2019qrels-pass.txt": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        "msmarco-test2019-queries.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv",
        "msmarco-passagetest2019-top1000.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv",
    }
    
    output_dir = "data/raw/trec_dl/2019"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for filename, url in files.items():
        output_path = os.path.join(output_dir, filename)
        if download_file(url, output_path):
            success_count += 1
    
    print(f"\n2019: {success_count}/{len(files)} files downloaded successfully")
    return success_count == len(files)

def download_trec_dl_2020():
    """Download TREC-DL 2020 dataset."""
    print("\n=== TREC Deep Learning 2020 ===")
    
    # Alternative URLs from official sources
    files = {
        "2020qrels-pass.txt": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        "msmarco-test2020-queries.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv",
        "msmarco-passagetest2020-top1000.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv",
    }
    
    output_dir = "data/raw/trec_dl/2020"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for filename, url in files.items():
        output_path = os.path.join(output_dir, filename)
        if download_file(url, output_path):
            success_count += 1
    
    print(f"\n2020: {success_count}/{len(files)} files downloaded successfully")
    return success_count == len(files)

def main():
    """Main download function."""
    print("=" * 60)
    print("TREC Deep Learning Track Dataset Downloader")
    print("=" * 60)
    
    # Create base directory
    os.makedirs("data/raw/trec_dl", exist_ok=True)
    
    # Download both years
    success_2019 = download_trec_dl_2019()
    success_2020 = download_trec_dl_2020()
    
    print("\n" + "=" * 60)
    if success_2019 and success_2020:
        print("SUCCESS! All TREC-DL datasets downloaded!")
    else:
        print("WARNING! Some files failed to download.")
        print("You may need to download them manually from:")
        print("  - https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019")
        print("  - https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020")
    print("=" * 60)
    
    # Show directory structure
    print("\nDownloaded files:")
    for root, dirs, files in os.walk("data/raw/trec_dl"):
        level = root.replace("data/raw/trec_dl", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()

