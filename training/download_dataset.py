#!/usr/bin/env python3
"""
Dataset Downloader for Bangladesh Road Traffic Sign Dataset (BTSD)
Downloads dataset from Zenodo and organizes it for training
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

# Zenodo record information
ZENODO_RECORD_ID = "14969122"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

def download_file(url: str, destination: Path, desc: str = "Downloading"):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        desc: Description for progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return destination

def extract_archive(archive_path: Path, extract_to: Path):
    """
    Extract zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    print(f"📦 Extracting {archive_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"✅ Extracted to {extract_to}")

def get_dataset_info():
    """
    Get dataset information from Zenodo API.
    
    Returns:
        Dictionary with dataset metadata and file information
    """
    print(f"🔍 Fetching dataset info from Zenodo...")
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    return response.json()

def organize_dataset(source_dir: Path, target_dir: Path):
    """
    Organize downloaded dataset into expected structure.
    
    Args:
        source_dir: Directory with extracted files
        target_dir: Target directory (raw data directory)
    """
    print(f"📁 Organizing dataset...")
    
    # Find all image and label files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Search for images in source directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_dir.rglob(f'*{ext}'))
        image_files.extend(source_dir.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"⚠️  No images found in {source_dir}")
        print("📂 Directory structure:")
        for item in source_dir.rglob('*'):
            if item.is_file():
                print(f"   {item.relative_to(source_dir)}")
        return False
    
    print(f"📊 Found {len(image_files)} images")
    
    # Copy files to target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    
    for img_file in tqdm(image_files, desc="Copying files"):
        # Copy image
        dst_img = target_dir / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)
            copied_count += 1
        
        # Copy corresponding label if exists
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            dst_label = target_dir / label_file.name
            if not dst_label.exists():
                shutil.copy2(label_file, dst_label)
    
    print(f"✅ Copied {copied_count} images to {target_dir}")
    return True

def validate_dataset(data_dir: Path):
    """
    Validate the downloaded dataset.
    
    Args:
        data_dir: Directory with dataset
    """
    print(f"\n🔍 Validating dataset...")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(data_dir.glob(f'*{ext}'))
        images.extend(data_dir.glob(f'*{ext.upper()}'))
    
    labels = list(data_dir.glob('*.txt'))
    
    print(f"   Images: {len(images)}")
    print(f"   Labels: {len(labels)}")
    
    if len(images) == 0:
        print("❌ No images found!")
        return False
    
    # Check if labels exist for images
    labeled_count = 0
    for img in images:
        label_file = img.with_suffix('.txt')
        if label_file.exists():
            labeled_count += 1
    
    print(f"   Labeled images: {labeled_count} ({labeled_count/len(images)*100:.1f}%)")
    
    if labeled_count == 0:
        print("⚠️  Warning: No labels found for images!")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Download Bangladesh Road Traffic Sign Dataset from Zenodo'
    )
    parser.add_argument('--output-dir', type=str, 
                       default='../data/raw',
                       help='Output directory for dataset')
    parser.add_argument('--download-dir', type=str,
                       default='../data/downloads',
                       help='Directory for downloaded archives')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    download_dir = Path(args.download_dir).resolve()
    
    print("🚀 Bangladesh Road Traffic Sign Dataset Downloader")
    print("=" * 60)
    print(f"   Zenodo Record: {ZENODO_RECORD_ID}")
    print(f"   Output directory: {output_dir}")
    print(f"   Download directory: {download_dir}")
    print("=" * 60)
    
    try:
        # Get dataset information
        dataset_info = get_dataset_info()
        
        print(f"\n📋 Dataset: {dataset_info['metadata']['title']}")
        print(f"   DOI: {dataset_info['doi']}")
        print(f"   Published: {dataset_info['created'][:10]}")
        
        # Get files
        files = dataset_info['files']
        print(f"\n📦 Available files: {len(files)}")
        
        # Download all files
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        for file_info in files:
            filename = file_info['key']
            file_url = file_info['links']['self']
            file_size = file_info['size']
            
            download_path = download_dir / filename
            
            print(f"\n📥 File: {filename} ({file_size / 1024 / 1024:.1f} MB)")
            
            if download_path.exists() and not args.force:
                print(f"   ✓ Already downloaded, skipping...")
                downloaded_files.append(download_path)
                continue
            
            download_file(file_url, download_path, desc=f"Downloading {filename}")
            downloaded_files.append(download_path)
        
        # Extract archives
        extract_dir = download_dir / "extracted"
        
        for archive_file in downloaded_files:
            if archive_file.suffix in ['.zip', '.tar', '.gz', '.bz2', '.xz']:
                extract_archive(archive_file, extract_dir)
        
        # Organize dataset
        if organize_dataset(extract_dir, output_dir):
            # Validate
            validate_dataset(output_dir)
            
            print("\n✅ Dataset download and setup completed successfully!")
            print(f"📁 Dataset location: {output_dir}")
            print("\n📝 Next steps:")
            print("   1. Run data preprocessing:")
            print(f"      cd training")
            print(f"      python data_preprocessing.py --raw-dir {output_dir} --output-dir ../data/processed")
            print("   2. Start training:")
            print(f"      python train_yolov11.py --data ../data/processed/data.yaml")
        else:
            print("\n⚠️  Dataset organization failed. Check the extracted files.")
            return 1
        
        # Citation information
        print("\n📚 Citation:")
        print("   Please cite this dataset if you use it in your research:")
        if 'metadata' in dataset_info and 'creators' in dataset_info['metadata']:
            creators = dataset_info['metadata']['creators']
            author_names = ', '.join([c['name'] for c in creators])
            print(f"   {author_names}")
        print(f"   DOI: {dataset_info['doi']}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
