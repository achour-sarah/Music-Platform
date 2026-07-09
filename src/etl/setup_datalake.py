import os
import shutil

def setup():
    base_dir = r"c:\SDV\Music"
    datalake_dir = os.path.join(base_dir, "data", "datalake")
    
    # Define zones
    zones = ["bronze", "silver", "gold"]
    for zone in zones:
        path = os.path.join(datalake_dir, zone)
        os.makedirs(path, exist_ok=True)
        print(f"Created zone directory: {path}")

    # Paths to raw source data
    raw_metadata_src = os.path.join(base_dir, "data", "fma_metadata")
    raw_audio_src = os.path.join(base_dir, "data", "fma_small1", "fma_small")
    
    # Destination for raw data in Bronze
    bronze_dir = os.path.join(datalake_dir, "bronze")
    
    print("Populating Bronze zone...")
    # Copy metadata files
    metadata_files = [
        "genres.csv",
        "raw_tracks.csv",
        "raw_albums.csv",
        "raw_artists.csv"
    ]
    
    for f in metadata_files:
        src = os.path.join(raw_metadata_src, f)
        if not os.path.exists(src):
            # Try alternate filename (e.g. raw-tracks.csv instead of raw_tracks.csv)
            alt_f = f.replace("_", "-") if "_" in f else f.replace("-", "_")
            src = os.path.join(raw_metadata_src, alt_f)
            
        dst = os.path.join(bronze_dir, f)
        
        if os.path.exists(src):
            if not os.path.exists(dst):
                print(f"Copying {os.path.basename(src)} to Bronze as {f}...")
                shutil.copy2(src, dst)
            else:
                print(f"{f} already exists in Bronze.")
        else:
            # Fallback for tracks.csv if raw_tracks.csv is completely missing
            if f == "raw_tracks.csv":
                fallback_src = os.path.join(raw_metadata_src, "tracks.csv")
                if os.path.exists(fallback_src):
                    print("raw_tracks.csv not found, copying tracks.csv to Bronze as tracks.csv...")
                    shutil.copy2(fallback_src, os.path.join(bronze_dir, "tracks.csv"))
                    continue
            print(f"Warning: Source metadata file not found at {src}")

    # For audio files, to save disk space and processing time, we will create a symlink or copy a subset of folders
    # We will copy/link the audio folder structure or copy the first few directories (e.g., 000, 001, 002)
    bronze_audio_dir = os.path.join(bronze_dir, "audio")
    os.makedirs(bronze_audio_dir, exist_ok=True)
    
    if os.path.exists(raw_audio_src):
        subdirs = sorted([d for d in os.listdir(raw_audio_src) if os.path.isdir(os.path.join(raw_audio_src, d))])
        # We will symlink or copy the first 5 subdirectories to Bronze to represent the landing zone
        limit = 5
        print(f"Linking/copying first {limit} audio directories to Bronze...")
        for i, subdir in enumerate(subdirs[:limit]):
            src_sub = os.path.join(raw_audio_src, subdir)
            dst_sub = os.path.join(bronze_audio_dir, subdir)
            
            if not os.path.exists(dst_sub):
                try:
                    # Try creating a symbolic link (requires admin rights sometimes on Windows, fallback to directory junction or copying if fails)
                    os.symlink(src_sub, dst_sub, target_is_directory=True)
                    print(f"Created symlink for audio subdir {subdir}")
                except Exception as e:
                    # Fallback: Copy first few files
                    print(f"Symlink failed for {subdir} ({e}), falling back to copying folder...")
                    shutil.copytree(src_sub, dst_sub)
                    print(f"Copied audio subdir {subdir}")
            else:
                print(f"Audio subdir {subdir} already exists in Bronze.")
    else:
        print(f"Warning: Source audio directory not found at {raw_audio_src}")
        
    print("Datalake folder setup complete!")

if __name__ == "__main__":
    setup()
