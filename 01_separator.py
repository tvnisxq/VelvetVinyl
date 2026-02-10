import os
import torch
from demucs import separate
import sys

def separate_audio(song_path, output_dir):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"---> Processing using: {device.upper()} (RTX 3060 should show CUDA)")

    # Demucs command construction
    model_name = "htdemucs"
    args = [
        "-n", model_name,
        "--two-stems", "vocals",
        "-d", device,
        "-o", output_dir,
        song_path
    ]

    print(f"--> Splitting: {os.path.basename(song_path)}... Please wait.")

    try:
        # Save original argv to restore later if needed, though mostly cosmetic here
        original_argv = sys.argv[:]
        sys.argv = ['demucs'] + args
        separate.main()
        sys.argv = original_argv
        
        print(f"--> Success! Output saved in {output_dir}")
        
        # Construct and return the expected vocal path
        song_base_name = os.path.splitext(os.path.basename(song_path))[0]
        vocals_path = os.path.join(output_dir, model_name, song_base_name, "vocals.wav")
        
        if os.path.exists(vocals_path):
            return os.path.abspath(vocals_path)
        else:
            print(f"Warning: Expected output file at {vocals_path} not found.")
            return None

    except Exception as e:
        print(f"--> Error: {e}")
        return None


if __name__ == "__main__":
    # Setup paths:
    project_root = os.getcwd()
    input_folder = os.path.join(project_root, "input_songs")
    output_folder = os.path.join(project_root, "separated_audio")

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Pick a song(Ensure to put a song with name test_song.mp3 in the input_songs folder)
    song_name = "Agar Tum Saath Ho.mp3"
    song_path = os.path.join(input_folder, song_name)

    if os.path.exists(song_path):
        separate_audio(song_path, output_folder)
    else:
        print(f"Please place '{song_name}' inside the 'input_songs' folder!")