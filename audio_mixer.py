import os
try:
    from pydub import AudioSegment
except ImportError:
    print("Warning: pydub not installed. Audio mixing will be skipped.")
    AudioSegment = None

def mix_audio(vocals_path, instrumental_path, output_path):
    """
    Mixes the converted vocals with the instrumental track.
    """
    if AudioSegment is None:
        return None

    if not os.path.exists(vocals_path):
        print(f"Error: Vocals file not found: {vocals_path}")
        return None
        
    if not os.path.exists(instrumental_path):
        print(f"Error: Instrumental file not found: {instrumental_path}")
        return None

    print(f"---> Mixing Audio...")
    print(f"     Vocals: {vocals_path}")
    print(f"     Instrumental: {instrumental_path}")

    try:
        vocals = AudioSegment.from_file(vocals_path)
        instrumental = AudioSegment.from_file(instrumental_path)

        # Overlay vocals on instrumental
        # Assuming they start at the same time (which they should for this pipeline)
        combined = instrumental.overlay(vocals)

        # Export result
        combined.export(output_path, format="wav")
        print(f"---> Mixing complete. Saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during mixing: {e}")
        print("Note: pydub requires ffmpeg. If it's not in PATH, this might fail.")
        return None

if __name__ == "__main__":
    # Test block
    pass
