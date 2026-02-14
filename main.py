import argparse
import os
import sys

# Import our modular scripts
# Make sure we can find them in the current directory
sys.path.append(os.getcwd())

import importlib.util

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load 01_separator.py as a module
separator = load_module_from_path("separator", "01_separator.py")
# Load voice_converter.py as a module
converter = load_module_from_path("converter", "voice_converter.py")
# Load audio_mixer.py
mixer = load_module_from_path("mixer", "audio_mixer.py")

def main():
    parser = argparse.ArgumentParser(description="Velvet Vinyl - AI Voice Conversion Pipeline")
    parser.add_argument("--input_song", required=True, help="Path to the input song (mp3/wav)")
    parser.add_argument("--target_voice", default="kishore", help="Target voice (kishore/lata/asha/rafi)")
    parser.add_argument("--output_dir", default="output", help="Directory for final output")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input_song)
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print("==========================================")
    print("      Velvet Vinyl - Unified Pipeline     ")
    print("==========================================")
    
    # Step 1: Separation
    print("\n[Step 1/3] Separating Vocals...")
    separation_output_dir = os.path.join(os.getcwd(), "separated_audio")
    
    # Using the refactored separate_audio function which returns the path
    vocals_path = separator.separate_audio(input_path, separation_output_dir)
    
    if not vocals_path or not os.path.exists(vocals_path):
        print("Error: Vocal separation failed or output file not found.")
        return

    # Demucs output structure: separated_audio/htdemucs/{song_name}/vocals.wav
    # We also need the instrumental track: no_vocals.wav
    song_dir = os.path.dirname(vocals_path)
    instrumental_path = os.path.join(song_dir, "no_vocals.wav")

    print(f"Vocals separated at: {vocals_path}")

    # Step 2: Conversion
    print(f"\n[Step 2/3] Converting Voice to {args.target_voice}...")
    
    # Define converted output path
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    converted_vocals_path = os.path.join(os.getcwd(), args.output_dir, f"{song_name}_vocals_{args.target_voice}.wav")
    
    try:
        # Renamed function in 03 script to generic name 'convert_vocals'
        # Pass target_voice
        converter.convert_vocals(
            input_vocals_path=vocals_path,
            target_voice=args.target_voice,
            output_path=converted_vocals_path
        )
    except Exception as e:
        print(f"\nError during conversion: {e}")
        return

    # Step 3: Mixing
    print("\n[Step 3/3] Mixing Audio...")
    final_output_path = os.path.join(os.getcwd(), args.output_dir, f"{song_name}_cover_{args.target_voice}.wav")
    
    try:
        mixed_path = mixer.mix_audio(converted_vocals_path, instrumental_path, final_output_path)
        if mixed_path:
            print("\n==========================================")
            print("          PIPELINE SUCCESS!               ")
            print("==========================================")
            print(f"Final Output: {mixed_path}")
        else:
             print("\nWarning: Mixing failed (likely due to missing ffmpeg).")
             print(f"Converted vocals available at: {converted_vocals_path}")
             print(f"Instrumental track at: {instrumental_path}")
             
    except Exception as e:
        print(f"Error during mixing: {e}")

if __name__ == "__main__":
    main()
