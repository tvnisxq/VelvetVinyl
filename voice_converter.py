import argparse
import os
import sys

from typing import Optional


VOICE_PRESETS = {
    "kishore": ("KishoreKumar.pth", "kishore.index"),
    "lata": ("LataMangeshkar.pth", "lata.index"),
    "asha": ("AshaBhosle.pth", "asha.index"),
    "rafi": ("MohammadRafi.pth", "rafi.index"),
}


def convert_vocals(
    input_vocals_path: str,
    target_voice: str = "kishore",
    output_path: Optional[str] = None,
    f0up_key: int = 0,
    device: Optional[str] = None,
) -> str:
    """
    Offline conversion helper that:
    - takes a separated vocal WAV file
    - runs it through the existing RVC backend
    - writes the converted vocals to a WAV file

    Returns the absolute path of the converted file.
    """

    # Resolve project paths BEFORE we change cwd
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_vocals_path = os.path.abspath(input_vocals_path)

    # Verify input file exists BEFORE changing directories
    if not os.path.isfile(input_vocals_path):
        raise FileNotFoundError(
            f"Input vocal file not found: {input_vocals_path}\n"
            f"Please ensure the file exists and the path is correct."
        )

    rvc_dir = os.path.join(project_root, "RVC_Backend")
    if not os.path.isdir(rvc_dir):
        raise RuntimeError(f"RVC_Backend directory not found at: {rvc_dir}")

    # Validate target voice
    if target_voice.lower() not in VOICE_PRESETS:
        supported = ", ".join(VOICE_PRESETS.keys())
        raise ValueError(
            f"Voice '{target_voice}' not supported. Available: {supported}"
        )

    # Set up environment variables expected by the RVC backend
    weight_root = os.path.join(rvc_dir, "assets", "weights")
    index_root = os.path.join(rvc_dir, "weights")
    os.environ.setdefault("weight_root", weight_root)
    os.environ.setdefault("index_root", index_root)

    # Model + index mapping
    model_filename, index_filename = VOICE_PRESETS[target_voice.lower()]
    model_name = model_filename
    index_path = os.path.join(index_root, index_filename)

    if not os.path.isfile(os.path.join(weight_root, model_name)):
        raise FileNotFoundError(
            f"Voice model not found: {os.path.join(weight_root, model_name)}"
        )
    if not os.path.isfile(index_path):
        print(f"Warning: Index file not found at {index_path}. Conversion may be less accurate.")
        # Some RVC implementations might work without index, but usually it's better to have it.
        # We will proceed but warn.
        # raise FileNotFoundError(f"Index file not found: {index_path}")

    # Default output: RVC_Backend/audios/vocals_converted.wav
    if output_path is None:
        audios_dir = os.path.join(rvc_dir, "audios")
        os.makedirs(audios_dir, exist_ok=True)
        output_path = os.path.join(audios_dir, f"vocals_converted_{target_voice}.wav")
    else:
        # Allow relative paths but store as absolute
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Store original cwd to restore later
    original_cwd = os.getcwd()
    
    # Change cwd so that RVC's relative paths (configs/, assets/, etc.) work
    os.chdir(rvc_dir)

    # Now import the RVC backend modules
    sys.path.append(rvc_dir)
    
    # Ensure input_vocals_path is still absolute (in case chdir affected it)
    if not os.path.isabs(input_vocals_path):
        input_vocals_path = os.path.abspath(input_vocals_path)

    from dotenv import load_dotenv  # type: ignore
    from scipy.io import wavfile  # type: ignore

    from configs.config import Config  # type: ignore
    from infer.modules.vc.modules import VC  # type: ignore

    load_dotenv()

    # Create config without consuming sys.argv
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        config = Config()
    finally:
        sys.argv = saved_argv
    
    if device:
        config.device = device

    # Initialize VC and load the target model
    vc = VC(config)
    vc.get_vc(model_name)

    print(f"---> Converting vocals to {target_voice} style...")
    print(f"     Input : {input_vocals_path}")
    print(f"     Model : {os.path.join(weight_root, model_name)}")
    print(f"     Index : {index_path}")

    # Parameters
    f0_method = "harvest"
    index_rate = 0.66
    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 1.0
    protect = 0.33

    print(f"---> Starting voice conversion (this may take a few minutes)...")
    
    try:
        _, wav_opt = vc.vc_single(
            0,
            input_vocals_path,
            f0up_key,
            None,
            f0_method,
            index_path,
            None,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
        )
    except Exception as e:
        os.chdir(original_cwd)
        raise RuntimeError(f"Conversion failed with error: {e}") from e

    if wav_opt is None or wav_opt[1] is None:
        os.chdir(original_cwd)
        raise RuntimeError("Conversion failed - RVC returned None.")
    
    wavfile.write(output_path, wav_opt[0], wav_opt[1])
    print(f"---> Conversion complete. Saved converted vocals to:\n     {output_path}")

    os.chdir(original_cwd)
    
    return os.path.abspath(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline RVC conversion (Generic).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input_vocals_path", type=str, required=True)
    parser.add_argument("--target_voice", type=str, default="kishore", choices=list(VOICE_PRESETS.keys()))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    try:
        output = convert_vocals(
            input_vocals_path=args.input_vocals_path,
            target_voice=args.target_voice,
            output_path=args.output_path,
            f0up_key=args.f0up_key,
            device=args.device,
        )
        print(f"\nFinal converted vocal file: {output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

