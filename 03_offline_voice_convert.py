"""
Offline voice conversion wrapper for VelvetVinyl.

Goal (v1):
- Input: a vocal-only WAV file path (e.g., output of 01_separator.py / Demucs)
- Choice: target_voice (currently only "kishore")
- Output: a converted WAV written into RVC_Backend/audios/ (or a user-specified path)

This script intentionally reuses the official RVC CLI inference entrypoint:
`RVC_Backend/tools/infer_cli.py`
so we don't re-implement the RVC pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


VOICE_PRESETS = {
    # target_voice -> (model_filename, index_relative_path)
    "kishore": ("KishoreKumar.pth", os.path.join("weights", "kishore.index")),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RVC voice conversion (file-based).")
    p.add_argument("--input_vocals_path", required=True, help="Path to vocal-only WAV input")
    p.add_argument(
        "--target_voice",
        default="kishore",
        choices=sorted(VOICE_PRESETS.keys()),
        help="Target voice preset (v1 only supports kishore)",
    )
    p.add_argument(
        "--output_path",
        default=None,
        help="Where to write converted WAV. Default: RVC_Backend/audios/vocals_converted.wav",
    )

    # expose a few useful knobs (safe defaults)
    p.add_argument("--f0up_key", type=int, default=0, help="Pitch shift in semitones")
    # Default to "harvest" because some repos don't ship rmvpe.pt by default.
    p.add_argument("--f0method", type=str, default="harvest", help="e.g. rmvpe/harvest/pm")
    p.add_argument("--index_rate", type=float, default=0.66, help="RVC retrieval index rate")
    p.add_argument("--filter_radius", type=int, default=3)
    p.add_argument("--resample_sr", type=int, default=0)
    p.add_argument("--rms_mix_rate", type=float, default=1.0)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--device", type=str, default=None, help='Override device, e.g. "cuda:0"')
    p.add_argument(
        "--is_half",
        type=int,
        default=None,
        help="Override half precision: 1=True, 0=False (default: backend config)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    rvc_root = repo_root / "RVC_Backend"
    # Ensure the RVC backend package root is importable even when this script is run from repo root.
    sys.path.insert(0, str(rvc_root))

    input_vocals = Path(args.input_vocals_path).expanduser().resolve()
    if not input_vocals.exists():
        print(f"[error] input_vocals_path not found: {input_vocals}", file=sys.stderr)
        return 2

    model_filename, index_rel = VOICE_PRESETS[args.target_voice]
    model_name = model_filename  # infer_cli expects filename under weight_root
    index_path = (rvc_root / index_rel).resolve()
    if not index_path.exists():
        print(f"[error] index file not found: {index_path}", file=sys.stderr)
        return 2

    default_out = (rvc_root / "audios" / "vocals_converted.wav").resolve()
    out_path = Path(args.output_path).expanduser().resolve() if args.output_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # VC.get_vc() uses os.getenv("weight_root") to locate the model file.
    # We set it explicitly to RVC_Backend/assets/weights so infer_cli can find the model.
    weight_root = (rvc_root / "assets" / "weights").resolve()
    model_path = weight_root / model_filename
    if not model_path.exists():
        print(f"[error] model file not found: {model_path}", file=sys.stderr)
        return 2
    os.environ["weight_root"] = str(weight_root)

    # infer_cli.py relies on being executed with CWD = RVC_Backend (relative configs paths).
    os.chdir(str(rvc_root))

    # Import after chdir so relative sys.path behavior matches how infer_cli is normally used.
    from tools import infer_cli  # type: ignore

    # Build argv for infer_cli.py and execute its main().
    # Note: infer_cli.arg_parse() resets sys.argv back to sys.argv[:1], so we must set it first.
    argv = [
        "infer_cli.py",
        "--input_path",
        str(input_vocals),
        "--index_path",
        str(index_path),
        "--opt_path",
        str(out_path),
        "--model_name",
        model_name,
        "--f0up_key",
        str(args.f0up_key),
        "--f0method",
        str(args.f0method),
        "--index_rate",
        str(args.index_rate),
        "--filter_radius",
        str(args.filter_radius),
        "--resample_sr",
        str(args.resample_sr),
        "--rms_mix_rate",
        str(args.rms_mix_rate),
        "--protect",
        str(args.protect),
    ]
    if args.device:
        argv += ["--device", args.device]
    if args.is_half is not None:
        # infer_cli uses type=bool (not ideal). We'll pass "True"/"False".
        argv += ["--is_half", "True" if int(args.is_half) == 1 else "False"]

    sys.argv = argv
    infer_cli.main()

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

