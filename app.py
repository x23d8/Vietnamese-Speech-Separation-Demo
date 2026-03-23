"""
app.py — Flask backend for the Speech Separation Web App
Standalone: does not depend on demo.py
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
import tempfile
import numpy as np
import soundfile as sf
import torch
import os
import time
import datetime
from pydub import AudioSegment

from models.conv_tasnet.model import ConvTasnet
from models.sepformer.model import Sepformer
SEPFORMER_CKPT     = r"checkpoints\sepformer_full_dynamicmixing"
SEPFORMER_OLD_CKPT = r"checkpoints\sepformer"
CONVTASNET_CKPT    = r"checkpoints\convtasnet\convtasnet_best.pth"

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Model state ───────────────────────────────────────────────────
current_model      = None   # ConvTasnet instance  OR  Sepformer instance
current_model_name = None
history            = []
mix_sources        = {}     # mix_path -> {person1: wav_path, person2: wav_path}


def load_model_if_needed(model_name: str) -> str:
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        return f"{model_name} already loaded."

    # Unload previous
    if current_model is not None:
        if hasattr(current_model, "unload_model"):
            try:
                current_model.unload_model()
            except Exception:
                pass
        del current_model
        torch.cuda.empty_cache()

    print(f"--- Loading {model_name} ---")

    if model_name == "ConvTasNet":
        current_model = ConvTasnet(checkpoint_path=CONVTASNET_CKPT)
    elif model_name == "SepFormer":
        current_model = Sepformer(checkpoint_path=SEPFORMER_CKPT)
    elif model_name == "SepFormer (old)":
        current_model = Sepformer(checkpoint_path=SEPFORMER_OLD_CKPT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    current_model.load_model()
    current_model_name = model_name
    return f"{model_name} loaded."


def ensure_model(model_name: str):
    load_model_if_needed(model_name)


# ── Helpers ──────────────────────────────────────────────────────

def to_wav(src_path: str) -> str:
    """Convert any audio format (webm, ogg, mp3, …) to a proper WAV file.
    Uses pydub + ffmpeg — identical to how Gradio handles audio uploads."""
    out = tempfile.mktemp(suffix=".wav")
    audio = AudioSegment.from_file(src_path)
    audio.export(out, format="wav")
    return out


def mix_wavs(path_a: str, path_b: str) -> str:
    wav_a, sr_a = sf.read(path_a, always_2d=False)
    wav_b, sr_b = sf.read(path_b, always_2d=False)

    if wav_a.ndim > 1: wav_a = wav_a.mean(axis=1)
    if wav_b.ndim > 1: wav_b = wav_b.mean(axis=1)

    if sr_a != sr_b:
        import resampy
        wav_b = resampy.resample(wav_b, sr_b, sr_a)

    n = max(len(wav_a), len(wav_b))
    wav_a = np.pad(wav_a, (0, n - len(wav_a)))
    wav_b = np.pad(wav_b, (0, n - len(wav_b)))

    eps = 1e-9
    rms_a = np.sqrt(np.mean(wav_a ** 2) + eps)
    rms_b = np.sqrt(np.mean(wav_b ** 2) + eps)
    wav_b = wav_b * (rms_a / rms_b)

    mix = wav_a + wav_b
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak

    out = tempfile.mktemp(suffix=".wav")
    sf.write(out, mix, sr_a)
    return out


def separate(audio_path: str, model_name: str):
    """Run inference and return (out1, out2, elapsed, label). Caller stores history."""
    ensure_model(model_name)
    t0 = time.time()

    wav1, wav2 = current_model.infer(audio_path)
    wav1 = wav1.detach().cpu().numpy()
    wav2 = wav2.detach().cpu().numpy()
    sr = current_model.sample_rate

    out1 = tempfile.mktemp(suffix="_spk1.wav")
    out2 = tempfile.mktemp(suffix="_spk2.wav")
    sf.write(out1, wav1, sr)
    sf.write(out2, wav2, sr)

    elapsed = time.time() - t0
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    label = f"[{ts}] {model_name} ({elapsed:.2f}s)"
    return out1, out2, elapsed, label


def audio_url(path):
    return f"/api/audio?path={path}" if path else None


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/separate/single", methods=["POST"])
def api_separate_single():
    """Mode 1: single mixed audio → separate."""
    model_name = request.form.get("model", "ConvTasNet")
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify(error="No audio file provided."), 400

    tmp = tempfile.mktemp(suffix=".webm")
    audio_file.save(tmp)

    try:
        tmp = to_wav(tmp)
        out1, out2, elapsed, label = separate(tmp, model_name)
    except Exception as e:
        return jsonify(error=str(e)), 500

    history.append({
        "display_name": label, "model": model_name, "mode": "single",
        "input": tmp, "out1": out1, "out2": out2, "infer_time": elapsed,
    })

    return jsonify(
        label=label,
        elapsed=round(elapsed, 2),
        input_url=audio_url(tmp),
        spk1=audio_url(out1),
        spk2=audio_url(out2),
    )


@app.route("/api/mix", methods=["POST"])
def api_mix():
    """Mode 2 step 1: mix two recordings."""
    file_a = request.files.get("person1")
    file_b = request.files.get("person2")
    if not file_a or not file_b:
        return jsonify(error="Both person1 and person2 audio are required."), 400

    tmp_a = tempfile.mktemp(suffix=".webm")
    tmp_b = tempfile.mktemp(suffix=".webm")
    file_a.save(tmp_a)
    file_b.save(tmp_b)

    try:
        tmp_a = to_wav(tmp_a)
        tmp_b = to_wav(tmp_b)
        mix_path = mix_wavs(tmp_a, tmp_b)
    except Exception as e:
        return jsonify(error=str(e)), 500

    mix_sources[mix_path] = {"person1": tmp_a, "person2": tmp_b}

    return jsonify(
        mix=audio_url(mix_path),
        mix_path=mix_path,
        person1_url=audio_url(tmp_a),
        person2_url=audio_url(tmp_b),
    )


@app.route("/api/separate/mix", methods=["POST"])
def api_separate_mix():
    """Mode 2 step 2: separate a pre-mixed file (path returned by /api/mix)."""
    model_name = request.form.get("model", "ConvTasNet")
    mix_path   = request.form.get("mix_path")
    if not mix_path or not os.path.exists(mix_path):
        return jsonify(error="Mix path not found. Please mix first."), 400

    try:
        out1, out2, elapsed, label = separate(mix_path, model_name)
    except Exception as e:
        return jsonify(error=str(e)), 500

    sources = mix_sources.get(mix_path, {})
    history.append({
        "display_name": label, "model": model_name, "mode": "mix",
        "input": mix_path, "out1": out1, "out2": out2, "infer_time": elapsed,
        "person1": sources.get("person1"), "person2": sources.get("person2"),
    })

    return jsonify(
        label=label,
        elapsed=round(elapsed, 2),
        person1_url=audio_url(sources.get("person1")),
        person2_url=audio_url(sources.get("person2")),
        mix_url=audio_url(mix_path),
        spk1=audio_url(out1),
        spk2=audio_url(out2),
    )


@app.route("/api/audio")
def api_audio():
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, mimetype="audio/wav")


@app.route("/api/history")
def api_history():
    out = []
    for h in history:
        rec = {
            "label":   h["display_name"],
            "model":   h["model"],
            "mode":    h.get("mode", "single"),
            "elapsed": h["infer_time"],
            "input_url": audio_url(h.get("input")),
            "spk1_url":  audio_url(h.get("out1")),
            "spk2_url":  audio_url(h.get("out2")),
        }
        if h.get("mode") == "mix":
            rec["person1_url"] = audio_url(h.get("person1"))
            rec["person2_url"] = audio_url(h.get("person2"))
            rec["mix_url"]     = audio_url(h.get("input"))
        out.append(rec)
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
