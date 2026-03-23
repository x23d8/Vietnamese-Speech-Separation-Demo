import gradio as gr
import torch
import datetime
import time
import tempfile
import shutil
import soundfile as sf
import os

# giả sử bạn đã implement 2 class này
from models.conv_tasnet.model import ConvTasnet
from models.sepformer.model import Sepformer

# =========================
# GLOBAL STATE
# =========================
current_model = None
current_model_name = None
history = []

# =========================
# MODEL FACTORY
# =========================
def get_model(model_name):
    if model_name == "ConvTasNet":
        return ConvTasnet(checkpoint_path=r"checkpoints\convtasnet\convtasnet_best.pth")
    elif model_name == "SepFormer":
        return Sepformer(checkpoint_path=r"checkpoints\sepformer")
    else:
        raise ValueError("Unknown model")


# =========================
# LOAD MODEL
# =========================
def load_model_if_needed(model_name):
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        return f"✅ {model_name} is already loaded."

    # unload cũ
    if current_model is not None:
        try:
            current_model.unload_model()
        except:
            pass
        del current_model
        torch.cuda.empty_cache()

    # load mới
    print(f"--- Loading {model_name} ---")
    current_model = get_model(model_name)
    current_model.load_model()

    current_model_name = model_name
    return f"🚀 Successfully loaded {model_name}!"


# =========================
# MIX AUDIO
# =========================
def mix_audios(path1, path2, target_sr=16000):
    wav1, sr1 = sf.read(path1)
    wav2, sr2 = sf.read(path2)

    # mono
    if len(wav1.shape) > 1:
        wav1 = wav1.mean(axis=1)
    if len(wav2.shape) > 1:
        wav2 = wav2.mean(axis=1)

    # resample nếu cần
    if sr1 != target_sr or sr2 != target_sr:
        import librosa
        if sr1 != target_sr:
            wav1 = librosa.resample(wav1, orig_sr=sr1, target_sr=target_sr)
        if sr2 != target_sr:
            wav2 = librosa.resample(wav2, orig_sr=sr2, target_sr=target_sr)

    wav1 = torch.tensor(wav1)
    wav2 = torch.tensor(wav2)

    # padding
    max_len = max(len(wav1), len(wav2))
    wav1 = torch.nn.functional.pad(wav1, (0, max_len - len(wav1)))
    wav2 = torch.nn.functional.pad(wav2, (0, max_len - len(wav2)))

    # mix
    mixture = wav1 + wav2

    # normalize
    mixture = mixture / (mixture.abs().max() + 1e-8)

    return mixture.numpy(), target_sr


# =========================
# INFERENCE
# =========================
def run_inference(model_name, audio_input_1, audio_input_2):
    global history

    if audio_input_1 is None:
        return None, None, "❌ No input audio provided.", gr.update()

    status = load_model_if_needed(model_name)
    start_time = time.time()

    # =========================
    # MIX nếu có 2 audio
    # =========================
    if audio_input_2 is not None:
        mixture, sr = mix_audios(audio_input_1, audio_input_2)
        saved_input_path = tempfile.mktemp(suffix="_mixed.wav")
        sf.write(saved_input_path, mixture, sr)
    else:
        saved_input_path = tempfile.mktemp(suffix="_input.wav")
        shutil.copy(audio_input_1, saved_input_path)

    # =========================
    # INFER
    # =========================
    wav1, wav2 = current_model.infer(saved_input_path)

    wav1 = wav1.detach().cpu().numpy()
    wav2 = wav2.detach().cpu().numpy()

    sr = current_model.sample_rate

    # save output
    out1_path = tempfile.mktemp(suffix="_spk1.wav")
    out2_path = tempfile.mktemp(suffix="_spk2.wav")

    sf.write(out1_path, wav1, sr)
    sf.write(out2_path, wav2, sr)

    infer_time = time.time() - start_time
    time_str = datetime.datetime.now().strftime("%H:%M:%S")

    display_name = f"[{time_str}] {model_name} ({infer_time:.2f}s)"

    record = {
        "display_name": display_name,
        "model": model_name,
        "input": saved_input_path,
        "out1": out1_path,
        "out2": out2_path,
        "infer_time": infer_time
    }
    history.append(record)

    history_display = [h['display_name'] for h in history]

    final_status = f"{status} | Done in {infer_time:.2f}s"

    return out1_path, out2_path, final_status, gr.update(
        choices=history_display,
        value=display_name
    )


# =========================
# HISTORY
# =========================
def load_from_history(selected_history):
    if not selected_history:
        return gr.update(), None, None, None, None, "No history selected."

    for record in history:
        if record["display_name"] == selected_history:
            return (
                record["model"],
                record["input"],
                None,
                record["out1"],
                record["out2"],
                f"📂 Loaded: {selected_history}"
            )

    return gr.update(), None, None, None, None, "Not found."


# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Speech Separation Demo")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            ["ConvTasNet", "SepFormer"],
            value="ConvTasNet",
            label="Select Model"
        )
        status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        audio_input_1 = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio 1"
        )

        audio_input_2 = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio 2 (optional)"
        )

    run_btn = gr.Button("Run Separation", variant="primary")

    with gr.Row():
        out1 = gr.Audio(label="Speaker 1")
        out2 = gr.Audio(label="Speaker 2")

    history_box = gr.Radio(
        choices=[],
        label="History",
        interactive=True
    )

    # =========================
    # EVENTS
    # =========================
    model_dropdown.change(
        fn=load_model_if_needed,
        inputs=[model_dropdown],
        outputs=[status_box]
    )

    run_btn.click(
        fn=run_inference,
        inputs=[model_dropdown, audio_input_1, audio_input_2],
        outputs=[out1, out2, status_box, history_box]
    )

    history_box.change(
        fn=load_from_history,
        inputs=[history_box],
        outputs=[model_dropdown, audio_input_1, audio_input_2, out1, out2, status_box]
    )

    demo.load(
        fn=load_model_if_needed,
        inputs=[model_dropdown],
        outputs=[status_box]
    )

demo.launch()
