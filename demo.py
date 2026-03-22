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
history = [] # Sẽ lưu list các dictionary chứa thông tin history


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
# LOAD MODEL (Triggered when changing Dropdown)
# =========================
def load_model_if_needed(model_name):
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        return f"✅ {model_name} is already loaded."

    # Unload cũ
    if current_model is not None:
        try:
            current_model.unload_model()
        except:
            pass
        del current_model
        torch.cuda.empty_cache()

    # Load mới
    print(f"--- Loading {model_name} ---")
    current_model = get_model(model_name)
    
    # Gọi hàm load_model và đảm bảo nó thực thi
    current_model.load_model()
    
    # MẸO: Nếu class của bạn có biến kiểm tra, hãy ép nó thành True nếu log báo đã load xong
    # current_model.is_loaded = True 
    
    current_model_name = model_name
    return f"🚀 Successfully loaded {model_name}!"


# =========================
# INFERENCE
# =========================
def run_inference(model_name, audio_input_path):
    global history

    if audio_input_path is None:
        return None, None, "❌ No input audio provided.", gr.update()

    # Đảm bảo model đã được load (phòng trường hợp user chưa đổi dropdown mà ấn run luôn)
    status = load_model_if_needed(model_name)

    # Bắt đầu đo thời gian
    start_time = time.time()

    # Lưu vĩnh viễn file input vào thư mục temp của hệ thống để làm history
    # (Vì Gradio có thể tự xóa temp file ban đầu của nó)
    saved_input_path = tempfile.mktemp(suffix="_input.wav")
    shutil.copy(audio_input_path, saved_input_path)

    # inference (Truyền trực tiếp filepath vào nếu hàm infer của bạn hỗ trợ)
    wav1, wav2 = current_model.infer(saved_input_path)

    # đảm bảo tensor -> cpu numpy
    wav1 = wav1.detach().cpu().numpy()
    wav2 = wav2.detach().cpu().numpy()

    sr = current_model.sample_rate 

    # save output audio
    out1_path = tempfile.mktemp(suffix="_spk1.wav")
    out2_path = tempfile.mktemp(suffix="_spk2.wav")

    sf.write(out1_path, wav1, sr)
    sf.write(out2_path, wav2, sr)
    
    # Tính thời gian chạy
    infer_time = time.time() - start_time
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Tạo chuỗi hiển thị cho History
    display_name = f"[{time_str}] {model_name} (Infer: {infer_time:.2f}s)"

    # Lưu vào history
    record = {
        "display_name": display_name,
        "model": model_name,
        "input": saved_input_path,
        "out1": out1_path,
        "out2": out2_path,
        "infer_time": infer_time
    }
    history.append(record)

    # Cập nhật list cho Radio Button
    history_display = [h['display_name'] for h in history]
    
    final_status = f"{status} | Inference completed in {infer_time:.2f}s."

    # Trả về output paths, status và update lại History box
    return out1_path, out2_path, final_status, gr.update(choices=history_display, value=display_name)


# =========================
# HISTORY PLAYBACK
# =========================
def load_from_history(selected_history):
    if not selected_history:
        return gr.update(), None, None, None, "No history selected."
        
    for record in history:
        if record["display_name"] == selected_history:
            return (
                record["model"], # Update Dropdown model
                record["input"], # Update Input Audio
                record["out1"],  # Update Speaker 1
                record["out2"],  # Update Speaker 2
                f"📂 Loaded history: {selected_history}"
            )
            
    return gr.update(), None, None, None, "History record not found."


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

    # Chuyển type sang filepath để xử lý file nhẹ nhàng hơn
    audio_input = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath", 
        label="Input Audio"
    )

    run_btn = gr.Button("Run Separation", variant="primary")

    with gr.Row():
        out1 = gr.Audio(label="Speaker 1")
        out2 = gr.Audio(label="Speaker 2")

    history_box = gr.Radio(
        choices=[],
        label="Inference History (Click to reload inputs & outputs)",
        interactive=True
    )

    # =========================
    # EVENTS
    # =========================
    
    # 1. Bắt sự kiện khi Dropdown thay đổi -> Tự động load model
    model_dropdown.change(
        fn=load_model_if_needed,
        inputs=[model_dropdown],
        outputs=[status_box]
    )

    # 2. Bắt sự kiện khi ấn nút Run
    run_btn.click(
        fn=run_inference,
        inputs=[model_dropdown, audio_input],
        outputs=[out1, out2, status_box, history_box]
    )
    
    # 3. Bắt sự kiện khi người dùng click vào một item trong History
    history_box.change(
        fn=load_from_history,
        inputs=[history_box],
        outputs=[model_dropdown, audio_input, out1, out2, status_box]
    )
    
    # (Optional) Tự động load model mặc định khi vừa mở app
    demo.load(
        fn=load_model_if_needed,
        inputs=[model_dropdown],
        outputs=[status_box]
    )

demo.launch()