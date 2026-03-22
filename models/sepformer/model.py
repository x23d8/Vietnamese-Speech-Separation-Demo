from ..model_base import BaseSpeechSeparationModel
import torch
import torchaudio
import os, sys
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F
import speechbrain.utils.seed

class Sepformer(BaseSpeechSeparationModel):
    def __init__(self, model_name="SepFormer", checkpoint_path=None):
        super().__init__(model_name = model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.masknet = None
        self.decoder = None
        self.num_spks = None
        self.sample_rate = None
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False

    def load_model(self):
        """Load the SepFormer model from a SpeechBrain YAML config and checkpoint."""

        # We only need a minimal set of YAML keys — supply a dummy data_folder
        # so the !PLACEHOLDER doesn't raise an error.
        overrides = "data_folder: ."

        with open(os.path.join(self.checkpoint_path, "hyperparams.yaml"), encoding="utf-8") as f:
            hparams = load_hyperpyyaml(f, overrides)

        encoder = hparams["Encoder"].to(self.device)
        masknet = hparams["MaskNet"].to(self.device)
        decoder = hparams["Decoder"].to(self.device)
        num_spks = hparams["num_spks"]
        sample_rate = hparams["sample_rate"]

        # Load checkpoint weights
        enc_ckpt = os.path.join(self.checkpoint_path, "encoder.ckpt")
        dec_ckpt = os.path.join(self.checkpoint_path, "decoder.ckpt")
        mask_ckpt = os.path.join(self.checkpoint_path, "masknet.ckpt")

        for path in (enc_ckpt, dec_ckpt, mask_ckpt):
            if not os.path.isfile(path):
                sys.exit(f"[ERROR] Checkpoint file not found: {path}")

        encoder.load_state_dict(torch.load(enc_ckpt, map_location=self.device))
        decoder.load_state_dict(torch.load(dec_ckpt, map_location=self.device))
        masknet.load_state_dict(torch.load(mask_ckpt, map_location=self.device))

        encoder.eval()
        masknet.eval()
        decoder.eval()

        self.encoder = encoder
        self.masknet = masknet
        self.decoder = decoder
        self.num_spks = num_spks
        self.sample_rate = sample_rate

        print(f"  ✔ Model loaded from: {self.checkpoint_path}")
        print(f"  ✔ num_spks={num_spks}, sample_rate={sample_rate}")

        self.is_loaded = True

    
    def preprocess(self, audio_path):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        return waveform


    def separate(self, processed_waveform):
        mix = processed_waveform.to(self.device)
        with torch.no_grad():
            # Encoder
            mix_w = self.encoder(mix)                           # [1, N, L]

            # Mask estimation
            est_mask = self.masknet(mix_w)                      # [num_spks, 1, N, L]

            # Apply masks
            mix_w = torch.stack([mix_w] * self.num_spks)        # [num_spks, 1, N, L]
            sep_h = mix_w * est_mask                       # [num_spks, 1, N, L]

            # Decoder
            est_sources = []
            for i in range(self.num_spks):
                est_source = self.decoder(sep_h[i])             # [1, T']
                est_sources.append(est_source)
            est_sources = torch.cat(est_sources, dim=0)    # [num_spks, T']

        # Fix length to match original
        T_origin = mix.size(1)
        T_est = est_sources.size(1)
        if T_origin > T_est:
            est_sources = F.pad(est_sources, (0, T_origin - T_est))
        else:
            est_sources = est_sources[:, :T_origin]

        return est_sources.cpu()
    
    def postprocess(self, outputs):
        processed_tensors = []
        for i in range(self.num_spks):
            # Normalization
            max_val = outputs[i].abs().max()
            if max_val > 0:
                outputs[i] = outputs[i] / max_val * 0.95
            
            # 3. FIX SHAPE: Squeeze để bỏ dimension 1, chỉ còn lại (T,)
            # Tránh lỗi sf.write hiểu lầm (1, T) là file có T channels
            processed_tensors.append(outputs[i].squeeze()) 
        
        # Trả về các mảng 1D
        return processed_tensors[0], processed_tensors[1]
    
    
    def unload_model(self):
        """Unload model from memory (CPU/GPU)."""

        if self.encoder is not None:
            del self.encoder
        if self.masknet is not None:
            del self.masknet
        if self.decoder is not None:
            del self.decoder

        self.encoder = None
        self.masknet = None
        self.decoder = None
        self.num_spks = None
        self.sample_rate = None

        # Clear GPU cache nếu dùng CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  ✔ Model unloaded successfully")


    



