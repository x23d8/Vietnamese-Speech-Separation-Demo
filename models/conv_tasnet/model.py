from ..model_base import BaseSpeechSeparationModel
import torch
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
import torchaudio

class ConvTasnet(BaseSpeechSeparationModel):
    def __init__(self, model_name="ConvTasnet", checkpoint_path=None):
        super().__init__(model_name = model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path 
        self.sample_rate = 8000

    def load_model(self):
        bundle = CONVTASNET_BASE_LIBRI2MIX
        self.model = bundle.get_model()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
    
    def preprocess(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        return waveform

    def separate(self, processed_waveform):
        with torch.no_grad():
            separated_sources = self.model(processed_waveform)
        return separated_sources.squeeze(0)
    
    def postprocess(self, outputs):
        output1 = outputs[0].unsqueeze(0).cpu()
        output1 = output1 / output1.abs().max()
        output2 = outputs[1].unsqueeze(0).cpu()
        output2 = output2 / output2.abs().max()
        return output1.squeeze(0), output2.squeeze(0)
    
    def unload_model(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.is_loaded = False
        


    



