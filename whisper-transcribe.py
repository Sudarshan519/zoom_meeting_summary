import whisper
import torch
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


model = whisper.load_model("medium",device='cpu')
# if torch.backends.mps.is_available():
#     model = model.to("mps")  # Move model to MPS GPU
# else:
# model = model.to("cpu")
result = model.transcribe("temp_9pX_h3n54d8d98CNAAAH.wav",)
print(result["text"])
# import torch
# print(torch.backends.mps.is_available())
# print(torch.backends.mps.is_built())