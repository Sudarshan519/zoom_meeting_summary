import whisper
import torch
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import torch
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())

model = whisper.load_model("medium")
# if torch.backends.mps.is_available():
#     model = model.to("mps")  # Move model to MPS GPU
# else:
# model = model.to("cpu")
result = model.transcribe("temp_s8fpu4Dhfg0xFSmIAAAB_1745936169.wav",)
print(result["text"])
# import torch
# print(torch.backends.mps.is_available())
# print(torch.backends.mps.is_built())