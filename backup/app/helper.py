import os
import subprocess
import cv2
import torch
import base64
import numpy as np
import supervision as sv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2_video_predictor
from typing import Dict
import uvicorn
import nest_asyncio
from pyngrok import ngrok
## added 

# Download the video from URL and save it
def download_video(url: str, output_path: str):
    cmd = f"ffmpeg -i \"{url}\" -c copy {output_path}"
    subprocess.run(cmd, shell=True)

# Extract frames from video
def extract_frames(video_path: str, output_dir: str):
    frames_generator = sv.get_video_frames_generator(video_path)
    sink = sv.ImageSink(
        target_dir_path=output_dir,
        image_name_pattern="{:05d}.jpeg"
    )
    with sink:
        for frame in frames_generator:
            sink.save_image(frame)

# Plot label and marker on image
def plot_label_with_marker_on_image(image_path, position, label, font_size=20, color='red', marker_radius=5):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    x, y = position['x'], position['y']
    draw.ellipse((x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius), outline=color, width=2)
    draw.text((x + marker_radius + 5, y - font_size // 2), label, fill=color, font=font)
    return img

# Initialize the inference state for SAM2 model
def initialize_inference_state(frames_dir: str):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = f"checkpoints/sam2_hiera_large.pt"
    CONFIG = "sam2_hiera_l.yaml"

    sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)
    inference_state = sam2_model.init_state(frames_dir)
    sam2_model.reset_state(inference_state)
    return inference_state
