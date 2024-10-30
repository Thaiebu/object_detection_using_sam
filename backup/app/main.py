
# pip install fastapi uvicorn nest-asyncio pyngrok python-multipart

from fastapi import FastAPI, File, UploadFile
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
import json
from fastapi.middleware.cors import CORSMiddleware

## added 
from .helper import  download_video,extract_frames,plot_label_with_marker_on_image,initialize_inference_state
from .ml_model import sam2_model

##
import os
HOME = os.getcwd()
print("HOME:", HOME)

nest_asyncio.apply()
app = FastAPI()


nest_asyncio.apply()
config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
current_dir = os.path.dirname(__file__)
json_file_path = os.path.join(current_dir, "highlightreels.json")

with open(json_file_path, "r") as buffer:  # Open the file in read mode
        data = json.load(buffer)
print(data.keys())
url = str(data["hReelArr"][0]["highlightS3Url"])
print(url)
# FastAPI route to download and process the video
@app.post("/process_video/")
async def process_video():
    # Download video
    output_video_path = f"{HOME}/output_video.mp4"
    download_video(url, output_video_path)

    # Extract frames
    output_frames_dir = f"{HOME}/output_1"
    os.makedirs(output_frames_dir, exist_ok=True)
    extract_frames(output_video_path, output_frames_dir)

    # Initialize SAM2 model for video inference
    inference_state = initialize_inference_state(output_frames_dir)
    return {"message": "Video processed and frames extracted"}

# Route for adding markers and labels to an image
@app.post("/annotate-image/")
async def annotate_image():
    image_path = f"{HOME}/output_1/00000.jpeg"
    position = {'x': 370, 'y': 370}
    label = 'ball'
    img_with_marker = plot_label_with_marker_on_image(image_path, position, label, font_size=30, color='red', marker_radius=2)

    # Save annotated image
    annotated_image_path = f"{HOME}/annotated_image.jpeg"
    img_with_marker.save(annotated_image_path)

    return {"message": f"Image annotated", "annotated_image_path": annotated_image_path}

# Route to handle object tracking
@app.post("/track-object/")
async def track_object():
    global inference_state
    points = np.array([[data["position"]["x"], data["position"]["y"]]], dtype=np.float32)
    labels = np.array([1])
    frame_idx = 0
    tracker_id = 1
    inference_state = initialize_inference_state(f"{HOME}/output_1")

    _, object_ids, mask_logits = sam2_model.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=tracker_id,
        points=points,
        labels=labels,
    )

    return {"message": "Object tracking initialized"}

# Route to process video and generate a final video with annotations
@app.post("/generate-annotated-video/")
async def generate_annotated_video():
    global inference_state
    video_info = sv.VideoInfo.from_video_path(f"{HOME}/output_video.mp4")
    frames_paths = sorted(sv.list_files_with_extensions(directory=f"{HOME}/output_1", extensions=["jpeg"]))
    colors = ['#FF1493']
    mask_annotator = sv.MaskAnnotator(
        color=sv.ColorPalette.from_hex(colors),
        color_lookup=sv.ColorLookup.TRACK)
    with sv.VideoSink(f"{HOME}/final_output.mp4", video_info=video_info) as sink:
        for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
            frame = cv2.imread(frames_paths[frame_idx])
            masks = (mask_logits > 0.0).cpu().numpy()
            N, X, H, W = masks.shape
            masks = masks.reshape(N * X, H, W)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=np.array(object_ids)
            )
            frame = mask_annotator.annotate(frame, detections)
            sink.write_frame(frame)

    return {"message": "Annotated video generated", "output_video": f"{HOME}/final_output.mp4"}

# public_url = ngrok.connect(8000)
# print(f"Public URL: {public_url}")
# origins = [
#     "http://localhost:3000"
# ]

# # Run the server
# uvicorn.run(app, host="127.0.0.1", port=8000)

