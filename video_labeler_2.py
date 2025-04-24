import os
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image
import time
import cv2
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

# mask = cv2.imread('mask.png')
# mask_real = (1 * (mask > 0)).astype(np.uint8)

yolov8_model_path = "yolov8x.pt"
input_video_path = "sample.mov"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.5,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

# xxx = os.listdir('second/images/train/') # FOLDER
# files = [i for i in xxx if '.jpg' in i]

print('load model finished')
vcap = cv2.VideoCapture("./1726614814.043377-1.MOV")
total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vcap.get(cv2.CAP_PROP_FPS))
waiting_jump = 5
print('video loaded')

# Calculate the number of frames to skip
frames_to_skip = fps - 1  # Since we want 1 frame per second

# Create output directory if it doesn't exist
os.makedirs('cycle_vid', exist_ok=True)

# Define video writer for saving annotated frames
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Dictionary to count occurrences of each class
class_counts = {}

# Initialize the progress bar
with tqdm(total=total_frames // fps, desc="Processing frames") as pbar:
    ii = 0
    frame_count = 0
    
    # Get first frame to initialize video writer
    ret, frame = vcap.read()
    if ret:
        video_writer = cv2.VideoWriter(
            output_video_path, 
            fourcc, 
            fps, 
            (frame.shape[1], frame.shape[0])
        )
    
    # Reset video capture to start
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = vcap.read()
        if not ret:
            break  # Exit the loop if no more frames
        
        frame_count += 1
        
        # Process current frame
        result = get_sliced_prediction(
            frame[:,:,::-1],
            detection_model,
            slice_height=1080,
            slice_width=1080,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2, 
            verbose=False
        )
        
        anns = result.to_coco_annotations()
        cycle_dict = {'motorcycle': 2, 'bicycle': 2}
        cycle = False
        
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Draw all detected objects on the annotated frame (for video)
        for ann in anns:
            # Count each class
            category_name = ann['category_name']
            if category_name not in class_counts:
                class_counts[category_name] = 0
            class_counts[category_name] += 1
            
            # Draw bounding box for all classes
            x, y, w, h = [int(coord) for coord in ann['bbox']]
            
            # Use different colors for different classes
            if category_name in cycle_dict:
                color = (0, 255, 0)  # Green for motorcycles/bicycles
                cycle = True
            else:
                color = (255, 0, 0)  # Red for other objects
                
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{category_name}: {ann['score']:.2f}"
            cv2.putText(annotated_frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save individual frame without annotations if it contains a motorcycle or bicycle
        if cycle:
            cv2.imwrite(f'cycle_vid/{ii}.jpg', frame)  # Save original frame without annotations
            ii += 1
        
        # Write annotated frame to video (all frames)
        video_writer.write(annotated_frame)
        
        # Skip frames to achieve 1 fps for the next iteration (only do object detection to first frame in one second video)
        for _ in range(frames_to_skip):
            if vcap.grab():  # Successfully grabbed a frame (skip 29 frames)
                frame_count += 1
            else:
                break  # No more frames to grab
            
        pbar.update(1)  # Update the progress bar

# Release resources
vcap.release()
video_writer.release()

print(f"Processing complete. Saved {ii} clean frames with motorcycle/bicycle detections.")
print(f"Complete annotated video saved to {output_video_path}")

# Print class counts
print("\nDetection counts by class:")
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{class_name}: {count}")

# Print total number of detections
total_detections = sum(class_counts.values())
print(f"\nTotal number of detections: {total_detections}")
print(f"Total number of frames processed: {frame_count}")
