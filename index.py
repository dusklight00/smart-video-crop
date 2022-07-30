import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm 
    return matrix

def create_heatmap(video_clip):
    heatmap = None

    for frame in video_clip.iter_frames():
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (3,3), 0) 
        edges = cv2.Canny(image=image_blur, threshold1=100, threshold2=200)

        if heatmap is None:
            heatmap = edges.copy()
            continue

        heatmap = np.add(heatmap, edges)
    
    return heatmap

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def score_window(window):
    white_pixels = 0
    for x in range(window.shape[0]):
        for y in range(window.shape[1]):
            if window[x,y] > 200:
                white_pixels += 1
    return white_pixels

def smart_crop_heatmap(heatmap, width, height, step_size = 100):    
    max_score = None
    max_window = None

    for window in sliding_window(heatmap, step_size, (width, height)):
        score = score_window(window[2])
        
        if max_score is None:
            max_score = score
            max_window = window
            continue

        if score > max_score:
            max_score = score
            max_window = window
        
    return max_window[0], max_window[1], width, height

def smart_crop_video(video_path, output_path, width, height):
    clip = VideoFileClip(video_path)
    heatmap = create_heatmap(clip)
    x, y, width, height = smart_crop_heatmap(heatmap, 100, 100)
    clip = clip.crop(x, y, x + width, y + height)
    clip.write_videofile(output_path)

smart_crop_video("sample.mp4", "output.mp4", 100, 100)