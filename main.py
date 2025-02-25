"""
Objective 

DistortMOT17 -> extractor -> clustering -> visualization
       MOT17 -> extractor -> clustering -> visualization


"""
import torch
import numpy as np

from extractor.ReID import ResNeXt50
from extract_features import extract_features
from generate_crop_bbox import read_bbox, read_image
from utils import cosine_distance, draw_heatmap

if __name__ == "__main__":   
    
    # Load images and bboxes
    img_folder = "assets/MOT17-04-SDP/img1"   
    gt_file    = "assets/MOT17-04-SDP/gt/gt.txt"
    
    model = ResNeXt50('cuda')
    data = {'1':{"bboxes":[], "features":[]}, '2':{"bboxes":[], "features":[]}}
    for id in ['1', '2']:
       bboxes = read_bbox(gt_file, id, xxyy = True)
       data[id]['bboxes'] = bboxes

       for i, bbox in enumerate(bboxes):
           img_name = f"{img_folder}/{str(i+1).zfill(6)}.jpg"
           img = read_image(img_name)
           feature = extract_features(model, img, bbox)
           data[id]['features'].append(feature)
 
    # Calculate cosine distance between features
    sizes    = len(data['1']['features']) + len(data['2']['features'])
    heatmaps = np.zeros((sizes, sizes))
    row_labels = [f"1_{i}" for i in range(len(data['1']['features']))]
    row_labels += [f"2_{i}" for i in range(len(data['2']['features']))] 
    col_labels = row_labels
    for i, f1 in enumerate(data['1']['features']):
       for j, f2 in enumerate(data['2']['features']):
              heatmaps[i, j] = cosine_distance(f1, f2)
    draw_heatmap(heatmaps, row_labels, col_labels)
        
