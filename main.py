"""
Objective 

DistortMOT17 -> extractor -> clustering -> visualization
       MOT17 -> extractor -> clustering -> visualization


"""
import torch
import numpy as np
from tqdm import tqdm

from extractor.ReID import ResNeXt50
from extract_features import extract_features
from generate_crop_bbox import read_bbox, read_image
from utils import cosine_distance, draw_heatmap

if __name__ == "__main__":   
    
    # Load images and bboxes
    img_folder = "assets/MOT17-04-SDP/img1"   
    gt_file    = "assets/MOT17-04-SDP/gt/gt.txt"
    
    model = ResNeXt50('cuda')
    data  = {'1':{"bboxes":[], "features":[]}, '2':{"bboxes":[], "features":[]}}
    for id in ['1', '2']:
       print(f'==== ID number : {id} ===')
       bboxes = read_bbox(gt_file, id, xxyy = True)
       data[id]['bboxes'] = bboxes[:100]

       for i, bbox in enumerate(data[id]['bboxes']):
              img_name = f"{img_folder}/{str(i+1).zfill(6)}.jpg"
              img      = read_image(img_name)
              print(f'Processing ... {img_name}')
              feature  = extract_features(model, img, bbox)
              data[id]['features'].append(feature)
 
    # Calculate cosine distance between features
    sizes       = 200
    heatmaps    = np.zeros((sizes, sizes))
    
    for i, fi in tqdm(enumerate(data['1']['features'] + data['2']['features'])):
       for j, fj in tqdm(enumerate(data['1']['features'] + data['2']['features'])):
              heatmaps[i, j] = cosine_distance(fi, fj)
    
    fig = draw_heatmap(heatmaps)
    fig.savefig('assets/heatmap.pdf', bbox_inches='tight')
        
