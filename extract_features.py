import cv2
import torch

from extractor.ReID import ResNeXt50
from generate_crop_bbox import read_bbox
from utils import cosine_distance
def extract_features(model, img, bbox):
    f = model.get_features(img, bbox)
    return f
    
def test_ResNeXt50():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = ResNeXt50(device)
    
    # id 1 frame 1
    img       = cv2.imread("assets/MOT17-04-SDP/img1/000001.jpg")
    bbox      = read_bbox("assets/MOT17-04-SDP/gt/gt.txt", '1', xxyy = True)[0]
    featue_11 = model.get_features(img, bbox)
    
    print(featue_11)
if __name__ == "__main__":
    # test_ResNeXt50()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = ResNeXt50(device)
    
    # id 1 frame 1
    img       = cv2.imread("assets/MOT17-04-SDP/img1/000001.jpg")
    bbox      = read_bbox("assets/MOT17-04-SDP/gt/gt.txt", '1', xxyy = True)[0]
    featue_11 = model.get_features(img, bbox)
    
    # id 1 frame 2
    img       = cv2.imread("assets/MOT17-04-SDP/img1/000002.jpg")
    bbox      = read_bbox("assets/MOT17-04-SDP/gt/gt.txt", '1', xxyy = True)[1]
    featue_12 = model.get_features(img, bbox)
    
    # id 2 frame 1
    img       = cv2.imread("assets/MOT17-04-SDP/img1/000001.jpg")
    bbox      = read_bbox("assets/MOT17-04-SDP/gt/gt.txt", '2', xxyy = True)[0]
    featue_21 = model.get_features(img, bbox)

    # id 2 frame 2
    img       = cv2.imread("assets/MOT17-04-SDP/img1/000002.jpg")
    bbox      = read_bbox("assets/MOT17-04-SDP/gt/gt.txt", '2', xxyy = True)[1]
    featue_22 = model.get_features(img, bbox)

    d_1 = cosine_distance(featue_11, featue_12)
    d_2 = cosine_distance(featue_21, featue_22)

    d_3 = cosine_distance(featue_11, featue_21)
    d_4 = cosine_distance(featue_12, featue_22)

    print(d_1, d_2, d_3, d_4)