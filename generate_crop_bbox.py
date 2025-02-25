import os
import cv2 

def read_image(img_name):
    if os.path.exists(img_name):
        img = cv2.imread(img_name)
        return img
    print("No images")
    return None

def read_bbox(gt_file, id, xxyy=False):
    '''
        gt_file: path to ground truth file
        id: id of the object to extract
        objective: extract the bbox of the object with id
    '''
    bboxes = []
    with open(gt_file, 'r') as f:
        lines = f.read().splitlines()
        line  = [line for line in lines if line.split(',')[1] == id] # Extract the line of the object with id

        # Extract the bbox of the object
        for l in line:
            bbox = l.split(',')[2:6]
            bbox = [int(b) for b in bbox]
            if xxyy:
                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            bboxes.append(bbox)
        
    return bboxes

def crop_bbox(img, bbox):
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def trim_bbox(bbox, img_shape):
    x, y, w, h = bbox
    img_h, img_w = img_shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h

def test_crop_bbox():
    img_name = "assets/MOT17-04-SDP/img1/000001.jpg"
    gt_file  = "assets/MOT17-04-SDP/gt/gt.txt"
    img = read_image(img_name)
    bbox = read_bbox(gt_file, '1')[0]
    cropped_img = crop_bbox(img, bbox)
    cv2.imwrite("assets/cropped_img.jpg", cropped_img)

if __name__ == "__main__":
    # crop bboxes id=1 following whole frames and save at id_1_bbox folder
    img_folder = "assets/MOT17-04-SDP/img1"
    gt_file = "assets/MOT17-04-SDP/gt/gt.txt"
    id = '1'
    bboxes = read_bbox(gt_file, id)

    if not os.path.exists(f"{img_folder}/id_{id}_bbox"):
        os.mkdir(f"{img_folder}/id_{id}_bbox")  
        
    for i, bbox in enumerate(bboxes):
        img_name = f"{img_folder}/{str(i+1).zfill(6)}.jpg"
        img = read_image(img_name)
        cropped_img = crop_bbox(img, bbox)
        x, y, w, h = trim_bbox(bbox, img.shape[:2])
        cv2.imwrite(f"assets/id_{id}_bbox/{str(i+1).zfill(6)}.jpg", cropped_img)
        print(f"Saved assets/id_{id}_bbox/{str(i+1).zfill(6)}.jpg")

    # crop bboxes id=2 following whole frames and save at id_2_bbox folder
    id = '2'
    bboxes = read_bbox(gt_file, id)
    if not os.path.exists(f"{img_folder}/id_{id}_bbox"):
        os.mkdir(f"{img_folder}/id_{id}_bbox")  
        
    for i, bbox in enumerate(bboxes):
        img_name = f"{img_folder}/{str(i+1).zfill(6)}.jpg"
        img = read_image(img_name)
        cropped_img = crop_bbox(img, bbox)
        x, y, w, h = trim_bbox(bbox, img.shape[:2])
        cv2.imwrite(f"assets/id_{id}_bbox/{str(i+1).zfill(6)}.jpg", cropped_img)
        print(f"Saved assets/id_{id}_bbox/{str(i+1).zfill(6)}.jpg")


    