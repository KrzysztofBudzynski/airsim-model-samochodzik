import airsim
import torch
import cv2 as cv
import numpy as np
from time import time
import lib
import sys, getopt

from utils.utils import letterbox, driving_area_mask, lane_line_mask,\
    split_for_trace_model, non_max_suppression, plot_one_box, scale_coords, clip_coords
 
def start_loop(client, camera_number = 1):
    print('using camera ', camera_number)
    loop_time = time()
    with torch.no_grad():
        while(True):
            scr = lib.photo_to_np_ndarray(client, camera_number)
            img0 = scr.copy()
            img = cv.resize(img0, (640,480), interpolation=cv.INTER_NEAREST)
            output = img.copy()
            
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).cuda()
            img = img.float().half()
            img /= 255.0
            img = img.unsqueeze(0)
            [pred,anchor_grid],seg,ll = model(img)

            masking = True
            obj_det = True
            
            if masking:
                da_seg_mask = seg
                _, da_seg_mask = torch.max(da_seg_mask, 1)
                da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                
                ll_seg_mask = ll
                ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
                ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                
                color_area = np.zeros((da_seg_mask.shape[0], da_seg_mask.shape[1], 3), dtype=np.uint8)
                
                color_area[da_seg_mask == 1] = [0, 255, 0]
                color_area[ll_seg_mask == 1] = [255, 0, 0]
                color_seg = color_area
                color_seg = color_seg[..., ::-1]
                color_mask = np.mean(color_seg, 2)
                output[color_mask != 0] = output[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                
            if obj_det:
                pred = split_for_trace_model(pred,anchor_grid)
                pred = non_max_suppression(pred)
                pred0 = pred[0]
                
                img0_shape = output.shape
                clip_coords(pred0, img0_shape)
                
                for det in pred0:
                    *xyxy, _, _ = det
                    plot_one_box(xyxy, output)
                    
            cv.imshow("YOLOPv2", output)
            
            print("FPS {}".format(1.0 / (time() - loop_time)))
            loop_time = time()
            cv.waitKey(5)

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,"hp:c:",["modelpath=","camera="])
    camera_number = 1
    model_path = 'data/weights/yolopv2.pt'
    
    for opt, arg in opts:
        if opt == '-h':
            print('-c = 1 - numer kamery, -m = data/weights/yolopv2.pt - ścieżka do modelu')
            sys.exit()
        elif opt in ['-m', 'modelpath']:
            model_path = arg
        elif opt in ['-c', 'camera']:
            camera_number = arg
            
    model = torch.jit.load(model_path)
    model.cuda()
    model.half()
    model.eval()

    imgsz = 640
    model(torch.zeros(1, 3, imgsz, imgsz).cuda().type_as(next(model.parameters())))
    client = airsim.CarClient()
    client.confirmConnection()
    start_loop(client, camera_number)