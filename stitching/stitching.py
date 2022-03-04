from .utils import test_list, hist_match
from .desc_extractor import Desc_Extractor
from .segfuel import SegFuel
from .feat_extractor import Feat_extractor

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

class Stitching:
    def __init__(self, segmodel_path, featmodel_path, device, nfeatures=10000):
        self.segfuel = SegFuel(segmodel_path, device)
        self.extractor = Desc_Extractor(nfeatures=nfeatures)
        self.feat = Feat_extractor(featmodel_path, device)
        self.device = device
        
    def crop_fuel(self, img, mask):
        contour, _ = cv2.findContours(255-mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area < 500000:
                continue
            x_min, x_max, y_min, y_max = np.min(cnt[:,:,0]), np.max(cnt[:,:,0]), np.min(cnt[:,:,1]), np.max(cnt[:,:,1])
            break
        x_min = max(0, x_min-15)
        y_min = max(0, y_min-15)
        x_max = min(img.shape[1], x_max+15)
        y_max = min(img.shape[0], y_max+15)
        crop = img[y_min:y_max, x_min:x_max]
        
        crop = Image.fromarray(crop)
        
        return crop, [x_min, y_min, x_max, y_max]
        
    '''
    bias_factor: 이미지에서 키포인트를 찾는 부분의 비율. 0~1
    thres_size: 키포인트 벡터의 임계값. 조정할 필요 X
    '''
    def concat_row(self, image_paths, bias_factor=0.4, thres_size=5):
        images = []

        ## load images in same row
        for image_path in image_paths:
            images.append(cv2.imread(image_path))

        features = []
        inter_features = []
        bboxes = []
        stitch_results = []
        mask_results = []
        stitch_mask_results = []
        ## for each cell
        for idx in range(len(images)-1):

            if idx == 0: # leftest image
                # set left img, right img
                left_img = images[idx]
                right_img = images[idx+1]
            
                # extract feature(left)
                start = time.time()
                left_infer = self.segfuel.inference_net(left_img, infer_size=(640,640))
                print("segment infer time per once(x16):",time.time()-start)
                
                ret, left_mask = cv2.threshold(left_infer, 127, 255, cv2.THRESH_BINARY)
                mask_results.append(left_mask)
                crop, bbox = self.crop_fuel(left_img, left_mask)
                bboxes.append(bbox)
                start = time.time()
                feat = self.feat.extract(crop)
                print("feature infer time: per once(x32)",time.time()-start)
                inter_feat = self.feat.inter_extract(crop)
                #TODO: force cpu to feature extract model
#                 if 'cuda' in self.device:
#                     features.append(feat.cpu().numpy().tolist())
#                     inter_features.append(inter_feat.cpu().numpy().tolist())
#                 else:
                features.append(feat.tolist())
                inter_features.append(inter_feat.tolist())
                
                background_color = (int(np.mean(left_img[:,250,0])),int(np.mean(left_img[:,250,1])),int(np.mean(left_img[:,250,2])))
            else:
                # set left img, right img
                left_img = result.copy()
                right_img = images[idx+1]
                
                # extract feature(left)
                left_mask = result_mask.copy()
                
#                 # 이미지 밝기 조정
#                 right_img = hist_match(right_img, left_img)
                # 붙이는 과정에서 손실될 수 있으므로 이미지 zero padding
                left_img = cv2.copyMakeBorder(left_img, 50, 50, 0, 0, cv2.BORDER_CONSTANT)
                left_mask = cv2.copyMakeBorder(left_mask, 50, 50, 0, 0, cv2.BORDER_CONSTANT)
                
            # extract feature(right)
            right_infer = self.segfuel.inference_net(right_img, infer_size=(640,640))
            ret, right_mask = cv2.threshold(right_infer, 127, 255, cv2.THRESH_BINARY)
            mask_results.append(right_mask)
            crop, bbox = self.crop_fuel(right_img, right_mask)
            feat = self.feat.extract(crop)
            inter_feat = self.feat.inter_extract(crop)
            #TODO: force cpu to feature extract model
#             if 'cuda' in self.device:
#                 features.append(feat.cpu().numpy().tolist())
#                 inter_features.append(inter_feat.cpu().numpy().tolist())
#             else:
            features.append(feat.tolist())
            inter_features.append(inter_feat.tolist())
            
            # 붙이는 과정에서 손실될 수 있으므로 이미지 zero padding
            right_img = cv2.copyMakeBorder(right_img, 200, 200, 0, 0, cv2.BORDER_CONSTANT)

            start = time.time()
            ## get keypoints
            left_keypoints = self.extractor.get_keypoints(left_img)
            right_keypoints = self.extractor.get_keypoints(right_img)
            print("get keypoints time per once(x24):",time.time()-start)
            
            ## filter keypoints
            left_filtered_keypoints = self.extractor.get_filtered_keypoints(left_keypoints, left_img, bias_factor, thres_size, 'left_image')
            right_filtered_keypoints1 = self.extractor.get_filtered_keypoints(right_keypoints, right_img, bias_factor, thres_size, 'right_image')

            
            ## compute sift vector
            start = time.time()
            left_kp = tuple(left_filtered_keypoints)
            right_kp = tuple(right_filtered_keypoints1)
            left_keys, left_desc = self.extractor.compute(left_img, left_kp)
            right_keys, right_desc = self.extractor.compute(right_img, right_kp)
            print("compute sift vector time(x24):",time.time()-start)

            start = time.time()
            ## match keypoints
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(left_desc, right_desc)

            ## filter matches
            good_matches = matches
#             min_match = min(good_matches, key=lambda x: x.distance)
#             good_matches = [gm for gm in good_matches if gm.distance <= min_match.distance*10]

            left_points = []
            right_points = []
            gap = []
            for match_idx in range(len(good_matches)):

                left_point = np.array(left_kp[good_matches[match_idx].queryIdx].pt)
                right_point = np.array(right_kp[good_matches[match_idx].trainIdx].pt)
                left_points.append(left_point)
                right_points.append(right_point)
                gap.append(left_point-right_point)

            print('matches:%d/%d' %(len(good_matches),len(matches)))

            ## sort matches
            good_matches = [gm for idx, gm in enumerate(good_matches) if left_points[idx][1] - right_points[idx][1] >= -500 and left_points[idx][1] - right_points[idx][1] <= -150]

            print('sorted matches:%d/%d' %(len(good_matches),len(matches)))


            print("match time(x12):",time.time()-start)
            
            ## image stitch
            src_pts = np.float32([ left_kp[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ right_kp[m.trainIdx].pt for m in good_matches ])
            
            # 오른쪽 이미지는 left_img.shape[1]-x_median, y_median 만큼 평행이동되어있음
            x_median = left_img.shape[1] - round(np.median(src_pts[:,0] - dst_pts[:,0]))
            y_median = round(np.median(src_pts[:,1] - dst_pts[:,1]))
            # bbox 평행이동
            bbox[0] += left_img.shape[1]-x_median
            bbox[1] += y_median
            bbox[2] += left_img.shape[1]-x_median
            bbox[3] += y_median

            result = np.zeros((right_img.shape[0], left_img.shape[1] + right_img.shape[1], 3), dtype=np.uint8)
            result[:left_img.shape[0], :left_img.shape[1]] = left_img

            result_mask = np.zeros((right_img.shape[0], left_img.shape[1] + right_img.shape[1]), dtype=np.uint8)
            result_mask[:left_img.shape[0], :left_img.shape[1]] = left_mask
            
            
            # 이미지 합칠 때 전지가 잘리지 않도록 경계구간 설정
            mask = right_mask # 오른쪽 이미지에서 경계구간을 찾음
            mask = cv2.copyMakeBorder(mask, 200, 200, 0, 0, cv2.BORDER_CONSTANT)
    
            first_w = 0
            last_w = 0
            # 이미지의 마지막 전지가 있는 지점부터 시작, 왼쪽으로 순회. 중앙은 이미 전지의 안 이므로
            for w in range(mask.shape[1] - 920, 0, -10):

                # 전지에서 벗어나는 첫 지점을 찾음
                if first_w == 0:
                    if sum(mask[:,w] == 0)/len(mask[:,w]) < 0.5:
                        first_w = w
                        last_w = first_w - 200
                # 두번째 전지로 들어가는 첫 지점을 찾음
                else:
                    if sum(mask[:,w] == 0)/len(mask[:,w]) > 0.5:
                        last_w = w
                        break
            trim_factor = int((first_w+last_w) / 2)
            
            # 이미지 밝기 조정
            right_pivot = (trim_factor, int((right_img.shape[0]-y_median)/2))
            left_pivot = (left_img.shape[1]-x_median+trim_factor, int((left_img.shape[0]+y_median)/2))
            right_img = hist_match(right_img, left_img, right_pivot, left_pivot)
            
            # 붙이기
            result[:y_median, left_img.shape[1]-x_median+trim_factor:left_img.shape[1]-x_median+right_img.shape[1]] = right_img[-y_median:,trim_factor:]
            result_mask[:y_median, left_img.shape[1]-x_median+trim_factor:left_img.shape[1]-x_median+right_img.shape[1]] = mask[-y_median:,trim_factor:]
            
            ## 필요없는 부분의 x,y,w,h 계산
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(result_gray)
            x, y, w, h = cv2.boundingRect(coords)
            # 이미지에서 필요없는 부분 제거
            result = result[y:y+h, x:x+w]
            result_mask = result_mask[y:y+h, x:x+w]
            # bbox 평행이동
            bbox[0] -= x
            bbox[1] -= y
            bbox[2] -= x
            bbox[3] -= y
            
            result_mask[(result==[0,0,0]).all(axis=2)] = 255
#             result[(result==[0,0,0]).all(axis=2)] = background_color

            ## 이미지 위아래 간격 조절
            result = result[35:-15]
            result_mask = result_mask[35:-15]
            # bbox 평행이동
            bbox[1] -= 35
            bbox[3] -= 35

#             # 이미지 zero padding 했던 것 계산
            bbox[1] += 200
            bbox[3] += 200
            
            stitch_results.append(result)
            stitch_mask_results.append(result_mask)
            bboxes.append(bbox)
            
        return result, result_mask, features, inter_features, bboxes, background_color, mask_results, stitch_results, stitch_mask_results

    def concat_col(self, images):
        first_flag = True
        for result in images:
            if first_flag:
                final_result = result
                first_flag = False
            else:
                if final_result.shape[1] > result.shape[1]:
                    final_result = final_result[:,:result.shape[1]]
                else:
                    result = result[:,:final_result.shape[1]]

                final_result = cv2.vconcat([final_result,result])

        return final_result