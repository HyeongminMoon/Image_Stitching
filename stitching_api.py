#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stitching import Stitching, test_list, hist_match

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import requests
import base64
import json
import torch

import argparse
parser = argparse.ArgumentParser(description="FUEL_feature_extractor_api")
parser.add_argument('--device', type=str, default='None')
parser.add_argument('--segmodel', type=str, default='stitching/saved_models/u2net.pth')
parser.add_argument('--featmodel', type=str, default = 'stitching/saved_models/wide_resnet101_2.onnx')
parser.add_argument('--port', type=int, default = 8933)

args = parser.parse_args()

if args.device == 'None':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
else:
    device = args.device
# load stitch model
stitch_model_path = args.segmodel
feature_extract_model_path = args.featmodel
stitch = Stitching(stitch_model_path, feature_extract_model_path, device, nfeatures=10000)

app = Flask(__name__)

print('-----------------------------')

prog = 0
stop_flag = False

@app.route('/stop', methods=['POST'])
def stop():
    global stop_flag
    if request.method == 'POST':
        stop_flag = True
        
        # lock until stopped
        while(1):
            if stop_flag == False:
                break
        return jsonify("stopped")

@app.route('/progress', methods=['POST'])
def progress():
    global prog
    if request.method == 'POST':
        return jsonify(prog)

@app.route('/stitching', methods=['POST'])
def stitching():
    global prog
    global stop_flag
    if request.method == 'POST':
        if prog != 0:
            return jsonify("false")
        
        
        #load request
        r = request
        data_json = r.data
        data_dict = json.loads(data_json)

        root_path = data_dict['root_path']
        
        debug = False
        if 'debug' in data_dict:
            debug = data_dict['debug']
        
        img_list = test_list(root_path)
        
        prog = 1
        result_dict = {}
        for key, value in img_list.items():
            if '+' in key:
                pole = 'plus'
            elif '-' in key:
                pole = 'minus'
            else:
                raise Exception('양극 또는 음극 이미지가 없습니다.')
                
            result_dict[pole] = {}
        
            results = []
            result_masks = []
            result_features = []
            result_inter_features = []
            result_bboxes = []
            if debug:
                mr_paths = []
                sr_paths = []
                smr_paths = []
            ## for each column
            for image_index, image_paths in enumerate(value):
                if stop_flag:
                    stop_flag = False
                    prog = 0
                    return jsonify("stopped")
                
                # 가로로 합체
                # image stitching & feature extract
                result, result_mask, feats, inter_feats, bboxes, bg_color, mask_results, stitch_results, stitch_mask_results = stitch.concat_row(image_paths, bias_factor=0.4, thres_size=5)
                prog += 1

                results.append(result)
                result_masks.append(result_mask)
                result_features.append(feats)
                result_inter_features.append(inter_feats)
                result_bboxes.append(bboxes)
                
                if debug:
                    os.makedirs(os.path.join(os.path.dirname(image_paths[0]), 'debug'), exist_ok=True)
                    for mr_idx, mr in enumerate(mask_results):
                        mr_path = os.path.join(os.path.dirname(image_paths[0]), 'debug', f"mask_{os.path.basename(image_paths[mr_idx])}")
                        cv2.imwrite(mr_path, mr)
                        mr_paths.append(mr_path)
                    for sr_idx, sr in enumerate(stitch_results):
                        sr_path = os.path.join(os.path.dirname(image_paths[0]), 'debug', f"stitch_{pole}_{image_index}_{sr_idx}.png")
                        cv2.imwrite(sr_path, sr)
                        sr_paths.append(sr_path)
                    for smr_idx, smr in enumerate(stitch_mask_results):
                        smr_path = os.path.join(os.path.dirname(image_paths[0]), 'debug', f"stitchmask_{pole}_{image_index}_{smr_idx}.png")
                        cv2.imwrite(smr_path, smr)
                        smr_paths.append(smr_path)

            # 세로로 합체
            final_result = stitch.concat_col(results)
            final_mask = stitch.concat_col(result_masks)
            for idx, result in enumerate(results):
                h = result.shape[0]
                result_bboxes = np.array(result_bboxes, dtype=np.int32)
                result_bboxes[idx+1:, :, 1::2] += h
                        
            full_img_path = os.path.join(os.path.dirname(image_paths[0]), f"full_{pole}.png")
            cv2.imwrite(full_img_path, final_result)
            full_mask_path = os.path.join(os.path.dirname(image_paths[0]), f"fullmask_{pole}.png")
            cv2.imwrite(full_mask_path, final_mask)
            
            result_dict[pole]['full_img_path'] = full_img_path
            result_dict[pole]['full_mask_path'] = full_mask_path
            result_dict[pole]['features'] = result_features
            result_dict[pole]['inter_features'] = result_inter_features
            result_dict[pole]['bboxes'] = result_bboxes.tolist()
            result_dict[pole]['bg_color'] = bg_color
            if debug:
                result_dict[pole]['mask_results'] = mr_paths
                result_dict[pole]['stitch_results'] = sr_paths
                result_dict[pole]['stitch_mask_results'] = smr_paths
        prog = 0
        # result to json
        return jsonify(result_dict)

if __name__ == '__main__':
                        
    app.run(host='0.0.0.0', port=args.port, debug=False)