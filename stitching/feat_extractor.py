import timm
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import onnxruntime
import cv2
import os
print(onnxruntime.get_available_providers())
class Feat_extractor:
    def __init__(self, featmodel_path, device):
        
        device = 'cpu'# force cpu
        
        print("using", os.path.basename(featmodel_path).split(".onnx")[0])
        if 'cuda' in device:
            self.feat_model = timm.create_model(os.path.basename(featmodel_path).split(".onnx")[0], pretrained=True)
            self.feat_model.eval()
            self.feat_model.to(device)
        elif device == 'cpu':
            self.session = onnxruntime.InferenceSession(featmodel_path, providers=['CUDAExecutionProvider'])
            
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = device
        
    def extract(self, img, infer_size=224):
        img = img.resize((infer_size,infer_size)).convert("L").convert("RGB")
        img = self.transform(img).unsqueeze(0)

        if 'cuda' in self.device:
            with torch.no_grad():
                feat = self.feat_model(img.to(self.device)).detach().cpu()
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        elif self.device == 'cpu':
            feat = self.session.run(['modelOutput'], {'modelInput': img.numpy()})[0]
        return feat
        
    def inter_extract(self, img, infer_size=224):
        cropped_img = np.array(img)
        cropped_img = cropped_img[int(0.1*cropped_img.shape[0]):-int(0.1*cropped_img.shape[0]), int(0.1*cropped_img.shape[1]):-int(0.1*cropped_img.shape[1])]
        cropped_img = Image.fromarray(cropped_img)
        
        cropped_img = cropped_img.resize((infer_size,infer_size)).convert("L").convert("RGB")
        cropped_img = self.transform(cropped_img).unsqueeze(0)
        
        if 'cuda' in self.device:
            with torch.no_grad():
                feat = self.feat_model(cropped_img.to(self.device)).detach().cpu()
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        elif self.device == 'cpu':
            feat = self.session.run(['modelOutput'], {'modelInput': cropped_img.numpy()})[0]

        return feat
    
    def dist(self, img1, img2, mode='outer', infer_size=224):
        if mode == 'outer':
            feat1 = self.extract(img1, infer_size)
            feat2 = self.extract(img2, infer_size)
        elif mode == 'inter':
            feat1 = self.inter_extract(img1, infer_size)
            feat2 = self.inter_extract(img2, infer_size)
        else:
            raise Exception("mode should be 'outer' or 'inter'")
        feat1 = torch.Tensor(feat1)
        feat2 = torch.Tensor(feat2)
        
        
        dist = torch.sum(torch.sqrt((feat1 - feat2)**2).squeeze())
        
        return dist.item()
        
        
if __name__ == '__main__':
    
    device = "cuda:0"
    feat = Feat_extractor('saved_models/wide_resnet101_2.onnx', device)
    
    path1 = '/mnt/vitasoft/NIPA_서버_백업/49.50.174.192/ETRI_FUEL/dataset/1/1-1.png'
    path2 = '/mnt/vitasoft/NIPA_서버_백업/49.50.174.192/ETRI_FUEL/dataset/1/1-2.png'
    
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    
    dist = feat.dist(img1, img2, mode='outer')
    inter_dist = feat.dist(img1, img2, mode='inter')
    
    print(dist, inter_dist)