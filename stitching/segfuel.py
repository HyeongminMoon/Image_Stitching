import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class SegFuel:
    def __init__(self, model_path, device):
    
        # load and initialize u2net segmentation model
        self.net = U2NET(3,1)
        torch_dict_seg = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(torch_dict_seg)
        
        if device == 'cpu':
            dtype = torch.qint8
            quantized_model = torch.quantization.quantize_dynamic(self.net, {torch.nn.LSTM, torch.nn.Linear, torch.nn.Conv2d}, dtype=dtype)
            self.net = quantized_model
        
        self.net.to(device)
        self.net.eval()
        
        self.model_path = model_path
        self.device = device
        
    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    def inference(self, input):

        # normalize the input
        tmpImg = np.zeros((input.shape[0],input.shape[1],3))
        input = input/np.max(input)

        tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
        tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

        # convert BGR to RGB
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = tmpImg[np.newaxis,:,:,:]
        tmpImg = torch.from_numpy(tmpImg)

        # convert numpy array to torch tensor
        tmpImg = tmpImg.type(torch.FloatTensor)
        tmpImg = tmpImg.to(self.device)
    #     if torch.cuda.is_available():
    #         tmpImg = Variable(tmpImg.cuda())
    #     else:
        tmpImg = Variable(tmpImg)

        # inference
        d1,d2,d3,d4,d5,d6,d7= self.net(tmpImg)

        # normalization
        pred = 1.0 - d1[:,0,:,:]
        pred = self.normPRED(pred)

        # convert torch tensor to numpy array
        pred = pred.squeeze()
        pred = pred.cpu().data.numpy()

        del d1,d2,d3,d4,d5,d6,d7

        return pred

    def inference_net(self, img, infer_size=None):
        with torch.no_grad():
            h,w = img.shape[:2]

            if infer_size:
                img = cv2.resize(img, dsize=infer_size)

            im_portrait = self.inference(img)
            dst = cv2.resize((im_portrait*255).astype(np.uint8), dsize = (w,h))

        return dst
    
import time
    
if __name__ == '__main__':
    from u2net import U2NET
    
    model_path = 'saved_models/u2net.pth'
    device = 'cpu'
    
    segfuel = SegFuel(model_path, device)
    
    img = cv2.imread('/mnt/vitasoft/NIPA_서버_백업/49.50.174.192/ETRI_FUEL/Sample2/Sample2[-] 4-1.png')
    
    
    start = time.time()
    for a in range(10):
        result = segfuel.inference_net(img, infer_size=(640,640))
    print("time per 10:", time.time()-start)
    
#     plt.imshow(result, cmap='gray')
#     plt.show()
    cv2.imwrite('segfueltest.png', result)
else:
    from stitching.u2net import U2NET