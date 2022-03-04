import cv2

class Desc_Extractor:
    def __init__(self, method='orb', nfeatures=10000):
        if method=='sift':
            self.Desc = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=5, contrastThreshold=0)
        if method=='orb':
            self.Desc = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=0)
#                                    scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)
        
    def get_keypoints(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        keypoints = self.Desc.detect(gray, None)
        return keypoints

    def get_filtered_keypoints(self, keypoints, img, bias_factor, thres_size, position):
        filtered_keypoints = []
        for keypoint in keypoints:
            if keypoint.size < thres_size:
                continue
            if position == 'left_image':
                if keypoint.pt[0] > img.shape[1] - 1920*(bias_factor):
                    filtered_keypoints.append(keypoint)
            elif position == 'right_image':
                if keypoint.pt[0] < 1920*(bias_factor):
                    filtered_keypoints.append(keypoint)
        return filtered_keypoints

    def compute(self, img, keypoints):
        return self.Desc.compute(img, keypoints)