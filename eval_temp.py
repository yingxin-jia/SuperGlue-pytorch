
import pdb

import torch.nn.modules
from superglue.dataloader import HomographyDataLoader, collater
from torch import optim
torch.autograd.set_detect_anomaly(True)
import cv2
from scipy.spatial.distance import cdist


nfeatures = opt.nfeatures
sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)

image = cv2.imread("image goes here")

height, width = image.shape[:2]
max_size = max(height, width)

corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

M = cv2.getPerspectiveTransform(corners, corners + warp)

warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))

kp1, descs1 = sift.detectAndCompute(image, None)
kp2, descs2 = sift.detectAndCompute(warped, None)

kp1 = kp1[:nfeatures]
kp2 = kp2[:nfeatures]

kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

kp1_np = kp1_np[:nfeatures, :]
kp2_np = kp2_np[:nfeatures, :]
descs1 = descs1[:nfeatures, :]
descs2 = descs2[:nfeatures, :]

kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
dists = cdist(kp1_projected, kp2_np)

kp1_np = kp1_np / max_size
kp2_np = kp2_np / max_size
descs1 = descs1 / 256.
descs2 = descs2 / 256.
sol = superglue.forward(torch.from_numpy(kp1_np).float().unsqueeze(0).cuda(),
                    torch.from_numpy(descs1).float().unsqueeze(0).cuda(),
                    torch.from_numpy(kp2_np).float().unsqueeze(0).cuda(),
                    torch.from_numpy(descs2).float().unsqueeze(0).cuda(), [])
matches = torch.where(sol[0, :50, :50] > 0.5)
matches_cv2 = []
pts_1 = []
pts_2 = []
print(matches)
for idx in range(matches[0].shape[0]):
    matches_cv2.append(cv2.DMatch(matches[0][idx], matches[1][idx], 0.0))
    pts_1.append(kp1[matches[0][idx].item()].pt)
    pts_2.append(kp2[matches[1][idx].item()].pt)

M2 = cv2.findHomography(np.array(pts_1), np.array(pts_2))

unwarped = cv2.warpPerspective(src=warped, M=M2[0], dsize=(image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP)
cv2.imshow('uw', unwarped)
cv2.imshow('w', warped)
cv2.waitKey(0)
pdb.set_trace()
img_out = cv2.drawMatches(image, kp1, warped, kp2, matches_cv2, None)
cv2.imshow('a', img_out)
cv2.waitKey(0)