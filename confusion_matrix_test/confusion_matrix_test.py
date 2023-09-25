from general import ap_per_class, ConfusionMatrix
confusion_matrix = ConfusionMatrix(nc=1)

import numpy as np

def xywh2xyxy(x): #坐标转化
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
#[N, 6] x1, y1, x2, y2, confidence, class
preds=np.array([[3.20333e+02, 1.08333e+00, 4.08333e+02, 2.37667e+02, 8.51074e-01, 0.00000e+00],
[8.02667e+02, 4.18000e+02, 1.06133e+03, 7.14667e+02, 8.42773e-01, 0.00000e+00],
[1.18067e+03, 5.42500e+01, 1.27667e+03, 3.76667e+02, 7.18262e-01, 0.00000e+00],
[5.24667e+02, 5.51667e+02, 7.94000e+02, 7.10000e+02, 6.36230e-01, 0.00000e+00],
[9.87333e+02, 0.00000e+00, 1.05400e+03, 6.83333e+01, 6.34766e-01, 0.00000e+00],
[1.06800e+03, 0.00000e+00, 1.14133e+03, 6.29167e+01, 4.94629e-01, 0.00000e+00],
[6.27000e+02, 5.50667e+02, 8.03333e+02, 7.13333e+02, 3.66455e-01, 0.00000e+00]])

#[M, 5] class,x1, y1, x2, y2,
gt_boxes=np.array([[0.00000e+00, 3.21000e+02, 2.99988e+00, 4.02000e+02, 2.40000e+02],
[0.00000e+00, 7.69001e+02, 4.23000e+02, 1.05900e+03, 7.16000e+02],
[0.00000e+00, 4.29000e+02, 5.49000e+02, 7.85999e+02, 7.16000e+02],
[0.00000e+00, 1.17000e+03, 1.87000e+02, 1.28000e+03, 3.95000e+02],
[0.00000e+00, 1.08200e+03, 9.99969e-01, 1.14900e+03, 6.30000e+01],
[0.00000e+00, 1.19800e+03, 3.10000e+01, 1.28000e+03, 2.28000e+02],
[0.00000e+00, 9.85999e+02, 9.99969e-01, 1.05300e+03, 6.30000e+01]])


confusion_matrix.process_batch(preds, gt_boxes)

print("result:",confusion_matrix.matrix)