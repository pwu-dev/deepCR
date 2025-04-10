import numpy as np


def medmask(image, mask):
    """
    inpaint using 5*5 median filter sampling.
    :param image:
    :param mask:
    :return: inpainted image
    """
    clean = np.copy(image)
    xmax = image.shape[0]
    ymax = image.shape[1]
    medianImage = np.median(image)
    good = image * (1 - mask)
    pos = np.where(mask)
    for i in range(len(pos[0])):
        x = pos[0][i]
        y = pos[1][i]
        img = good[max(0, x-2):min(x+3, xmax+1),
                   max(0, y-2):min(y+3, ymax+1)]
        if img.sum() != 0:
            clean[x, y] = np.median(img[img != 0])
        else:
            clean[x, y] = medianImage
    return clean


def maskMetric(PD, GT):
    """
    Compute metrics on detected CRs mask with ground truth mask
    
    Args:
        PD : Predicted Detections
        GT : Ground Truth mask
        
    Returns:
        np.array: metrics array with TP, TN, FP and FN.
    """
    # If PD/GT is not a batch of images
    if len(PD.shape) == 2:
        # Add the batch dimension
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)

    # Initialize metrics variables
    TP, TN, FP, FN = 0, 0, 0, 0
    
    # Browse all images
    for i in range(GT.shape[0]):
        # P not used ? 
        # P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return np.array([TP, TN, FP, FN])

