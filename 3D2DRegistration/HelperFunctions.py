import numpy as np

def normalizeImage(iImage, iType = "whitening"):
    """
    Normalize an image based on the type
    :param iImage: Input image
    :param iType: Type of normalization
    :return: Normalized image
    """
    if iType == "whitening":
        oImage = (iImage - np.mean(iImage)) / np.std(iImage)
    elif iType == "range":
        oImage = (iImage - np.min(iImage)) / (np.max(iImage) - np.min(iImage))

    return oImage

def scaleImage(iImage, iSlope, iOffset):
    """
    Scale input image based on y = iSlope * x + iOffset
    :param iImage: Input image
    :param iSlope: Slope of linear function
    :param iOffset: Offset of linear function
    :return: Scaled image
    """
    iImageType = iImage.dtype
    iImage = np.array(iImage, dtype="float")
    oImage = iImage - iSlope * iOffset
    if iImageType.kind in ("u", "i"):
        oImage[oImage < np.iinfo(iImageType).min] = np.iinfo(iImageType).min
        oImage[oImage > np.iinfo(iImageType).max] = np.iinfo(iImageType).max
    return np.array(oImage, dtype=iImageType)

def windowImage(iImage, iCenter, iWidth):
    """
    Linear windowing
    :param iImage: Input image
    :param iCenter: Center of window
    :param iWidth: Width of window
    :return: New image
    """
    iImageType = iImage.dtype
    if iImageType.kind in ("u", "i"):
        maxValue = np.iinfo(iImageType).max
        minValue = np.iinfo(iImageType).min
    else:
        minValue = np.min(iImage)
        maxValue = np.max(iImage)

    slope = (maxValue - minValue) / iWidth
    offset = -slope * (float(iCenter) - iWidth / 2.0)

    return scaleImage(iImage, slope, offset)

def addHomogCoord(iPts):
    """
    Adds homogenous coordinates to grid
    :param iPts: Input points
    :return: Points with homogenous coordinates
    """
    iPts = np.asarray(iPts)
    iPts = np.hstack(
        (iPts, np.ones(
        (iPts.shape[0], 1)
        ))
    )

    return iPts

def prepareImages(iCtImage, iXrayImage, iCtPosition, iXrayPosition, iSourcePosition):
    iCtImage = normalizeImage(iCtImage)
    ctData = {
        "img": iCtImage,
        "position": iCtPosition
    }

    iXrayImage = windowImage(iXrayImage, 60.0, 120.0)
    xRayData = {
        "img": iXrayImage,
        "position": iXrayPosition,
        "source": iSourcePosition
    }

    return ctData, xRayData

# arr = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ]
#
# arr2 = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ]
#
# arr = np.asarray(arr)
# print(np.hstack((arr, np.ones((arr.shape[0], 1))))) # Adds column
# print(np.vstack((arr, np.ones((1, arr.shape[1]))))) # Adds row
# print(np.stack((arr, arr2)))