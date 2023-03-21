import airsim
import io
import numpy as np
from PIL import Image

def photo_to_np_ndarray(client, CameraNumber = 0, ImageType = airsim.ImageType.Scene):
    CameraName = str(CameraNumber)
    response = client.simGetImage(CameraName, ImageType)
    img_bytes = io.BytesIO(response)
    img_np = np.array(Image.open(img_bytes))
    img_np = img_np[:, :, :3]
    return img_np