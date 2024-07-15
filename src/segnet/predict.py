import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from dataset.utils import cvtColor, preprocess_input
from nets.segent import SegNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "./logs_DMCD/ep050-loss0.020-val_loss0.015.pth"
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = SegNet(num_classes=2)
    state_dict = torch.load(weights_path, map_location='cpu')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)

    mask_save = './data/DMCCD/Predictimg_mask/'

    imgs = os.listdir('./data/DMCCD/Predictimg')
    for img_sin in imgs:
        img_path = './data/DMCCD/Predictimg/' + img_sin
        # load image
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((512, 512))
        original_img = preprocess_input(np.array(original_img, np.float64))

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             ])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.float().to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output.argmax(1).squeeze(0)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(0).unsqueeze(0), size=(84, 84), mode='nearest').squeeze()
            print(prediction.shape)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            np.savetxt(mask_save + img_sin[:-4] + '.txt', prediction, fmt='%d')
            prediction[prediction == 1] = 255
            # prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)

            mask.save("./data/DMCCD/Predictimg_out/" + img_sin)


if __name__ == '__main__':
    main()
