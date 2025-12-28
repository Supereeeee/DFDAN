import cv2
import numpy as np
import torch
import tempfile
import shutil
import logging
import traceback
from basicsr.archs.DFDAN_arch import DFDAN

logging.basicConfig(level=logging.INFO)
logging.info(f"using cuda: {torch.cuda.is_available()}")


def sacle(src_path, dst_path, scale=2, model_name=None):
    # 注释
    """
    Scale an image using DFDAN model.

    Parameters:
        src_path (str): path to the source image.
        dst_path (str): path to save the scaled image.
        scale (int): scale factor, default is 2. must be in [2, 3, 4].
        model_name (str): model name, default is None.

    Returns:
        None
    """
    if scale not in [2, 3, 4]:
        return
    model_name = f'DFDAN_x{scale}.pth' if model_name is None else model_name
    model_path = f'../experiments/pretrained_models/{model_name}'
    logging.info(f'Using model: {model_path}')
    logging.info(f'Scaling {src_path} to {dst_path} with scale factor {scale}x')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DFDAN(in_channels=3, channels=56, num_block=8, out_channels=3, upscale=scale)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # test_results = OrderedDict()
    # test_results['runtime'] = []

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # 生成图片到临时文件，然后进行推理
    with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as fp:
        imgname = fp.name
        logging.info(f'Testing {imgname}')
        # read image
        img = cv2.imread(src_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                # start.record()
                output = model(img)
                # end.record()
                # torch.cuda.synchronize()
                # test_results['runtime'].append(start.elapsed_time(end))  # milliseconds
        except Exception as error:
            logging.error(f'Error: {error} {imgname}')
            logging.error(traceback.format_exc())
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(imgname, output)
            shutil.copyfile(imgname, dst_path)

    # tot_runtime = sum(test_results['runtime']) / 1000.0
    # ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    # 打印运行时间
    # print('------> Total runtime of ({}) is : {:.6f} seconds = {:.2f} ms'.format(src_path, tot_runtime, tot_runtime * 1000))
    # print('------> Average runtime of ({}) is : {:.6f} seconds = {:.2f} ms'.format(src_path, ave_runtime, ave_runtime * 1000))

if __name__ == '__main__':
    sacle('./src/0001.jpg', './dst/0001.jpg')

