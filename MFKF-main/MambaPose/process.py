import cv2
import torch
from argparse import Namespace
from pathlib import Path
from utils.general import non_max_suppression, set_logging
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.torch_utils import select_device
from utils.general import scale_coords
from actions import actionPredictor
from detect import detect

def detect_and_analyze(opt, action_predictor):
    set_logging()
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    half = device.type != 'cpu' and not save_txt_tidl

    save_dir, dataset = detect(opt)  # 调用detect函数获取检测结果

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):
            if len(det):
                for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                    # Process each detection
                    if opt.save_txt:
                        # Write detection results to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:
                        # Draw bounding box and keypoints
                        c = int(cls)
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness,
                                     kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Analyze action status using action predictor
                boundingboxes = [xyxy]
                joints_humans = [det[:, 6:].cpu().numpy()]
                statuses = action_predictor.analyze_joints(im0s, joints_humans, boundingboxes)

                # Display status
                print(statuses)


# Example usage:
opt = Namespace(
    source='data/images',  # input source
    weights='yolov7-w6-pose.pt',  # model weights
    img_size=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=True,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    kpt_label=False,  # add human pose keypoints to image
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    save_img=False,  # save inference images
    save_crop=False,
    save-txt-tidl = True
    # save cropped prediction boxes
)

action_predictor = actionPredictor()
detect_and_analyze(opt, action_predictor)
