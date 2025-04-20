import numpy as np
from collections import deque
import argparse
import torch
import os
import cv2
import time
import multiprocessing
import pyrealsense2 as rs
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to

from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.pytorch3d_render import render_mesh_perspective, render_mesh_orthogonal
from pytorch3d.transforms import matrix_to_axis_angle
from cvzone.HandTrackingModule import HandDetector
from std_msgs.msg import Float64MultiArray
import rospy
from termcolor import cprint

LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)

# Parameters for exponential moving average
ALPHA = 0.35 # Smoothing factor for EMA

import matplotlib.pyplot as plt

# fix z axis

# ax = plt.axes(projection="3d")
HAND_VISULIZATION_LINKS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

def plot_hand_keypoints(keypoints: np.ndarray, ax):
    assert keypoints.shape[0] == 21
    lines = HAND_VISULIZATION_LINKS

    ax.cla()
    # limit = 20
    # ax.set_xlim3d(-limit, limit)
    # ax.set_ylim3d(-limit, limit)
    # ax.set_zlim3d(-limit, limit)
    ax.set_xlim3d(0, 0.2)
    ax.set_ylim3d(-0.1, 0.1)
    ax.set_zlim3d(9, 12)
    
    for line in lines:
        ax.plot3D(
            keypoints[line, 0],
            keypoints[line, 1],
            keypoints[line, 2],
            "gray",
        )
    ax.scatter3D(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], "black")
    
    plt.pause(.001)

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('207122078785')
    # config.enable_device('332122060334')
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 200.000)
    # sensor.set_option(rs.option.auto_exposure_priority, True)
    return pipeline

def start_detection(queue_1: multiprocessing.Queue, args, queue_2: multiprocessing.Queue):
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    os.makedirs(args.out_folder, exist_ok=True)

    with torch.no_grad():
        while True:
            img_cv2, wrist_trans, count, boxes, right = queue_1.get()
            if img_cv2 is None:
                time.sleep(.01)
                continue

            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                import time
                st = time.time()
                out = model(batch)
                print("Time taken for inference: ", time.time() - st)
                queue_2.put((out, batch, img_cv2, wrist_trans, model.mano.faces, count))

def apply_exponential_moving_average(new_data, prev_ema, alpha):
    if prev_ema is None:
        return new_data
    return alpha * new_data + (1 - alpha) * prev_ema

def mano_params_to_msg(mano_params):
    keypoints = mano_params.cpu().numpy().flatten().tolist()
    data = keypoints

    return Float64MultiArray(data=data)

def calculate_finger_distance(keypoints):
    """
    Calculate the distance between thumb tip and index finger tip
    
    Args:
        keypoints (torch.Tensor): Hand keypoints tensor of shape [21, 3]
        
    Returns:
        float: Distance between thumb tip and index finger tip
    """
    # Thumb tip is keypoint 4, index finger tip is keypoint 8
    thumb_tip = keypoints[4]
    index_tip = keypoints[8]
    
    # Calculate Euclidean distance
    distance = torch.sqrt(torch.sum((thumb_tip - index_tip) ** 2))
    return distance.item()

def vis_detection(queue: multiprocessing.Queue):
    prev_ema = None
    rospy.init_node('hand_keypoints', anonymous=True)
    # use ros publish to send the pred_mano_params to the client
    pub = rospy.Publisher('/hamer/hand_keypoints', Float64MultiArray, queue_size=1)
    pub_wrist_coordinates = rospy.Publisher('/hamer/wrist_coordinates', Float64MultiArray, queue_size=1)
    pub_finger_distance = rospy.Publisher('/hamer/finger_distance', Float64MultiArray, queue_size=1)

    prev_keypoints = None
    while True:
        out, batch, img_cv2, wrist_trans, mano_faces, count = queue.get() # wrist_trans,
        if out is None:
            time.sleep(.01)
            continue

        pred_vertices = out.get('pred_vertices')
        

        if pred_vertices is not None:
            pred_vertices = pred_vertices.reshape(-1, 3).cpu().numpy()
            pred_mano_params = out.get('pred_mano_params') 
            pred_keypoints = out.get('pred_keypoints_3d') # torch.Size([1, 21, 3])

            if prev_ema is None:
                prev_ema = pred_vertices
            else:
                prev_ema = apply_exponential_moving_average(pred_vertices, prev_ema, ALPHA)

            verts = torch.tensor(prev_ema).unsqueeze(0).to(batch['img'].device)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = (2 * batch['right'] - 1) * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            img_height = int(img_size[0, 1].item())
            img_width = int(img_size[0, 0].item())
            scaled_focal_length = 5000.0 / 256.0 * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
            pred_keypoints = pred_keypoints.squeeze(0)
            pred_keypoints += pred_cam_t_full
            # print(pred_keypoints[:, 2])

            weights = [0.8, 0.2]
            # 0.7 is the weight for the current frame, 0.3 is the weight for the previous frame
            if prev_keypoints is None:
                prev_keypoints = pred_keypoints
            else:
                pred_keypoints = weighted_moving_average(pred_keypoints, prev_keypoints, weights)
                prev_keypoints = pred_keypoints

            # plot_hand_keypoints(pred_keypoints.cpu().numpy(), ax)
            # cprint(pred_cam_t_full, 'green')

            if pred_mano_params is not None:
                mano_msg = mano_params_to_msg(pred_keypoints)
                pub.publish(mano_msg)
                
                # Calculate and publish the distance between thumb and index finger
                # finger_distance = calculate_finger_distance(pred_keypoints)
                finger_distance = 0
                pub_finger_distance.publish(Float64MultiArray(data=[finger_distance]))
                
                # wrist translation
                arm_msg = wrist_trans
                pub_wrist_coordinates.publish(Float64MultiArray(data=arm_msg))
                    

            verts += pred_cam_t_full.unsqueeze(0)
            face = torch.from_numpy(mano_faces[None, :, :].astype(np.int32)).cuda()
            render_cam_params = {'focal': torch.tensor([[scaled_focal_length, scaled_focal_length]]).cuda(), 'princpt': torch.tensor([[img_width / 2.0, img_height / 2.0]]).cuda()}

            rgb, depth = render_mesh_perspective(verts, face, render_cam_params, (img_height, img_width), 'right')
            fg, is_fg = rgb[0].cpu().numpy(), (depth[0].cpu().numpy() > 0)
            bg = img_cv2[:, :, :].copy()
            render_out = (fg * is_fg + bg * (1 - is_fg)).astype(np.uint8)
            
            # Render the finger distance on the image
            finger_distance = calculate_finger_distance(pred_keypoints)
            distance_text = f"Distance: {finger_distance:.3f}"
            cv2.putText(render_out, distance_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw a line between thumb tip and index finger tip
            thumb_tip = pred_keypoints[4].cpu().numpy()
            index_tip = pred_keypoints[8].cpu().numpy()
            
            # Project 3D points to 2D image space
            focal = scaled_focal_length
            princpt_x, princpt_y = img_width / 2.0, img_height / 2.0
            
            thumb_x = int((thumb_tip[0] * focal / thumb_tip[2]) + princpt_x)
            thumb_y = int((thumb_tip[1] * focal / thumb_tip[2]) + princpt_y)
            
            index_x = int((index_tip[0] * focal / index_tip[2]) + princpt_x)
            index_y = int((index_tip[1] * focal / index_tip[2]) + princpt_y)
            
            # Draw the line
            cv2.line(render_out, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

            cv2.imshow('render_out', render_out)
            cv2.waitKey(1)
            
def weighted_moving_average(current, previous, weights):
    """
    Apply weighted moving average filter.
    
    Args:
        current (torch.Tensor): Current keypoints.
        previous (torch.Tensor): Previous keypoints.
        weights (list): Weights for the current and previous keypoints.

    Returns:
        torch.Tensor: Filtered keypoints.
    """
    assert len(weights) == 2, "Weights should be a list of length 2"
    weight_current, weight_previous = weights
    
    # Compute the weighted average
    filtered = weight_current * current + weight_previous * previous
    return filtered

def produce_frame(queue: multiprocessing.Queue):
    detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.8)
    camera = init_realsense()
    count = 0

    

    while True:
        frames = camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:

            if not color_frame:
                print("No color frame")

            if not depth_frame:
                print("No depth frame")
            continue
        count += 1
        frame = np.asanyarray(color_frame.get_data())
        img_cv2 = cv2.cvtColor((frame), cv2.COLOR_RGB2BGR)
        hands, img = detector.findHands(img_cv2, draw=False, flipType=True)


        bboxes = []
        is_right = []
        
        # depth


        if hands:
            hand1 = hands[0]
            center1 = hand1['center']
            # Modify: create a new bounding box based on the center of the hand
            bboxes.append([center1[0] - 100, center1[1] - 100, center1[0] + 100, center1[1] + 100])
            handType1 = hand1["type"]
            is_right.append(1 if handType1 == 'Right' else 0)

        if len(bboxes) == 1:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
        
            # wrist translation
            wrist_keypoint_index = 0
            wrist_coords_2d = hands[0]['lmList'][wrist_keypoint_index][:2]

            # print(hands[0]['lmList'])

            x_pixel = int(wrist_coords_2d[0])
            y_pixel = int(wrist_coords_2d[1])

            half_window_size = 3
            y_min = max(0, y_pixel - half_window_size)
            y_max = min(480, y_pixel + half_window_size)  # 480 is hacked here for the height reslution
            x_min = max(0, x_pixel - half_window_size)
            x_max = min(640, x_pixel + half_window_size)  # 640 is hacked here for the width reslution

            # please note the depth_frame_crop is a crop of the depth frame accroding to the cropped rgb hand bbox
            # wrist_depth_array = depth_frame[y_min:y_max, x_min:x_max]
            # wrist_z = np.median(wrist_depth_array)

            # Retrieve depth values within the window
            wrist_depth_array = []
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value != 0:  # Filter out invalid depth values
                        wrist_depth_array.append(depth_value)

            # Calculate median depth value
            if wrist_depth_array:
                wrist_z = np.median(wrist_depth_array)
            else:
                wrist_z = 0  # Handle case where no valid depth values were found

            # camera intrinsics: adjust here according to the camera you are using
            fx = 390.329
            fy = 390.329
            cx = 320.386
            cy = 235.952
            camera_intrinsics = np.array([[fx, 0, cx],
                                        [0, fy, cy]])

            wrist_x = wrist_z * (x_pixel - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]
            wrist_y = wrist_z * (y_pixel - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]
            wrist_trans = [wrist_x, wrist_y, wrist_z]
            # print(wrist_trans)
            
            queue.put((img_cv2, wrist_trans, count, boxes, right))

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    queue_1 = multiprocessing.Queue(maxsize=1)
    queue_2 = multiprocessing.Queue(maxsize=1)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue_1,))
    consumer_process_1 = multiprocessing.Process(target=start_detection, args=(queue_1, args, queue_2))
    consumer_process_2 = multiprocessing.Process(target=vis_detection, args=(queue_2,))

    producer_process.start()
    consumer_process_1.start()
    consumer_process_2.start()

    producer_process.join()
    consumer_process_1.join()
    consumer_process_2.join()

    time.sleep(1)

    print("done")

if __name__ == '__main__':
    main()
