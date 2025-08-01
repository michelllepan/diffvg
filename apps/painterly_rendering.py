"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse

from datetime import datetime
import pickle
import os
import sys
import json
import numpy as np

pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S-") + args.target.split('/')[-1].split('.')[0]
    os.makedirs(f'results/{dir_name}', exist_ok=True)

    # Save command used to run the script
    with open(f'results/{dir_name}/command.txt', 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []
    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            shape_groups.append(path_group)
    else:
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = torch.tensor([random.random(),
                                                                          random.random(),
                                                                          random.random(),
                                                                          random.random() if not args.ignore_alpha else 1.0]))
            shape_groups.append(path_group)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/init.png', gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/iter_{t}.png', gamma=gamma)
        pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/latest.png', gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()

        # Zero out alpha gradients for color variables
        if args.ignore_alpha:
            for color_var in color_vars:
                if color_var.grad is not None:
                    color_var.grad[..., 3] = 0.0

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()
        color_optim.step()
        if len(stroke_width_vars) > 0:
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg(f'results/{dir_name}/iter_{t}.svg',
                              canvas_width, canvas_height, shapes, shape_groups)
            # Save shapes and shape groups using pickle
            with open(f'results/{dir_name}/shapes.pkl', 'wb') as f:
                pickle.dump(shapes, f)
            with open(f'results/{dir_name}/shape_groups.pkl', 'wb') as f:
                pickle.dump(shape_groups, f)
    
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/final.png', gamma=gamma)
    # Convert the intermediate renderings to a video.
    # from subprocess import call
    # call(["ffmpeg", "-framerate", "24", "-i",
    #     f"results/{dir_name}/iter_%d.png", "-vb", "20M",
    #     f"results/{dir_name}/out.mp4"])

    # Convert strokes to json
    def bezier_curve(P0, P1, P2, n_points=5):
        """Quadratic Bezier curve."""
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

    def cubic_bezier_curve(P0, P1, P2, P3, n_points=5):
        """Cubic Bezier curve."""
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    
    json_strokes = []
    for i in range(len(shapes)):
        s = shapes[i].points.detach().cpu().numpy()
        num_control_points = shapes[i].num_control_points.detach().cpu().numpy()
        pts, idx = [], 0
        for n in num_control_points: 
            if n == 0:
                p0, p1 = s[idx:idx+2]
                pts.append(np.stack([p0, p1]))
                idx += 1
            elif n == 1:
                p0, p1, p2 = s[idx:idx+3]
                pts.append(bezier_curve(p0, p1, p2))
                idx += 2
            elif n == 2:
                p0, p1, p2, p3 = s[idx:idx+4]
                pts.append(cubic_bezier_curve(p0, p1, p2, p3))
                idx += 3

        pts = np.concatenate(pts, axis=0)
        color = shape_groups[i].stroke_color.detach().cpu().numpy()
        stroke_width = shapes[i].stroke_width.detach().cpu().numpy()
        json_strokes.append({
            "points": pts.tolist(),
            "color": color.tolist(),
            "stroke_width": float(stroke_width),
        })

    with open(f'results/{dir_name}/{dir_name.split("-")[1]}_strokes_diffvg.json', 'w') as f:
        json.dump({"strokes": json_strokes}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    parser.add_argument("--ignore_alpha", dest='ignore_alpha', action='store_true')
    args = parser.parse_args()
    main(args)
