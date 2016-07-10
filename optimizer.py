import sys
sys.path.append('/usr/local/Cellar/opencv3/3.1.0_2/lib/python3.4/site-packages/')
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import random
import math

def dist(p1, p2):
    return math.sqrt((p1[0][0] - p2[0][0]) ** 2 + (p1[0][1] - p2[0][1]) ** 2)

def capture_shapes(img_path, shape_images_output_dir):
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    if width > 1000:
        # i.e. it comes from my phone.
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    height, width, channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edged = cv2.Canny(gray, 40, 250)
    closed = edged

    # I couldn't find a magic closing kernel size.
    # So we bump it up until the interior angles aren't too acute (<11deg)
    # Performance is overrated, anyway.
    close_kern_val = 0
    while True:
        valid_shapes = []
        if close_kern_val:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kern_val, close_kern_val))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        too_acute = False
        for c in cnts:
            peri = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if area > 200:
                valid_shapes.append(c)
                for i in range(len(approx) - 2):
                    if not dist(approx[i + 1], approx[i]) * dist(approx[i + 1], approx[i + 2]):
                        continue
                    diff = math.acos((dist(approx[i + 1], approx[i]) ** 2 + dist(approx[i + 1], approx[i + 2]) ** 2 - dist(approx[i], approx[i + 2]) ** 2) / (2 * dist(approx[i + 1], approx[i]) * dist(approx[i + 1], approx[i + 2])))
                    if diff < 0.2:
                        too_acute = True
                        break
            if too_acute:
                break
        if too_acute:
            close_kern_val += 1
            continue
        else:
            break

    # Flatten to ((x,y), (x2,y2), ...) and place at origin
    shapes_at_origin = []
    for shape in valid_shapes:
        min_x = None
        min_y = None
        for pt in shape:
            min_x = min(min_x, pt[0][0]) if min_x is not None else pt[0][0]
            min_y = min(min_y, pt[0][1]) if min_y is not None else pt[0][1]
        shapes_at_origin.append([(int(pt[0][0] - min_x), int(pt[0][1] - min_y)) for pt in shape])

    # Spit out individual images for web UI.
    shape_images = []
    for i, shape in enumerate(shapes_at_origin):
        shape_poly = Polygon(shape) # Lazy, to use .bounds
        width = shape_poly.bounds[2] - shape_poly.bounds[0]
        height = shape_poly.bounds[3] - shape_poly.bounds[1]
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image[:,:] = (255,255,255)
        out_path = shape_images_output_dir + "/shape_%d.jpg" % i

        raw_pts = [(int(x[0]), int(x[1])) for x in shape_poly.exterior.coords]
        np_pts = np.array(raw_pts, np.int32)
        cv2.fillPoly(blank_image, [np_pts], (0, 0, 0))
        cv2.imwrite(out_path, blank_image)
        shape_images.append(out_path)

    return shapes_at_origin, shape_images

def capture_container(img_path):
    image = cv2.imread(img_path)
    # Magic scaling factor for the sake of the demo.
    image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation = cv2.INTER_LINEAR)
    height, width, channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edged = cv2.Canny(gray, 40, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = max(cnts, key=lambda c: cv2.contourArea(c))

    return image, ((pt[0][0], pt[0][1]) for pt in best_contour)

def pack_shapes(result_image, container, shapes, result_img_path, progress_cb):
    container = Polygon(container)
    container_min_x, container_min_y, container_max_x, container_max_y = container.bounds
    shapes = [Polygon(shape) for shape in shapes]
    simplified_shapes = []
    for shape in shapes:
        simplified_shapes.append(shape.simplify(5))

    fail_inner = 0
    fail_outer = 0

    # This problem appears to be NP-complete, so we just brute-force it for a while.
    # Cookie annealing is coming in version 2.0.
    total_attempts = 0
    placed_simplified_shapes = []
    placed_shapes = []
    attempts_per_iter = 500
    for si, (shape_pair) in enumerate(sorted(zip(simplified_shapes, shapes), key=lambda x: -x[0].area)):
        simple_shape, shape = shape_pair
        while True:
            did_place_shape = False
            for i in range(attempts_per_iter):
                total_attempts += 1
                rot = random.uniform(0, 360)
                rot_ctr = simple_shape.centroid
                new_simple_shape = affinity.rotate(simple_shape, rot, rot_ctr)
                trans_x = -new_simple_shape.bounds[0]
                trans_y = -new_simple_shape.bounds[1]
                new_simple_shape = affinity.translate(new_simple_shape, trans_x, trans_y)
                new_shape_bounds = new_simple_shape.bounds
                pos = (random.randint(container_min_x, container_max_x - int(new_shape_bounds[2] - new_shape_bounds[0]) - 1), random.randint(container_min_y, container_max_y - int(new_shape_bounds[3] - new_shape_bounds[1]) - 1))
                trans_x += pos[0]
                trans_y += pos[1]
                new_simple_shape = affinity.translate(new_simple_shape, pos[0], pos[1])
                if not container.contains(new_simple_shape):
                    fail_outer += 1
                    continue
                ok = True
                for placed_shape in placed_simplified_shapes:
                    if placed_shape.intersects(new_simple_shape):
                        ok = False
                        fail_inner += 1
                        break
                if ok:
                    placed_simplified_shapes.append(new_simple_shape)
                    new_shape = affinity.rotate(shape, rot, rot_ctr)
                    new_shape = affinity.translate(new_shape, trans_x, trans_y)
                    placed_shapes.append(new_shape)
                    progress_cb(len(placed_shapes), total_attempts)
                    did_place_shape = True
                    break
                elif total_attempts % 100 == 0:
                    progress_cb(len(placed_shapes), total_attempts)
            if not did_place_shape:
                break

    # Draw guides on image.
    for shape in placed_shapes:
        raw_pts = [(int(x[0]), int(x[1])) for x in shape.exterior.coords]
        np_pts = np.array(raw_pts, np.int32)
        cv2.drawContours(result_image, [np_pts], -1, (255, 255, 255), 3);

    cv2.imwrite(result_img_path, result_image)
