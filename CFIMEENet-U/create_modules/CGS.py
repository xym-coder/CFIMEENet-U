import numpy as np
import cv2
from shapely.geometry import Polygon
from config.config import config as cfg
from scipy import ndimage as ndimg
from skimage.morphology import skeletonize
from shapely.geometry import LineString
from PIL import Image 
from shapely.ops import unary_union

def get_midline_from_proposal_skeleton(proposal_points, inst_mask, target_pts):
    h, w = inst_mask.shape
    inst_mask_binary = (inst_mask > 0).astype(np.uint8)

    proposal_binary = np.zeros((h, w), dtype=np.uint8)

    valid_mask = (proposal_points[:, 0] > 0) & (proposal_points[:, 1] > 0) & \
                 (proposal_points[:, 0] < w) & (proposal_points[:, 1] < h)
    valid_points = proposal_points[valid_mask].astype(np.int32)

    if len(valid_points) > 2:

        cv2.fillPoly(proposal_binary, [valid_points], color=(1,))

        proposal_binary = proposal_binary & inst_mask_binary

        from skimage.morphology import skeletonize
        skeleton = skeletonize(proposal_binary > 0).astype(np.uint8)
    else:

        from skimage.morphology import skeletonize
        skeleton = skeletonize(inst_mask_binary > 0).astype(np.uint8)

    skeleton_coords = get_ordered_skeleton_points(skeleton)
    midline_points = sample_points_equally(skeleton_coords, target_pts)
    return midline_points.astype(np.int32)

def get_ordered_skeleton_points(skeleton):

    y_coords, x_coords = np.where(skeleton > 0)
    skeleton_points = np.column_stack([x_coords, y_coords])

    if len(skeleton_points) < 2:
        return skeleton_points

    def count_neighbors(point):
        y, x = point[1], point[0]
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbors += 1
        return neighbors

    endpoints = []
    for point in skeleton_points:
        if count_neighbors(point) == 1:
            endpoints.append(point)

    if len(endpoints) >= 2:
        ordered_points = trace_skeleton_path(skeleton, endpoints[0], skeleton_points)
    else:

        ordered_points = order_points_by_pca(skeleton_points)

    return ordered_points

def trace_skeleton_path(skeleton, start_point, all_points):
    visited = set()
    path = []
    current = start_point

    while current is not None:
        path.append(current)
        visited.add(tuple(current))

        next_point = None
        y, x = current[1], current[0]

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0 and (nx, ny) not in visited:
                        next_point = np.array([nx, ny])
                        break
            if next_point is not None:
                break

        current = next_point

    return np.array(path) if path else all_points

def order_points_by_pca(skeleton_points):
    center = np.mean(skeleton_points, axis=0)
    centered_points = skeleton_points - center
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    main_direction = eigenvectors[:, -1]

    projections = np.dot(centered_points, main_direction)
    sorted_indices = np.argsort(projections)

    return skeleton_points[sorted_indices]

def sample_points_equally(points, target_pts):
    if len(points) < 2:
        return np.tile(points[0] if len(points) > 0 else [0, 0], (target_pts, 1))

    try:

        from shapely.geometry import LineString
        line = LineString(points)
        distances = np.linspace(0, line.length, target_pts)
        sampled_points = []

        for dist in distances:
            point = line.interpolate(dist)
            sampled_points.append([point.x, point.y])

        return np.array(sampled_points)
    except Exception as e:

        print(f"Shapely interpolation failed: {e}, using simple interpolation")
        return simple_interpolate(points, target_pts)

def simple_interpolate(points, target_pts):
    if len(points) < 2:
        return np.tile(points[0] if len(points) > 0 else [0, 0], (target_pts, 1))

    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i - 1])
        distances.append(distances[-1] + dist)

    total_length = distances[-1]
    if total_length == 0:
        return np.tile(points[0], (target_pts, 1))

    target_distances = np.linspace(0, total_length, target_pts)
    sampled_points = []

    for target_dist in target_distances:

        for i in range(len(distances) - 1):
            if distances[i] <= target_dist <= distances[i + 1]:
                if distances[i + 1] == distances[i]:
                    sampled_points.append(points[i])
                else:

                    ratio = (target_dist - distances[i]) / (distances[i + 1] - distances[i])
                    point = points[i] + ratio * (points[i + 1] - points[i])
                    sampled_points.append(point)
                break

    return np.array(sampled_points)

