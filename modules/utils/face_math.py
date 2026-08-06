import cv2
import numpy as np

WARP_TEMPLATE_SET = {
    'arcface_112_v1': np.array([
        [ 0.35473214, 0.45658929 ],
        [ 0.64526786, 0.45658929 ],
        [ 0.50000000, 0.61154464 ],
        [ 0.37913393, 0.77687500 ],
        [ 0.62086607, 0.77687500 ]
    ]),
    'arcface_112_v2': np.array([
        [ 0.34191607, 0.46157411 ],
        [ 0.65653393, 0.45983393 ],
        [ 0.50022500, 0.64050536 ],
        [ 0.37097589, 0.82469196 ],
        [ 0.63151696, 0.82325089 ]
    ]),
    'arcface_128': np.array([
        [ 0.36167656, 0.40387734 ],
        [ 0.63696719, 0.40235469 ],
        [ 0.50019687, 0.56044219 ],
        [ 0.38710391, 0.72160547 ],
        [ 0.61507734, 0.72034453 ]
    ]),
    'ffhq_512': np.array([
        [ 0.37691676, 0.46864664 ],
        [ 0.62285697, 0.46912813 ],
        [ 0.50123859, 0.61331904 ],
        [ 0.39308822, 0.72541100 ],
        [ 0.61150205, 0.72490465 ]
    ])
}

def estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size):
    warp_template_norm = WARP_TEMPLATE_SET.get(warp_template) * crop_size
    affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, warp_template_norm, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    return affine_matrix

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, warp_template, crop_size):
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop_vision_frame, affine_matrix

def calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix):
    temp_height, temp_width = temp_vision_frame.shape[:2]
    crop_height, crop_width = crop_vision_frame.shape[:2]
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    crop_points = np.array([ [ 0, 0 ], [ crop_width, 0 ], [ crop_width, crop_height ], [ 0, crop_height ] ])
    paste_region_points = transform_points(crop_points, inverse_matrix)
    paste_region_point_min = np.floor(paste_region_points.min(axis=0)).astype(int)
    paste_region_point_max = np.ceil(paste_region_points.max(axis=0)).astype(int)
    x1, y1 = np.clip(paste_region_point_min, 0, [ temp_width, temp_height ])
    x2, y2 = np.clip(paste_region_point_max, 0, [ temp_width, temp_height ])
    paste_bounding_box = np.array([ x1, y1, x2, y2 ])
    paste_matrix = inverse_matrix.copy()
    paste_matrix[0, 2] -= x1
    paste_matrix[1, 2] -= y1
    return paste_bounding_box, paste_matrix

def paste_back(temp_vision_frame, crop_vision_frame, crop_vision_mask, affine_matrix):
    paste_bounding_box, paste_matrix = calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix)
    x1, y1, x2, y2 = paste_bounding_box
    paste_width = x2 - x1
    paste_height = y2 - y1
    inverse_vision_mask = cv2.warpAffine(crop_vision_mask, paste_matrix, (paste_width, paste_height)).clip(0, 1)
    inverse_vision_mask = np.expand_dims(inverse_vision_mask, axis=-1)
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, paste_matrix, (paste_width, paste_height), borderMode=cv2.BORDER_REPLICATE)
    
    # We must operate on a copy if we want functional purity, but since we modify in place we do:
    temp_vision_frame = temp_vision_frame.copy()
    paste_vision_frame = temp_vision_frame[y1:y2, x1:x2]
    paste_vision_frame = paste_vision_frame * (1 - inverse_vision_mask) + inverse_vision_frame * inverse_vision_mask
    temp_vision_frame[y1:y2, x1:x2] = paste_vision_frame.astype(temp_vision_frame.dtype)
    return temp_vision_frame

def warp_face_by_bounding_box(temp_vision_frame, bounding_box, crop_size):
    source_points = np.array([ [ bounding_box[0], bounding_box[1] ], [ bounding_box[2], bounding_box[1] ], [ bounding_box[0], bounding_box[3] ] ]).astype(np.float32)
    target_points = np.array([ [ 0, 0 ], [ crop_size[0], 0 ], [ 0, crop_size[1] ] ]).astype(np.float32)
    affine_matrix = cv2.getAffineTransform(source_points, target_points)
    if bounding_box[2] - bounding_box[0] > crop_size[0] or bounding_box[3] - bounding_box[1] > crop_size[1]:
        interpolation_method = cv2.INTER_AREA
    else:
        interpolation_method = cv2.INTER_LINEAR
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, flags=interpolation_method)
    return crop_vision_frame, affine_matrix

def warp_face_by_translation(temp_vision_frame, translation, scale, crop_size):
    affine_matrix = np.array([ [ scale, 0, translation[0] ], [ 0, scale, translation[1] ] ])
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
    return crop_vision_frame, affine_matrix

def create_static_anchors(feature_stride, anchor_total, stride_height, stride_width):
    x, y = np.mgrid[:stride_width, :stride_height]
    anchors = np.stack((y, x), axis=-1)
    anchors = (anchors * feature_stride).reshape((-1, 2))
    anchors = np.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
    return anchors

def create_rotation_matrix_and_size(angle, size):
    rotation_matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
    rotation_size = np.dot(np.abs(rotation_matrix[:, :2]), size)
    rotation_matrix[:, -1] += (rotation_size - size) * 0.5
    rotation_size = int(rotation_size[0]), int(rotation_size[1])
    return rotation_matrix, rotation_size

def normalize_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    x1, x2 = sorted([ x1, x2 ])
    y1, y2 = sorted([ y1, y2 ])
    return np.array([ x1, y1, x2, y2 ])

def transform_points(points, matrix):
    points = points.reshape(-1, 1, 2)
    points = cv2.transform(points, matrix)
    points = points.reshape(-1, 2)
    return points

def transform_bounding_box(bounding_box, matrix):
    points = np.array([
        [ bounding_box[0], bounding_box[1] ],
        [ bounding_box[2], bounding_box[1] ],
        [ bounding_box[2], bounding_box[3] ],
        [ bounding_box[0], bounding_box[3] ]
    ])
    points = transform_points(points, matrix)
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return normalize_bounding_box(np.array([ x1, y1, x2, y2 ]))

def distance_to_bounding_box(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.column_stack([ x1, y1, x2, y2 ])

def distance_to_face_landmark_5(points, distance):
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]
    return np.stack((x, y), axis=-1)

def convert_to_face_landmark_5(face_landmark_68):
    return np.array([
        np.mean(face_landmark_68[36:42], axis=0),
        np.mean(face_landmark_68[42:48], axis=0),
        face_landmark_68[30],
        face_landmark_68[48],
        face_landmark_68[54]
    ])

def estimate_face_angle(face_landmark_68):
    x1, y1 = face_landmark_68[0]
    x2, y2 = face_landmark_68[16]
    theta = np.arctan2(y2 - y1, x2 - x1)
    theta = np.degrees(theta) % 360
    angles = np.linspace(0, 360, 5)
    index = np.argmin(np.abs(angles - theta))
    return int(angles[index] % 360)

def apply_nms(bounding_boxes, scores, score_threshold, nms_threshold):
    bounding_boxes_norm = [ (x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bounding_boxes ]
    keep_indices = cv2.dnn.NMSBoxes(bounding_boxes_norm, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    return keep_indices
