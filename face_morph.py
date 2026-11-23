"""Three-way landmark-based face morphing between anchor emotion images."""
import cv2
import face_alignment
import numpy as np
import torch
from scipy.spatial import Delaunay

_fa = None


def detect_landmarks(img, pad_frac=0.3, threshold=0.8):
    """Detect 68 face landmarks + 8 image-boundary points on an RGB uint8 image.

    The image is reflect-padded first, since tightly cropped faces often have the
    chin/jawline touching the frame. Raises ValueError if no face is found.
    """
    global _fa
    if _fa is None:
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector='sfd',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            face_detector_kwargs={'filter_threshold': threshold},
        )

    h, w = img.shape[:2]
    pad_h, pad_w = int(h * pad_frac), int(w * pad_frac)
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    preds = _fa.get_landmarks(padded)
    if not preds:
        raise ValueError("No face detected")
    landmarks = preds[0] - np.array([pad_w, pad_h])

    boundary = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2],
    ])
    return np.concatenate((landmarks, boundary), axis=0)


def warp_triangle_mesh(img_src, pts_src, pts_dst):
    """Piecewise-affine warp of img_src so that pts_src move to pts_dst (Delaunay mesh)."""
    warped = np.zeros_like(img_src)
    img_h, img_w = img_src.shape[:2]
    for simplex in Delaunay(pts_dst).simplices:
        src_tri = pts_src[simplex].astype(np.float32)
        dst_tri = pts_dst[simplex].astype(np.float32)
        x, y, w, h = cv2.boundingRect(dst_tri)
        # Clip bounding rect to image bounds
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, img_w), min(y + h, img_h)
        if w == 0 or h == 0 or x1 >= x2 or y1 >= y2:
            continue
        dst_tri_local = dst_tri - np.array([x, y], dtype=np.float32)
        M = cv2.getAffineTransform(src_tri, dst_tri_local)
        patch = cv2.warpAffine(img_src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_local.astype(np.int32), 255)
        # Crop patch and mask to the clipped region
        patch = patch[y1 - y:y2 - y, x1 - x:x2 - x]
        mask = mask[y1 - y:y2 - y, x1 - x:x2 - x]
        warped[y1:y2, x1:x2][mask > 0] = patch[mask > 0]
    return warped


def morph_face_tri(imgs, landmarks_list, weights):
    """Morph three face images (RGB uint8 arrays) using barycentric weights."""
    pts_target = sum(w * l for w, l in zip(weights, landmarks_list)).astype(np.float32)
    morphed = sum(
        w * warp_triangle_mesh(img, lmks.astype(np.float32), pts_target).astype(np.float64)
        for w, img, lmks in zip(weights, imgs, landmarks_list)
    )
    return np.clip(morphed, 0, 255).astype(np.uint8)
