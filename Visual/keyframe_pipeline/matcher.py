import sys
from pathlib import Path

import cv2
import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
LIGHTGLUE_REPO = CURRENT_DIR.parent / 'LightGlue'

if str(LIGHTGLUE_REPO) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_REPO))

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import numpy_image_to_torch, rbd  # noqa: E402


class LightGlueMatcher:
    def __init__(self, max_num_keypoints=2048, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.extractor = SuperPoint(
            max_num_keypoints=max_num_keypoints
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features='superpoint'
        ).eval().to(self.device)

    def _to_rgb(self, img):
        if img is None:
            raise ValueError('img 是 None')

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    @torch.inference_mode()
    def extract(self, img_bgr):
        """
        只對單張影像抽特徵，可拿來做快取
        回傳內容:
        {
            'raw_features': 給 LightGlue matcher 用的原始特徵(dict, 含 batch 維度),
            'reduced_features': 經 rbd 後的特徵(dict),
            'keypoints': numpy 格式的 keypoints
        }
        """
        img_rgb = self._to_rgb(img_bgr)
        t = numpy_image_to_torch(img_rgb).to(self.device)

        raw_features = self.extractor.extract(t)
        reduced_features = rbd(raw_features)

        kpts = reduced_features['keypoints']
        keypoints = kpts.detach().cpu().numpy().astype(np.float32)

        return {
            'raw_features': raw_features,
            'reduced_features': reduced_features,
            'keypoints': keypoints,
        }

    @torch.inference_mode()
    def match_features(self, feats0, feats1):
        """
        直接拿兩份已經抽好的特徵做 matching
        這樣上一張 keyframe 的特徵就可以重複利用
        """
        if feats0 is None or feats1 is None:
            raise ValueError('feats0 或 feats1 是 None')

        matches01 = self.matcher({
            'image0': feats0['raw_features'],
            'image1': feats1['raw_features']
        })

        matches01 = rbd(matches01)

        kpts0 = feats0['reduced_features']['keypoints']
        kpts1 = feats1['reduced_features']['keypoints']
        matches = matches01.get('matches', None)
        scores = matches01.get('scores', None)

        keypoints0 = feats0['keypoints']
        keypoints1 = feats1['keypoints']

        if matches is None or len(matches) == 0:
            empty_pts = np.empty((0, 2), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_mask = np.empty((0,), dtype=bool)
            empty_indices = np.empty((0, 2), dtype=np.int32)
            return {
                'num_keypoints0': int(len(keypoints0)),
                'num_keypoints1': int(len(keypoints1)),
                'keypoints0': keypoints0,
                'keypoints1': keypoints1,
                'num_matches': 0,
                'match_indices': empty_indices,
                'mkpts0': empty_pts,
                'mkpts1': empty_pts,
                'pts0': empty_pts,
                'pts1': empty_pts,
                'match_scores': empty_scores,
                'mean_match_score': -1.0,
                'inlier_mask': empty_mask,
                'inlier_ratio': 0.0,
            }

        match_indices = matches.detach().cpu().numpy().astype(np.int32).reshape(-1, 2)

        mkpts0 = kpts0[matches[:, 0]].detach().cpu().numpy().astype(np.float32)
        mkpts1 = kpts1[matches[:, 1]].detach().cpu().numpy().astype(np.float32)

        if scores is None:
            match_scores = np.full((len(matches),), -1.0, dtype=np.float32)
        else:
            match_scores = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)

        inlier_mask, inlier_ratio = self._compute_fundamental_inlier_mask_and_ratio(mkpts0, mkpts1)

        valid_scores = match_scores[match_scores >= 0.0]
        mean_match_score = float(valid_scores.mean()) if len(valid_scores) > 0 else -1.0

        return {
            'num_keypoints0': int(len(keypoints0)),
            'num_keypoints1': int(len(keypoints1)),
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'num_matches': int(len(matches)),
            'match_indices': match_indices,
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'pts0': mkpts0,
            'pts1': mkpts1,
            'match_scores': match_scores,
            'mean_match_score': mean_match_score,
            'inlier_mask': inlier_mask,
            'inlier_ratio': float(inlier_ratio),
        }

    def _compute_fundamental_inlier_mask_and_ratio(self, pts0, pts1):
        if len(pts0) < 8:
            mask = np.zeros((len(pts0),), dtype=bool)
            return mask, 0.0

        _, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99
        )

        if mask is None:
            mask = np.zeros((len(pts0),), dtype=bool)
            return mask, 0.0

        mask = mask.reshape(-1).astype(bool)
        total = int(len(mask))
        if total == 0:
            return mask, 0.0

        inliers = int(mask.sum())
        inlier_ratio = float(inliers) / float(total)
        return mask, inlier_ratio