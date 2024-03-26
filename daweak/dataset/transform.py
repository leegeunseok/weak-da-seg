from numpy import random
from typing import Sequence
import cv2
import numpy as np


class Dilate(): 
    """Dilate Label
    """

    def __init__(self, target_class, kernel_size, iteration=1):
        """
        Args:
            target_class (int): Target class to dilate label 
            kernel_size (int or tuple): kernel_size for dilation 
            iteration (int): num of iteration to run dilation
        """

        assert isinstance(target_class, int)
        assert isinstance(kernel_size, (int, tuple))
        assert isinstance(iteration, int)

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2

        self.target_class = target_class
        self.kernel_size = kernel_size
        self.iteration = iteration
        self.target_keys = ['segmap']


    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        for k in sample.keys():

            if k in self.target_keys:
                sample[k] = self._dilate(sample[k])

    def _dilate(self, segmap):
        """
        Args:
            segmap (np.arr, (H x W, uint8)): Segmentation map

        Returns:
            segmap (np.arr, (H x W, uint8)): Segmentation map
        """

        segmap_dilate = (segmap == self.target_class).astype(np.uint8)
        
        if isinstance(self.kernel_size, tuple):
            kernel = np.ones(self.kernel_size, np.uint8)
        elif isinstance(self.kernel_size, int):
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        segmap_dilate = cv2.dilate(segmap_dilate, kernel, iterations=self.iteration)

        segmap[segmap_dilate] = self.target_class

        return segmap


class RandomFlipLR():
    """
    Horizontally flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1

        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        if np.random.rand() < self.prob:
        
            for k in sample.keys():

                if k in self.target_keys:
                    sample[k] = cv2.flip(sample[k], 1)

        return sample


class RandomFlipUD():
    """
    Vertically flip the image and segmap.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): The flipping probability. Between 0 and 1.
        """
        self.prob = prob
        assert prob >=0 and prob <= 1
        self.target_keys = ['image', 'segmap']

    def __call__(self, sample):
        """
        Args:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        
        Returns:
            sample (dict, {image: np.arr (H x W x C, uint8), segmap: np.arr (H x W, uint8)})
        """

        if np.random.rand() < self.prob:
        
            for k in sample.keys():

                if k in self.target_keys:
                    sample[k] = cv2.flip(sample[k], 0)

        return sample


class RandomRotFlip:
    """Rotate and flip the image & seg or just rotate the image & seg.

    Args:
        rotate_prob (float): The probability of rotating the image.
        flip_prob (float): The probability of flipping the image.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of a tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
    """

    def __init__(self, rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)):
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        assert 0 <= rotate_prob <= 1 and 0 <= flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def transform(self, results: dict) -> dict:
        """Call function to rotate or rotate & flip image, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated or rotated & flipped results.
        """
        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rot_flip(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results


class PhotoMetricDistortion:
    def __init__(
            self,
            brightness_delta: int = 32,
            contrast_range: Sequence[float] = (0.5, 1.5),
            saturation_range: Sequence[float] = (0.5, 1.5),
            hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.float32)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        if random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        img = self.brightness(img)
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        results['img'] = img
        return results