import numpy as np
import math
import cv2
from matplotlib import pyplot as plt


class ImageAugmentation:
    def __init__(self, noise=False, scale=False, rotate=False, normalize=False, pca=False):
        self.noise = noise
        self.rotate = rotate
        self.scale = scale
        self.normalize = normalize
        self.pca = pca

    def augment_batch(self, batch):
        for im_num in range(batch.shape[0]):
            batch[im_num] = self.augment(batch[im_num])
        return batch

    def augment(self, image):
        # plt.imshow(image)
        # plt.show()
        if self.rotate:
            image = self.add_rotation(image)
        if self.scale:
            image = self.random_zoom(image)
        if self.noise:
            image = self.add_noise(image)
        if self.normalize:
            image = self.add_norm(image)
        if self.pca:
            image = self.add_pca(image)
        return image

    def add_noise(self, image):
        return cv2.GaussianBlur(image, (0, 0), cv2.BORDER_DEFAULT)

    def add_rotation(self, image):
        # The manual way
        # new_image = 10.0 * np.random.random_sample(image.shape)
        # new_image = np.zeros(image.shape, dtype=np.uint8)
        # theta = math.pi * 2 * np.random.random_sample(1)
        # trans_matrix = [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
        # Pixel by pixel
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         new_pos = np.matmul(trans_matrix, [[i - 16], [j - 16]])
        #         new_pos = new_pos.round()
        #         new_i = int(new_pos[0][0])
        #         new_j = int(new_pos[1][0])
        #         if (new_i < -16 or new_i > 15) or (new_j < -16 or new_j > 15):
        #             print(i - 16, j - 16, new_i, new_j)
        #             continue
        #         new_image[new_i+16][new_j+16] = image[i-16][j-16]
        #         print(new_i, new_j, new_image[new_i][new_j], image[i][j])
        theta = 360 * np.random.random_sample(1)
        rows, cols, rgb = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        new_image = cv2.warpAffine(image, M, (cols, rows))
        return new_image

    # Randomly rescale image, then crop randomly to a smaller section or place randomly and fill background with black
    def random_zoom(self, image):
        scale = 0.5 + 1.0 * np.random.random_sample(1)
        new_height = int(scale * image.shape[0])
        new_width = int(scale * image.shape[1])
        new_image = self.fit_image(image, (new_height, new_width))
        if scale > 1:
            # Randomly crop a zoomed in image
            new_image = self.random_crop(new_image, image.shape)
        elif scale < 1:
            # Randomly place a smaller image
            h_start = int((image.shape[0] - new_image.shape[0]) * np.random.random_sample(1))
            h_end = h_start + new_height
            w_start = int((image.shape[1] - new_image.shape[1]) * np.random.random_sample(1))
            w_end = w_start + new_width
            new_image = np.pad(new_image, ((h_start, 32-h_end), (w_start, 32-w_end), (0, 0)), mode='constant', constant_values=(0, 0))
        return new_image

    # size is height, width
    def fit_image(self, image, size):
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    # size is height, width
    # Randomly chooses where to start, selects portion of size size
    def random_crop(self, image, size):
        h_start = int((image.shape[0] - size[0]) * np.random.random_sample(1))
        w_start = int((image.shape[1] - size[1]) * np.random.random_sample(1))
        new_image = image[h_start: h_start + size[0], w_start: w_start + size[1]]
        return new_image

    def add_norm(self, image):
        if len(image.shape) > 2:
            new_image = np.reshape(image, (image.shape[0] * image.shape[1], 3))
        else:
            new_image = image
        new_image = new_image.astype('float32')
        new_image -= np.mean(new_image, axis=0)
        new_image /= np.std(new_image, axis=0)
        return np.reshape(new_image, image.shape)

    # PCA Color Augmentation
    def add_pca(self, image):
        new_image = self.add_norm(np.reshape(image, (image.shape[0] * image.shape[1], 3)))

        cov = np.cov(new_image, rowvar=False)
        val, vec = np.linalg.eig(cov)
        # Proportional constants w/ mean 0, std 0.1
        alphas = np.random.normal(0, 0.1, 3)

        # Unit Eigen vectors * randomly scaled Eigen value vector
        delta = np.dot(vec, alphas * val)
        mean = np.mean(new_image, axis=0)
        std = np.std(new_image, axis=0)

        new_image += delta
        new_image = new_image * std + mean

        new_image = np.maximum(np.minimum(new_image, 255), 0).astype('uint8')

        return np.reshape(new_image, image.shape)

    def fit_batch(self, batch, size):
        new_size = [batch.shape[0], size[0], size[1], size[2]]
        new_batch = np.zeros(new_size)
        for im_num in range(new_size[0]):
            new_batch[im_num] = self.fit_image(batch[im_num], size)
        return new_batch

    # Local Warping

    # Color Shifting
