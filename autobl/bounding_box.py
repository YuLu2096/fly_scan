import numpy as np


class BoundingBox(list):

    def __init__(self, bbox_list):
        """
        Bounding box.

        :param bbox_list: list. [sy, sx, ey, ex].
        """
        super().__init__(bbox_list)
        self.sy, self.sx, self.ey, self.ex = bbox_list

    def contains(self, bbox):
        if self.sy <= bbox[0] and self.sx <= bbox[1] and self.ey >= bbox[2] and self.ex >= bbox[3]:
            return True
        else:
            return False

    def is_isolated_from(self, bbox):
        if self.sx > bbox.ex or self.ex < bbox.sx or self.sy > bbox.ey or self.ey < bbox.sy:
            return True
        else:
            return False

    @property
    def height(self):
        return self.ey - self.sy

    @property
    def width(self):
        return self.ex - self.sx

    def set_sy(self, v):
        self[0] = v
        self.sy = v

    def set_sx(self, v):
        self[1] = v
        self.sx = v

    def set_ey(self, v):
        self[2] = v
        self.ey = v

    def set_ex(self, v):
        self[3] = v
        self.ex = v

    def generate_mask(self, image_size):
        mask = np.zeros(image_size, dtype='bool')
        mask[self.sy:self.ey, self.sx:self.ex] = True
        return mask