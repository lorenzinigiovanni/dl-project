import matplotlib.pyplot as plt
import random
import math


class RandomErasing(object):
    def __init__(self, prob=0.5, r1=0.3, r2=3.33, s1=0.02, s2=0.4):
        self.prob = prob
        self.r1 = r1
        self.r2 = r2
        self.s1 = s1
        self.s2 = s2

    def __call__(self, img):
        if (random.random() >= self.prob):
            return img

        image_area = img.size()[1]*img.size()[2]
        min_area = self.s1*image_area
        max_area = self.s2*image_area

        x = img.size()[2]
        y = img.size()[1]

        for _ in range(100):
            aspect_ratio = random.uniform(self.r1, self.r2)
            area = random.uniform(min_area, max_area)

            We = int(round(math.sqrt(area * aspect_ratio)))
            He = int(round(math.sqrt(area / aspect_ratio)))

            if We < x and He < y:
                y1 = random.randint(0, y - He)
                x1 = random.randint(0, x - We)

                img[0, y1:y1 + He, x1:x1 + We] = 0.485
                img[1, y1:y1 + He, x1:x1 + We] = 0.456
                img[2, y1:y1 + He, x1:x1 + We] = 0.406

                return img
