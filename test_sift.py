import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузите два изображения
image1 = cv2.imread('phototest.jpg')  # Используйте правильный путь к изображению
image2 = cv2.imread('phototest2.jpg')  # Используйте правильный путь к изображению

# Преобразуем изображения в серые оттенки
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Инициализация SIFT
sift = cv2.SIFT_create()

# Находим ключевые точки и дескрипторы для обоих изображений
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

# Сопоставляем дескрипторы
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Применяем фильтрацию на основе отношения расстояний
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Отображаем совпадения
img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Покажем изображение с совпадениями
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Matching keypoints between image1 and image2")
plt.show()
