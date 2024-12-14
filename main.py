import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('matttest1.jpg')
assert img is not None, 'Image not found'

plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_height = 512 
new_width = int((new_height / img.shape[0] * img.shape[1]))
resized_img = cv2.resize(gray_img, (new_width, new_height))
cv2.imwrite('resized_image.jpg', resized_img)

sigma = 2
blurred_img = cv2.GaussianBlur(resized_img, (0,0), sigma)

mean_val = np.mean(blurred_img)
std_val = np.std(blurred_img)
low_thresh = max(0, (mean_val - std_val)/255)
high_thresh = min(1, (mean_val + std_val)/255)
low_thresh = max(0.1, min(low_thresh, 0.2))
high_thresh = max(0.2, min(high_thresh, 0.3))

edges = cv2.Canny(blurred_img, int(low_thresh*255), int(high_thresh*255))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10,5))
plt.imshow(closed_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

contour, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contour, key=cv2.contourArea)

paper_mask = np.zeros_like(closed_edges)
cv2.drawContours(paper_mask, [largest_contour], -1, (255), -1)

flitered_edges = cv2.bitwise_and(closed_edges, paper_mask)

lines = cv2.HoughLines(flitered_edges, 1, np.pi/180, threshold=100)

hough_img = cv2.cvtColor(flitered_edges, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB))
plt.title("Hough Transform Lines")
plt.axis("off")
plt.show()

epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) == 4:
    corners = approx.reshape(4, 2)
else:
    x, y, w, h = cv2.boundingRect(largest_contour)
    corners = np.array([[x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]], dtype=np.float32)

corners = np.float32(corners)

corners_sorted = corners[np.lexsort((corners[:,0], corners[:,1]))]
top_two = corners_sorted[:2]
bottom_two = corners_sorted[2:]

top_left = top_two[np.argmin(top_two[:,0])]
top_right = top_two[np.argmax(top_two[:,0])]
bottom_left = bottom_two[np.argmin(bottom_two[:,0])]
bottom_right = bottom_two[np.argmax(bottom_two[:,0])]

ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

for corner in ordered_corners:
    x, y = corner
    cv2.circle(resized_img, (int(x), int(y)), 5, 255, -1)

plt.figure(figsize=(10, 5))
plt.imshow(resized_img, cmap="gray")
plt.title("Detected Corners")
plt.axis("off")
plt.show()

target_width = 512
target_height = int((target_width / new_width) * new_height)
target_corner = np.array([[0, 0],
                          [target_width - 1, 0],
                          [target_width - 1, target_height - 1],
                          [0, target_height - 1]], dtype=np.float32)

scale_x = img.shape[1] / new_width
scale_y = img.shape[0] / new_height

original_corners = np.zeros_like(ordered_corners, dtype=np.float32)
original_corners[:, 0] = ordered_corners[:, 0] * scale_x
original_corners[:, 1] = ordered_corners[:, 1] * scale_y

H, _ = cv2.findHomography(original_corners, target_corner)

rectified_img = cv2.warpPerspective(img, H, (target_width, target_height))

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
plt.title("Rectified Image (High Quality)")
plt.axis('off')
plt.show()

# Now convert to grayscale for a "scanned" look without losing text clarity
rectified_gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)

# Optional: Apply adaptive thresholding to enhance text readability
rectified_thresh = cv2.adaptiveThreshold(rectified_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(10,5))
plt.imshow(rectified_gray, cmap='gray')
plt.title("Rectified Grayscale Image (High Quality)")
plt.axis('off')
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(rectified_thresh, cmap='gray')
plt.title("Rectified 'Scanned' Image")
plt.axis('off')
plt.show()
