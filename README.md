# lane-detection-python-and-opencv


![image](https://github.com/VasuTammisetti/lane-detection-python-and-opencv/assets/59999074/08e83573-bd3d-4c1a-93d3-2530f9ebb3d9)


Creating a basic lane detection system in Python involves using computer vision techniques to identify and highlight lanes on the road in images or videos. Here's a high-level overview of how you might approach building a simple lane detection system using Python and the OpenCV library:

1. **Install OpenCV:**
   First, make sure you have OpenCV installed. You can install it using pip:

   ```
   pip install opencv-python
   ```

2. **Capture Video or Load Image:**
   Decide whether you want to work with a video stream or individual images. You can use OpenCV's `VideoCapture` class to capture video from a camera or load images using the `imread` function.

   ```python
   import cv2

   # For video
   cap = cv2.VideoCapture("video.mp4")

   # For images
   image = cv2.imread("image.jpg")
   ```

3. **Preprocess the Image or Frame:**
   Lane detection involves several preprocessing steps, such as converting the image to grayscale, applying Gaussian blur, and performing edge detection using Canny edge detector.

   ```python
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blurred, threshold1, threshold2)
   ```

4. **Region of Interest (ROI) Selection:**
   Define the region of interest where you expect the lanes to appear. This step helps to reduce the processing load and focus on the relevant parts of the image.

   ```python
   mask = np.zeros_like(edges)
   roi_vertices = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], np.int32)
   cv2.fillPoly(mask, roi_vertices, 255)
   masked_edges = cv2.bitwise_and(edges, mask)
   ```

5. **Hough Transform for Lane Detection:**
   Use the Hough line transform to detect lines in the masked edge image. This will help you identify the lane lines.

   ```python
   lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
   ```

6. **Draw Lane Lines:**
   Once you have the lines detected, you can draw them on the original frame or image to visualize the detected lanes.

   ```python
   line_image = np.zeros_like(frame)
   for line in lines:
       for x1, y1, x2, y2 in line:
           cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
   ```

7. **Combine Lane Image and Original Image:**
   Merge the lane lines image with the original frame using techniques like weighted addition.

   ```python
   combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
   ```

8. **Display the Result:**
   Show the final processed image or video with lane lines highlighted.

   ```python
   cv2.imshow("Lane Detection", combo_image)
   cv2.waitKey(0)  # Wait for a key press to close the window
   cv2.destroyAllWindows()
   ```

Remember, this is a simplified explanation, and a real-world lane detection system might involve more advanced techniques, handling different lighting conditions, curve detection, and more. It's recommended to explore tutorials and resources that provide more in-depth explanations and examples for creating lane detection systems in Python using OpenCV.
