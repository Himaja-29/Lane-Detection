import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def process_frame(frame):
    height, width = frame.shape[:2]

    # Resize the frame to fit the display window
    display_width = 800  # Desired width for display
    display_height = int(height * (display_width / width))
    frame = cv2.resize(frame, (display_width, display_height))
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    roi_vertices = np.array([[(0, height), (width, height), (width // 2, height // 2)]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)

    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    line_image = np.zeros_like(frame)
    draw_lines(line_image, lines)

    # Combine original frame with line image
    lanes_detected = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return lanes_detected

def main(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for lane detection
        processed_frame = process_frame(frame)

        # Display the frame
        cv2.imshow('Lane and Object Detection', processed_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "E:\\Autonomous\\activities\\lane detect.mp4"  # Replace with your video file path
    main(video_path)