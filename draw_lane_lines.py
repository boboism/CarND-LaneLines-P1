import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import math

gaussian_kernel_size = 3

canny_low_threshold  = 50
canny_high_threshold = 150

hough_rho             = 2
hough_theta           = np.pi/180
hough_threshold       = 15
hough_min_line_length = 10
hough_max_line_gap    = 20

mask_trap_top_ratio    = 0.07
mask_trap_bottom_ratio = 0.85
mask_trap_height_ratio = 0.4

# slope=(y2-y2)/(x2-x1) , threshold for lane lines, only accept slopes >= threshold
slope_threshold = 0.5

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    right_lines = []
    left_lines  = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # advoid divided by 0 exception
        if x1 == x2:
            continue

        slope = (y2-y1)/(x2-x1)
        # only accept slopes >= threshold
        if abs(slope) < slope_threshold:
            continue

        ## suppose right line should be on the right side of the image
        ## and the left line should be on the left side of the image.
        ## don't know wheter it's ok for the vehicle drive accross the
        ## the lanes
        img_center_x = img.shape[1]/2
        if slope > 0 and x1 > img_center_x and x2 > img_center_x:
            right_lines.append(line[0])
        elif slope < 0 and x1 < img_center_x and x2 < img_center_x:
            left_lines.append(line[0])

    right_m, right_b = 1, 1
    # collect right lines x and y sets for least-squares curve-fitting calculating
    right_lines_x    = [x1 for x1, y1, x2, y2 in right_lines] + [x2 for x1, y1, x2, y2 in right_lines]
    right_lines_y    = [y1 for x1, y1, x2, y2 in right_lines] + [y2 for x1, y1, x2, y2 in right_lines]

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b

    left_m, left_b = 1, 1
    # collect left lines x and y sets for least-squares curve-fitting calculating
    left_lines_x   = [x1 for x1, y1, x2, y2 in left_lines] + [x2 for x1, y1, x2, y2 in left_lines]
    left_lines_y   = [y1 for x1, y1, x2, y2 in left_lines] + [y2 for x1, y1, x2, y2 in left_lines]

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b

    y1 = img.shape[0]
    y2 = int(img.shape[0]*(1-mask_trap_height_ratio))

    right_x1 = int((y1-right_b)/right_m)
    right_x2 = int((y2-right_b)/right_m)

    left_x1 = int((y1-left_b)/left_m)
    left_x2 = int((y2-left_b)/left_m)

    if len(right_lines_x) > 0:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if len(left_lines_x) > 0:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #line_img = np.zeros((*img.shape, 3), dtype=np.uint8) 
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def mask_img(img, top_ratio = 0.07, bottom_ratio = 0.85, height_ratio = 0.4):
    img_width  = img.shape[1]
    img_height = img.shape[0]

    mask_bottom_left  = [img_width*(1-bottom_ratio)//2, img_height]
    mask_top_left     = [img_width*(1-top_ratio)//2,    img_height*(1-height_ratio)]
    mask_bottom_right = [img_width*(1+bottom_ratio)//2, img_height]
    mask_top_right    = [img_width*(1+top_ratio)//2,    img_height*(1-height_ratio)]

    return np.array([[mask_bottom_left, mask_bottom_right, mask_top_right, mask_top_left]], np.int32)

def sanitize_img(img):
    mask_white_min = np.array([200, 200, 200], np.int32)
    mask_white_max = np.array([255, 255, 255], np.int32)
    mask_white = cv2.inRange(img, mask_white_min, mask_white_max)
    sanitized_white_img = cv2.bitwise_and(img, img, mask = mask_white)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_yellow_min = np.array([20, 100, 100])
    mask_yellow_max = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, mask_yellow_min, mask_yellow_max)
    sanitized_yellow_img = cv2.bitwise_and(img, img, mask = mask_yellow)

    result = weighted_img(sanitized_white_img, sanitized_yellow_img, 1., 1., 0.)
    return result

def annotate_img(img):
    img_wip    = sanitize_img(img)
    img_wip    = grayscale(img_wip)
    img_wip    = gaussian_blur(img_wip, gaussian_kernel_size)
    img_wip    = canny(img_wip, canny_low_threshold, canny_high_threshold)
    mask_array = mask_img(img_wip, mask_trap_top_ratio, mask_trap_bottom_ratio, mask_trap_height_ratio)
    img_wip    = region_of_interest(img_wip, mask_array)
    img_anno   = hough_lines(img_wip, hough_rho, hough_theta, hough_threshold, hough_min_line_length, hough_max_line_gap)
    result     = weighted_img(img_anno, img)
    return result

def process_image(input_file, output_file):
    image_input  = mpimg.imread(input_file)
    image_output = annotate_img(image_input)
    plt.imsave(output_file, image_output)

def process_video(input_file, output_file):
    video_input  = VideoFileClip(input_file)
    video_output = video_input.fl_image(annotate_img)
    video_output.write_videofile(output_file, audio=False)

    #cap = cv2.VideoCapture(input_file)
    #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    #out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    #while (cap.isOpened()):
    #    ret, frame = cap.read()
    #    if ret == False:
    #        break
    #    img_annotated = annotate_img(frame)
    #    out.write(img_annotated)
    #cap.release()
    #out.release()


def process_file(input_file, output_file):
    image_extensions = {".jpg", ".png", ".gif"}
    video_extensions = {".mp4", ".mkv", ".avi"}
    is_image = any(input_file.endswith(ext) for ext in image_extensions)
    is_video = any(input_file.endswith(ext) for ext in video_extensions)
    if is_image:
        process_image(input_file, output_file)
    elif is_video:
        process_video(input_file, output_file)

#process_file('./challenge.mp4', './challenge_output.mp4')
#cap = cv2.VideoCapture('./challenge.mp4')
#print(cv2.GetCaptureProperty(cap, cv.CV_CAP_PROP_FPS))
#while (cap.isOpened()):
#    ret, img = cap.read()
#    result = annotate_img(img)
#    cv2.imshow('result', result)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cap.release()
#cv2.destroyAllWindows()

if __name__ == '__main__':
    from optparse import OptionParser

    # Configure command line options
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                    help="Input video/image file")
    parser.add_option("-o", "--output_file", dest="output_file",
                    help="Output video/image file")

    # Get and parse command line options
    options, args = parser.parse_args()

    input_file  = options.input_file
    output_file = options.output_file

    process_file(input_file, output_file)
