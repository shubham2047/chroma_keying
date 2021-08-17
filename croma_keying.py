import cv2
import numpy as np

video_path = './imgs/sample.mp4'
background = cv2.imread('./imgs/weather.png', cv2.IMREAD_COLOR)

video_capture  = cv2.VideoCapture(video_path)

corner = None
opposite_corner = None
image_patch = None
frame = None
foreground = None
new_background = None
alpha = None
is_enabled = 0
tola = 40
tolb = 70
height = None
Width = None
max_dist = np.sqrt(2)
background_color = None
frameYCrCb = None
softness = 0

def backgroundSelector(action,x,y,flag,userdata):
    global corner, opposite_corner, image_patch, frameYCrCb,background_color
    if action == cv2.EVENT_LBUTTONDOWN:
        corner = (x,y)
    elif action == cv2.EVENT_LBUTTONUP:
        opposite_corner = (x,y)
        image_patch = frameYCrCb[min(corner[1],opposite_corner[1]):max(corner[1],opposite_corner[1]),min(corner[0],opposite_corner[0]):max(corner[0],opposite_corner[0]),1:]
        background_color = np.mean(image_patch, axis=0)
        background_color = np.mean(background_color, axis=0)
    

def enableChromaKeying(*args):
    global is_enabled
    is_enabled = args[0]


def tolaSelector(*args):
    global tola
    tola = args[0]


def tolbSelector(*args):
    global tolb
    tolb = args[0]

def distanceCalculator(frameYCrCb):
    global background_color
    dist = np.linalg.norm(cv2.merge(((frameYCrCb[:,:,1] - background_color[0]),(frameYCrCb[:,:,2] - background_color[1]))),axis=-1)
    return dist
    

def softnessSelector(*args):
    global softness
    softness = args[0]

cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('window',backgroundSelector)
cv2.createTrackbar('Enable Croma keying :', 'window', is_enabled, 1, enableChromaKeying)
cv2.createTrackbar('Tolerance a :', 'window', tola, 100, tolaSelector)
cv2.createTrackbar('Tolerance b :', 'window', tolb, 100, tolbSelector)
cv2.createTrackbar('Softness :', 'window', softness, 4, softnessSelector)

print('1. Select a rectangular color patch using mouse.')
print('2. To enable chroma keying toggle "Enable Croma keying" slider.')
print('3. Set "Tolerance a" and "Tolerance b" as per your need.')
print('4. press "spacebar" to pause video.')
print('5. press "esc" to exit.')


k=None
while video_capture.isOpened():
    ret,frame = video_capture.read()
    
    if ret:
        frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        (height, width)= frame[:,:,0].shape
        
        alpha = np.zeros((height, width),dtype=np.uint8)
        frameYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)/255
        if background_color is not None:           
            alpha = np.zeros((height, width),dtype=np.float32)
            dist = distanceCalculator(frameYCrCb)
            tolb = max(tolb,tola)
            below_tola = dist<=(tola*max_dist/100)
            above_tolb = dist>=(tolb*max_dist/100)
            between_tola_tolb = np.bitwise_and(np.bitwise_not(below_tola),np.bitwise_not(above_tolb))
            alpha[below_tola] = 0
            alpha[above_tolb] = 1
            alpha[between_tola_tolb] = (alpha[between_tola_tolb] - (tola*max_dist/100))/((tolb*max_dist/100) - (tola*max_dist/100))
            ksize = 3+2*softness
            alpha = cv2.GaussianBlur(alpha,(ksize,ksize),0)
            if is_enabled:
                alpha_merge = cv2.merge((alpha,alpha,alpha))
                frame_float = frame/255.0
                background_float = cv2.resize(background,dsize=(width,height))/255.0
                frame = cv2.add(255*cv2.multiply(frame_float,alpha_merge,dtype=cv2.CV_64F),255*cv2.multiply(1-alpha_merge,background_float,dtype=cv2.CV_64F),dtype=cv2.CV_8U)
            alpha = np.uint8(alpha*255)
        frame = np.hstack((frame,cv2.merge((alpha,alpha,alpha))))
        cv2.imshow('window',frame)
        k = cv2.waitKey(int(1000/video_capture.get(cv2.CAP_PROP_FPS)))
        if k == 32:
            k = cv2.waitKey(0)
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
video_capture.release()

