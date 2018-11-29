import sys
import win32gui
import mss
import numpy as np
import time
import cv2
import keyboard
import mouse
import tensorflow as tf

# Window aspect ratio is fixed
# Update if changed
desired_width=1126
desired_height=720
# Margin of error for desired width and height
size_moe=5
verbose=False
# Defines regions of interests displaying numbers
num_roi={
    "enemy_power": (608,300,820,350),
    "your_power": (314,300,512,350),   
}
current_capture= None
recognizer = None
running=False

def get_chars(img):
    """Break an image of numbers into a list of images of characters
    
    Reference: https://stackoverflow.com/questions/50713023/license-plate-character-segmentation-python-opencv
    """
    chars=[]
    _,contours,_=cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # Sort bounding boxes from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for ctr in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = img[y:y+h, x:x+w]
        chars.append(roi)
    return chars

def filter_chars(img):
    """Filter out non-character glyphs based on height/width ratio
    """
    h,w=img.shape
    ratio=h/w
    if ratio<=1 or ratio>2.1:
        return False
    else:
        return True

def conform_size(img):
    """Resize images to conform to the size of MNIST images, which is 28x28
    """
    target_width=20
    target_height=20
    padding_second=4
    h,w=img.shape
    if w>h:
        # Resize without stretching
        new_h=int(np.round(h*target_height/w))
        resized_img = cv2.resize(img, (target_width, new_h))
        # Pad image
        blank=target_height-new_h
        pad_t=int(np.floor(blank/2))
        pad_b=int(np.floor((blank+1)/2))
        padded_img=cv2.copyMakeBorder(resized_img,pad_t,pad_b,0,0,cv2.BORDER_CONSTANT,value=0)
        # Do padding again to add blanks so that the image resembles those from MNIST
        # Otherwise the trained model is destined to fail
        padded_img2=cv2.copyMakeBorder(padded_img,padding_second,padding_second,padding_second,padding_second,cv2.BORDER_CONSTANT,value=0)
    else:
        new_w=int(np.round(w*target_width/h))
        resized_img = cv2.resize(img, (new_w,target_height))
        blank=target_width-new_w
        pad_l=int(np.floor(blank/2))
        pad_r=int(np.floor((blank+1)/2))
        padded_img=cv2.copyMakeBorder(resized_img,0,0,pad_l,pad_r,cv2.BORDER_CONSTANT,value=0)
        padded_img2=cv2.copyMakeBorder(padded_img,padding_second,padding_second,padding_second,padding_second,cv2.BORDER_CONSTANT,value=0)
    # 0-255 to 0-1
    normalized=padded_img2/255
    normalized = normalized.reshape(28, 28, 1)
    return normalized

def run_check():
    while True:
        hwnd=check_window()
        rect=get_rect(hwnd,verbose=verbose)
        if not check_size(*rect):
            adjust_size(hwnd)
            continue
        return (hwnd,rect)

def check_window():
    hwnd = win32gui.FindWindowEx(None, None, None, "BlueStacks")
    if hwnd==0:
        print("\nBlueStacks not running\n\tPress ESC to quit")
        # No interaction allowed if BlueStacks not running or crashed
        ##################################
        ### If crashed, may need to pause
        ##################################
        keyboard.unhook_all()
        keyboard.wait('esc')
        sys.exit()
    else:
        return hwnd

def check_size(x,y,w,h):
    if np.abs(w-desired_width)<size_moe and np.abs(h-desired_height)<size_moe:
        return True
    else:
        return False
    
def get_rect(hwnd,verbose=False):
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x +1
    h = rect[3] - y +1
    if verbose:
        print("Window size: {}\t{}".format(w,h))
        # Prevent stucking in cache upon crash
        sys.stdout.flush()
    return (x,y,w,h)

def adjust_size(hwnd):
    print("Window size changed, resetting...")
    win32gui.MoveWindow(hwnd, 100, 100, desired_width, desired_height, True)
    time.sleep(2)
    
def show_help():
    help_msg="""
    Print cursor location: Ctrl+Shift+`+L
    Resume/Pause: Ctrl+Shift+`+F9
    """
    print(help_msg)
    
def cursor_location():
    hwnd,rect=run_check()
    x,y=mouse.get_position()    
    print("Relative cursor location (starts from 0):\nX: {}, Y: {}".format(x-rect[0],y-rect[1]))
    
def get_roi(key,img):
    global num_roi
    x1,y1,x2,y2=num_roi[key]
    w=x2-x1+1
    h=y2-y1+1
    return img[y1:y1+h,x1:x1+w,:]

def img_preproc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Only look at white parts
    _,bwimg=cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
    return bwimg

def read_roi(key):
    global recognizer
    img_orig=get_roi(key,current_capture)
    cv2.imwrite("roi_shot.png",img_orig)
    img=img_preproc(img_orig)
    characters=get_chars(img)
    # Filter based on height/width ratio
    filt_chars=[img for img in characters if filter_chars(img)]
    
    if len(filt_chars)==0:
        #print("No number found in region of interest")
        return None
    # "Normalized" characters
    norm_chars=[conform_size(char) for char in filt_chars]
    #for char in norm_chars:
    #    cv2.imshow("ABC",char)
    #    cv2.waitKey(0)
    pred_digits=recognizer.predict_classes(np.array(norm_chars),verbose=0)
    pred_chars=[str(x) for x in pred_digits]
    pred_str="".join(pred_chars)
    pred_num=int(pred_str)
    return pred_num

def get_powers():
    enemy_power=read_roi("enemy_power")
    your_power=read_roi("your_power")
    if enemy_power is None or your_power is None:
        print("No squad power detected")
        return
    print("Your power: {}\tEnemy power: {}".format(your_power,enemy_power))
    if your_power-enemy_power>20000:
        print("Possible victory")
    elif enemy_power>your_power:
        print("Possible defeat")
    else:
        print("Outcome unsure")
    print()

def onoff():
    global running
    if running:
        running=False
    else:
        running=True
    
def update_capture(x,y,w,h,verbose=False):
    global current_capture
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(sct.grab(monitor))
        current_capture=img
    
    #bwimg=img_preproc(img_power)
    #cv2.imshow("OpenCV/Numpy normal", bwimg)
    #while True:
    #    if cv2.waitKey(25) & 0xFF == ord("q"):
    #        cv2.destroyAllWindows()
    #        break

if __name__ == '__main__':
    # Print help info
    print("Press Ctrl+Shift+`+F1 to show help")
    
    # Register hotkeys
    keyboard.add_hotkey('ctrl+shift+`+f1', show_help)
    keyboard.add_hotkey('ctrl+shift+`+l', cursor_location)
    keyboard.add_hotkey('ctrl+shift+`+f9', onoff)
    recognizer = tf.keras.models.load_model("models/castle_cnn.h5")
    while True:
        if running:
            hwnd,rect=run_check()
            update_capture(*rect)
            get_powers()
        time.sleep(1)
 

# Something to test each update
# Is the window open?
# Is the window on top?
# Is the window at the desired size?
