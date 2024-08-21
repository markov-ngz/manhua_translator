

#---1. SETUP -------------------------------------------------------------------------------------------------

#1.1. Load image
# path 
path = 'images/manhua_3.PNG'

# Reading an image in default mode 
image = cv2.imread(path) 
# image PIL 
image_PIL = Image.open(path).convert("RGB")

img_read = image.copy()


# 1.2 Font parameters
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1
# Green color in BGR 
color = (0, 255, 0) 
# Line thickness of 2 px 
thickness = 2

# 1.3 Functions to save clicking positions

rectangle = []
rectangles = []
points = []

def rectangle_clicks(event, x, y, flags, param):
   """
   """
   global rectangle, rectangles
   if event == cv2.EVENT_LBUTTONDOWN:
        rectangle.append(x)
        rectangle.append(y)
        # 4 coordinates of a rectangles => load them 
        if len(rectangle) == 4:
            rectangles.append(rectangle)
            rectangle = []

def point_clicks(event, x, y, flags, param):
   """
   """
   global points
   if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

#--- 2. PROCEED -----------------------------------------------------------------------------------------------------


# Using cv2.putText() method 
img_read = cv2.putText(img_read, 'Click to draw', org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

# Displaying the image 
cv2.imshow('image', img_read) 

#---Rectangle Loop-----------------------------------------------------------------------------------------------
while True :
    cv2.setMouseCallback('image', rectangle_clicks)
    # waits for user to press key 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# closing all open windows 
cv2.destroyAllWindows() 

# Drawing the rectangles on a new image
image_rec= image.copy()

for rectangle in rectangles : 
    x, y , z ,v = rectangle
    image_rec = cv2.rectangle(image_rec,(x,y),(z,v), (0, 255, 0), 2)


cv2.imshow('image_rec',image_rec)

#---Point's Loop-------------------------------------------------------------------------------------------------

while True :
    cv2.setMouseCallback('image_rec', point_clicks)
    # waits for user to press key 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for point in points:
    x ,y = point
    image_rec = cv2.circle(image_rec, (x,y), radius=2, color=(0, 255, 0), thickness=-1)

cv2.destroyAllWindows() 


cv2.imshow('image_rec',image_rec)

cv2.waitKey(0)
cv2.destroyAllWindows()

confirm = input(" Confirmer votre choix ? [y/n]")

confirm_drawings = False

if confirm == "y" or confirm == "Y":
    confirm_drawings = True



if confirm_drawings : 
    input_point = np.array(points) # x, y 
    input_label = np.array([1 for i in range(len(points))])
    input_boxes = [rectangles] #( x, y ), (z, w )
    # 60 sec approx on poor cpu 
    texts = manhua_translation(sam_processor,sam,translator,image_PIL, input_boxes,input_point, input_label)
    new_image = write_all_texts(texts,image_PIL)