import cv2
from PIL import Image
import numpy as np


class GUI():

    def __init__(self) -> None:
        # font 
        self.font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        self.org = (50, 50) 
        # fontScale 
        self.fontScale = 1
        # Green color in BGR 
        self.color = (0, 255, 0) 
        # Line thickness of 2 px 
        self.thickness = 2
        # Line type
        self.line_type = cv2.LINE_AA

        # 1.3 Functions to save clicking positions

        self.rectangle = []
        self.rectangles = []
        self.points = []        

    def rectangle_clicks(self,event, x, y, flags, param):
        """
        """
        if event == cv2.EVENT_LBUTTONDOWN:
                self.rectangle.append(x)
                self.rectangle.append(y)
                # 4 coordinates of a rectangles => load them 
                if len(self.rectangle) == 4:
                    self.rectangles.append(self.rectangle)
                    self.rectangle = []

    def point_clicks(self,event, x, y, flags, param):
            """
            """

            if event == cv2.EVENT_LBUTTONDOWN:
                    self.points.append([x, y])

    def get_inputs(self,path:str)->tuple:
        """
        From a given path image, trigger an interaction with the image to set the boxes and inputs 

        Return image , input_boxes , input_point 
            PIL.Image.Image , list, np.array
        """
        # image cv2 
        image = cv2.imread(path) 
        
        # image PIL 
        image_PIL = Image.open(path).convert("RGB")
        
        img_read = image.copy()

        #--- 2. PROCEED -----------------------------------------------------------------------------------------------------


        # Print indication
        img_read = cv2.putText(img_read, 
                               'Click 2x to draw,\'q\' to confirm', 
                               self.org, 
                               self.font,  
                               self.fontScale, 
                               self.color, 
                               self.thickness, 
                               self.line_type) 

        # Displaying the image 
        cv2.imshow('image', img_read) 

        #---Rectangle Loop-----------------------------------------------------------------------------------------------
        while True :
            cv2.setMouseCallback('image', self.rectangle_clicks)
            # waits for user to press key 'q' 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # closing all open windows 
        cv2.destroyAllWindows() 

        # Drawing the rectangles on a new image
        image_rec= image.copy()

        for rectangle in self.rectangles : 
            x, y , z ,v = rectangle
            image_rec = cv2.rectangle(image_rec,(x,y),(z,v), (0, 255, 0), 2)

        image_rec = cv2.putText(image_rec, 
                               'Click 1x for the point the q ', 
                               self.org, 
                               self.font,  
                               self.fontScale, 
                               self.color, 
                               self.thickness, 
                               self.line_type) 
        cv2.imshow('image_rec',image_rec)

        #---Point's Loop-------------------------------------------------------------------------------------------------

        while True :
            cv2.setMouseCallback('image_rec', self.point_clicks)
            # waits for user to press key 'q' 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for point in self.points:
            x ,y = point
            image_rec = cv2.circle(image_rec, (x,y), radius=2, color=(0, 255, 0), thickness=-1)

        cv2.destroyAllWindows() 

        image_rec = cv2.putText(image_rec, 
                               'this is your choices', 
                               self.org, 
                               self.font,  
                               self.fontScale, 
                               self.color, 
                               self.thickness, 
                               self.line_type) 

        cv2.imshow('image_rec',image_rec)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        confirm = input(" Confirmer votre choix ? [y/n]")

        confirm_drawings = False

        if confirm == "y" or confirm == "Y":
            confirm_drawings = True

        if confirm_drawings : 
            input_point = np.array(self.points) # x, y 
            # input_label = np.array([1 for i in range(len(points))])
            input_boxes = [self.rectangles] #( x, y ), (z, w )

            return image_PIL , input_boxes, input_point
        else : 
             return None, None, None