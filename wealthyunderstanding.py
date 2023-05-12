
import cv2
import numpy as np
from numpy import asarray
from ultralytics import YOLO
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
from shapely.geometry import Polygon



# input: an image
# directive: Use a pre-trained NN model
# return: coordinates of the chessboard corner
def find_corners(image):
    
    # a model trained to find the chessboard's corners
    model_trained = YOLO("cornerfinder.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.1, save_txt=True, save=True)

    # corner coordinate return
    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:, 0:2]
    corners = order_points(points)
    
    return corners


# input: an image
# directive: Use a pre-trained NN model
# return: the location and class of the identified chess piece above the determined confidence threshold

def piece_identifier(image):
    #put here for easy tweaking
    confidence = 0.3
    model_trained = YOLO("piece_identifier.pt")
    results = model_trained.predict(source=image, line_width=1, conf=confidence, augment=False, save_txt=True, save=True)
    boxes = results[0].boxes
    detections = boxes.xyxy.numpy()

    return detections, boxes


# based on the four corners, perspective transform the image
def four_point_transform(image, pts):
      
    img = Image.open(image)
    image = asarray(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # width of the image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # height of the image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to get a "top-down" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    img = Image.fromarray(warped, "RGB")
    # return the transformed image
    return img

# intersection over union calculation

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


# calculates chessboard grid
def plot_grid_on_transformed_image(image):
    
    corners = np.array([[0,0], 
                    [image.size[0], 0], 
                    [0, image.size[1]], 
                    [image.size[0], image.size[1]]])
    
    corners = order_points(corners)

    figure(figsize=(10, 10), dpi=80)
    imageplot = plt.imshow(image)
    
    #T -> Top       | Felső
    #B -> Bottom    | Alsó
    #R -> Right     | Jobb
    #L -> Left      | Bal
    TL = corners[0]
    TR = corners[1]
    BR = corners[2]
    BL = corners[3]

    def interpolation(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        points = [(x0 + i * dx, y0 + i * dy) for i in range(9)]
        return points

    #"pont-párok" az oldalak meghatározásához és a transzformáláshoz

    ptsT = interpolation(TL, TR)
    ptsL = interpolation(TL, BL)
    ptsR = interpolation(TR, BR)
    ptsB = interpolation(BL, BR)
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="-")
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'bo', linestyle="-" )
        
    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL



def order_points(pts):
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left

    rect = np.zeros((4, 2), dtype="float32")
    sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum)]
    rect[2] = pts[np.argmax(sum)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# connects detected piece to the right square

def connect_square_to_detection(detections, square):
    #we define the dictionary
    #Legend:
    #0 -> Black (b)ishop
    #1 -> Black (k)ing
    #2 -> Black k(n)ight
    #3 -> Black (p)awn
    #4 -> Black (q)ueen
    #5 -> Black (r)ook
    #6 -> White (B)ishop
    #7 -> White (K)ing
    #8 -> White K(n)ight
    #9 -> White (P)awn
    #10 -> White (Q)ueen
    #11 -> White (R)ook
    #[sorted alphabetically because that's how the model was trained, didn't care to change]

    di = {0: 'b', 1: 'k', 2: 'n',
      3: 'p', 4: 'q', 5: 'r', 
      6: 'B', 7: 'K', 8: 'N',
      9: 'P', 10: 'Q', 11: 'R'}

    intersection_listing=[]
    
    for i in detections:

        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]

        # Turns out cutting the entire chessboard is not a perfect idea as it will decimate the 8th rank pieces
        # But we accept whatever guess it makes based on what's visible and put it on the screen anyway
        if box_y4 - box_y1 > 60:
            box_complete = np.array([[box_x1,box_y1+40], [box_x2, box_y2+40], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])

        intersection_listing.append(calculate_iou(box_complete, square))

    num = intersection_listing.index(max(intersection_listing))

    piece = boxes.cls[num].tolist()

    #This part tries to correct misplaced pieces and overhanging
    #if there is enough intersection between a suare and a piece found there, the piece probably belongs there
    if max(intersection_listing) > 0.25:
        piece = boxes.cls[num].tolist()
        return di[piece]
    #if it just barely hangs over/on the line, it's probably empty
    else:
        piece = "empty"
        return piece
    #Problem: Tall pieces "occupying multiple squares" despite attempted correction, further negative points for misinterpreting


#MAIN:
#TODO: API hook here to receive pictures from mobile app / servlet [omitted from thesis]
board_image_raw = 'images/bpr.jpg'
#Corners of the image
corners = find_corners(board_image_raw)
#top-down stretched image
top_down_img = four_point_transform(board_image_raw, corners)
#Points from the Top, and points from the Left
ptsT, ptsL = plot_grid_on_transformed_image(top_down_img)

detections, boxes = piece_identifier(top_down_img)

#we need two groups of 9-lines laterally to the other group to create an 8x8 grid

# Lines for Files
xA = ptsT[0][0]
xB = ptsT[1][0]
xC = ptsT[2][0]
xD = ptsT[3][0]
xE = ptsT[4][0]
xF = ptsT[5][0]
xG = ptsT[6][0]
xH = ptsT[7][0]
xI = ptsT[8][0]
# Lines for Ranks
y8 = ptsL[0][1]
y7 = ptsL[1][1]
y6 = ptsL[2][1]
y5 = ptsL[3][1]
y4 = ptsL[4][1]
y3 = ptsL[5][1]
y2 = ptsL[6][1]
y1 = ptsL[7][1]
y0 = ptsL[8][1]

# We determine the squares file-by-file by the intersections of the lines

#File A
A8 = np.array([[xA, y8], [xB, y8], [xB, y7], [xA, y7]])
A7 = np.array([[xA, y7], [xB, y7], [xB, y6], [xA, y6]])
a6 = np.array([[xA, y6], [xB, y6], [xB, y5], [xA, y5]])
A5 = np.array([[xA, y5], [xB, y5], [xB, y4], [xA, y4]])
A4 = np.array([[xA, y4], [xB, y4], [xB, y3], [xA, y3]])
A3 = np.array([[xA, y3], [xB, y3], [xB, y2], [xA, y2]])
A2 = np.array([[xA, y2], [xB, y2], [xB, y1], [xA, y1]])
A1 = np.array([[xA, y1], [xB, y1], [xB, y0], [xA, y0]])
#File B
B8 = np.array([[xB, y8], [xC, y8], [xC, y7], [xB, y7]])
B7 = np.array([[xB, y7], [xC, y7], [xC, y6], [xB, y6]])
B6 = np.array([[xB, y6], [xC, y6], [xC, y5], [xB, y5]])
B5 = np.array([[xB, y5], [xC, y5], [xC, y4], [xB, y4]])
B4 = np.array([[xB, y4], [xC, y4], [xC, y3], [xB, y3]])
B3 = np.array([[xB, y3], [xC, y3], [xC, y2], [xB, y2]])
B2 = np.array([[xB, y2], [xC, y2], [xC, y1], [xB, y1]])
B1 = np.array([[xB, y1], [xC, y1], [xC, y0], [xB, y0]])
#File C
C8 = np.array([[xC, y8], [xD, y8], [xD, y7], [xC, y7]])
C7 = np.array([[xC, y7], [xD, y7], [xD, y6], [xC, y6]])
C6 = np.array([[xC, y6], [xD, y6], [xD, y5], [xC, y5]])
C5 = np.array([[xC, y5], [xD, y5], [xD, y4], [xC, y4]])
C4 = np.array([[xC, y4], [xD, y4], [xD, y3], [xC, y3]])
C3 = np.array([[xC, y3], [xD, y3], [xD, y2], [xC, y2]])
C2 = np.array([[xC, y2], [xD, y2], [xD, y1], [xC, y1]])
C1 = np.array([[xC, y1], [xD, y1], [xD, y0], [xC, y0]])
#File D
D8 = np.array([[xD, y8], [xE, y8], [xE, y7], [xD, y7]])
D7 = np.array([[xD, y7], [xE, y7], [xE, y6], [xD, y6]])
D6 = np.array([[xD, y6], [xE, y6], [xE, y5], [xD, y5]])
D5 = np.array([[xD, y5], [xE, y5], [xE, y4], [xD, y4]])
D4 = np.array([[xD, y4], [xE, y4], [xE, y3], [xD, y3]])
D3 = np.array([[xD, y3], [xE, y3], [xE, y2], [xD, y2]])
D2 = np.array([[xD, y2], [xE, y2], [xE, y1], [xD, y1]])
D1 = np.array([[xD, y1], [xE, y1], [xE, y0], [xD, y0]])
#File E
E8 = np.array([[xE, y8], [xF, y8], [xF, y7], [xE, y7]])
E7 = np.array([[xE, y7], [xF, y7], [xF, y6], [xE, y6]])
E6 = np.array([[xE, y6], [xF, y6], [xF, y5], [xE, y5]])
E5 = np.array([[xE, y5], [xF, y5], [xF, y4], [xE, y4]])
E4 = np.array([[xE, y4], [xF, y4], [xF, y3], [xE, y3]])
E3 = np.array([[xE, y3], [xF, y3], [xF, y2], [xE, y2]])
E2 = np.array([[xE, y2], [xF, y2], [xF, y1], [xE, y1]])
E1 = np.array([[xE, y1], [xF, y1], [xF, y0], [xE, y0]])
#File F
F8 = np.array([[xF, y8], [xG, y8], [xG, y7], [xF, y7]])
F7 = np.array([[xF, y7], [xG, y7], [xG, y6], [xF, y6]])
F6 = np.array([[xF, y6], [xG, y6], [xG, y5], [xF, y5]])
F5 = np.array([[xF, y5], [xG, y5], [xG, y4], [xF, y4]])
F4 = np.array([[xF, y4], [xG, y4], [xG, y3], [xF, y3]])
F3 = np.array([[xF, y3], [xG, y3], [xG, y2], [xF, y2]])
F2 = np.array([[xF, y2], [xG, y2], [xG, y1], [xF, y1]])
F1 = np.array([[xF, y1], [xG, y1], [xG, y0], [xF, y0]])
#File G
G8 = np.array([[xG, y8], [xH, y8], [xH, y7], [xG, y7]])
G7 = np.array([[xG, y7], [xH, y7], [xH, y6], [xG, y6]])
G6 = np.array([[xG, y6], [xH, y6], [xH, y5], [xG, y5]])
G5 = np.array([[xG, y5], [xH, y5], [xH, y4], [xG, y4]])
G4 = np.array([[xG, y4], [xH, y4], [xH, y3], [xG, y3]])
G3 = np.array([[xG, y3], [xH, y3], [xH, y2], [xG, y2]])
G2 = np.array([[xG, y2], [xH, y2], [xH, y1], [xG, y1]])
G1 = np.array([[xG, y1], [xH, y1], [xH, y0], [xG, y0]])
#File H
H8 = np.array([[xH, y8], [xI, y8], [xI, y7], [xH, y7]])
H7 = np.array([[xH, y7], [xI, y7], [xI, y6], [xH, y6]])
H6 = np.array([[xH, y6], [xI, y6], [xI, y5], [xH, y5]])
H5 = np.array([[xH, y5], [xI, y5], [xI, y4], [xH, y4]])
H4 = np.array([[xH, y4], [xI, y4], [xI, y3], [xH, y3]])
H3 = np.array([[xH, y3], [xI, y3], [xI, y2], [xH, y2]])
H2 = np.array([[xH, y2], [xI, y2], [xI, y1], [xH, y1]])
H1 = np.array([[xH, y1], [xI, y1], [xI, y0], [xH, y0]])

# put in a "2d array" to FEN-tansform easily

FEN_Coordinates = [[A8, B8, C8, D8, E8, F8, G8, H8],
                   [A7, B7, C7, D7, E7, F7, G7, H7],
                   [a6, B6, C6, D6, E6, F6, G6, H6],
                   [A5, B5, C5, D5, E5, F5, G5, H5],
                   [A4, B4, C4, D4, E4, F4, G4, H4],
                   [A3, B3, C3, D3, E3, F3, G3, H3],
                   [A2, B2, C2, D2, E2, F2, G2, H2],
                   [A1, B1, C1, D1, E1, F1, G1, H1]]

board_FEN = []
corrected_FEN = []
FEN_Table_Final = []

for line in FEN_Coordinates:
    line_to_FEN = []
    for square in line:
        piece_on_square = connect_square_to_detection(detections, square)    
        line_to_FEN.append(piece_on_square)
    corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
    board_FEN.append(corrected_FEN)

#print(board_FEN)

#we make a copy of the board
rotate_corrected_FEN = board_FEN

#if we find a white pawn on the 8th rank (where it cannot be as it should promote) we rotate the image by 90 degrees counter-clockwise
for x in range(len(board_FEN[1])):
    if board_FEN[1][x] == "P":
        rotate_corrected_FEN = list(reversed(list(zip (*board_FEN[::-1]))))

#we now mirror the board and it should be correctly aligned
rotate_corrected_FEN = np.fliplr(rotate_corrected_FEN)

print(np.matrix(rotate_corrected_FEN))


FEN_Table_Final = [''.join(line) for line in rotate_corrected_FEN]

FEN = '/'.join(FEN_Table_Final)

#Returns a link on the output that opens Lichess immediately with the found position
print("The position can be viewed in digital form under the following link:")
print("https://lichess.org/analysis/" + FEN)