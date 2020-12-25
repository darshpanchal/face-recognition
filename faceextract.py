import cv2

def faceextract(img, cascader):
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height) 
    img = cv2.resize(img, dim)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    try:
        # faces = cascader.detectMultiScale(gray, 1.1, 4)
        faces = cascader.detectMultiScale(gray)
        editedfaces = []
        for face in faces:
            print(face)
            x,y,w,h = face
            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (224,224))
            editedfaces.append(img)
    except:
        editedfaces = []
    # x,y,w,h = faces[0]
    return editedfaces