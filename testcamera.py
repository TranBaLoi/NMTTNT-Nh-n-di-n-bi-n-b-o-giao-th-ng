import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPU detected, running on CPU.")

threshold = 0.7  
font = cv2.FONT_ITALIC

def detect_traffic_signs(image):
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Màu xanh dương (cho biển báo hiệu lệnh)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Tạo mask cho các màu
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Kết hợp các mask
    mask = cv2.bitwise_or(mask_red, mask_blue)
    
    # Áp dụng morphology để loại bỏ nhiễu
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours theo kích thước và hình dạng
    sign_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:  # Giảm kích thước tối thiểu để phát hiện biển báo nhỏ hơn
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            # Kiểm tra tỷ lệ cạnh để đảm bảo hình vuông/chữ nhật hợp lý
            if 0.5 <= aspect_ratio <= 2.0:
                # Thêm điểm tin cậy của contour
                confidence = min(area / 10000.0, 1.0)  # Normalize area as confidence
                sign_regions.append((x, y, w, h, confidence))
    
    # Sắp xếp các vùng theo độ tin cậy (diện tích)
    sign_regions.sort(key=lambda x: x[4], reverse=True)
    return sign_regions

def draw_detection(image, x, y, w, h, class_name, probability, color=(0, 255, 0)):
    # Vẽ khung
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    # Tạo background cho text để dễ đọc
    text = f"{class_name} ({probability:.1f}%)"
    (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 2)
    cv2.rectangle(image, (x, y-40), (x + text_width, y), color, -1)
    
    # Viết text
    cv2.putText(image, text, (x, y-10), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 200)

model = load_model('my_model.h5')

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Gioi han toc do 20 km/h'
    elif classNo == 1:
        return 'Gioi han toc do 30 km/h'
    elif classNo == 2:
        return 'Gioi han toc do 50 km/h'
    elif classNo == 3:
        return 'Gioi han toc do 60 km/h'
    elif classNo == 4:
        return 'Gioi han toc do 70 km/h'
    elif classNo == 5:
        return 'Gioi han toc do 80 km/h'
    elif classNo == 6:
        return 'Ket thuc gioi han toc do 80 km/h'
    elif classNo == 7:
        return 'Gioi han toc do 100 km/h'
    elif classNo == 8:
        return 'Gioi han toc do 120 km/h'
    elif classNo == 9:
        return 'Cam vuot'
    elif classNo == 10:
        return 'Cam vuot doi voi xe tren 3.5 tan'
    elif classNo == 11:
        return 'Uu tien tai nga tu ke tiep'
    elif classNo == 12:
        return 'Duong uu tien'
    elif classNo == 13:
        return 'Nhuong duong'
    elif classNo == 14:
        return 'Dung lai'
    elif classNo == 15:
        return 'Cam tat ca phuong tien'
    elif classNo == 16:
        return 'Cam phuong tien tren 3.5 tan'
    elif classNo == 17:
        return 'Cam vao'
    elif classNo == 18:
        return 'Canh bao chung'
    elif classNo == 19:
        return 'Khuc cua nguy hiem ben trai'
    elif classNo == 20:
        return 'Khuc cua nguy hiem ben phai'
    elif classNo == 21:
        return 'Khuc cua kep'
    elif classNo == 22:
        return 'Duong xoc'
    elif classNo == 23:
        return 'Duong tron'
    elif classNo == 24:
        return 'Duong hep ben phai'
    elif classNo == 25:
        return 'Cong truong'
    elif classNo == 26:
        return 'Den giao thong'
    elif classNo == 27:
        return 'Nguoi di bo'
    elif classNo == 28:
        return 'Tre em qua duong'
    elif classNo == 29:
        return 'Xe dap qua duong'
    elif classNo == 30:
        return 'Canh bao bang tuyet'
    elif classNo == 31:
        return 'Dong vat hoang da qua duong'
    elif classNo == 32:
        return 'Ket thuc tat ca gioi han toc do va cam vuot'
    elif classNo == 33:
        return 'Re phai phia truoc'
    elif classNo == 34:
        return 'Re trai phia truoc'
    elif classNo == 35:
        return 'Chi di thang'
    elif classNo == 36:
        return 'Di thang hoac re phai'
    elif classNo == 37:
        return 'Di thang hoac re trai'
    elif classNo == 38:
        return 'Giua ben phai'
    elif classNo == 39:
        return 'Giua ben trai'
    elif classNo == 40:
        return 'Vong xuyen bat buoc'
    elif classNo == 41:
        return 'Ket thuc cam vuot'
    elif classNo == 42:
        return 'Ket thuc cam vuot doi voi xe tren 3.5 tan'

while True:
    success, imgOrignal = cap.read()
    if not success:
        break
        
    # Phát hiện các vùng có khả năng là biển báo
    sign_regions = detect_traffic_signs(imgOrignal)
    
    # Theo dõi số lượng biển báo đã phát hiện
    detected_signs = 0
    
    # Xử lý từng vùng phát hiện được
    for (x, y, w, h, confidence) in sign_regions:
        if detected_signs >= 3:  # Giới hạn số lượng biển báo hiển thị cùng lúc
            break
            
        # Cắt vùng chứa biển báo
        sign_roi = imgOrignal[y:y+h, x:x+w]
        
        # Xử lý ảnh biển báo để nhận dạng
        if sign_roi.size > 0:
            img = cv2.resize(sign_roi, (32, 32))
            img = preprocessing(img)
            img = img.reshape(1, 32, 32, 1)
            
            # Dự đoán loại biển báo
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)
            
            # Vẽ kết quả nếu độ tin cậy cao
            if probabilityValue > threshold:
                class_name = getCalssName(classIndex[0])
                print(f"Detected: {class_name} with probability: {probabilityValue * 100:.2f}%")
                draw_detection(imgOrignal, x, y, w, h, class_name, probabilityValue * 100)
                detected_signs += 1
    
    # Hiển thị số lượng biển báo đã phát hiện
    # cv2.putText(imgOrignal, f"Signs detected: {detected_signs}", 
    #             (10, 30), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("AI", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()