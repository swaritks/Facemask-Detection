import cv2
import torch
import sys
from torchvision import transforms

sys.path.append(".") 
from models.cnn import CNN

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("models/facemask_cnn.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)  

class_names = ['with_mask', 'without_mask']  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,        
        minNeighbors=3,        
        minSize=(30, 30),       
        maxSize=(300, 300),     
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,   
            minNeighbors=2,    
            minSize=(20, 20),   
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        face_tensor: torch.Tensor = transform(face_rgb)  # type: ignore
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        mask_prob = float(probs[0, 0].item()) * 100
        no_mask_prob = float(probs[0, 1].item()) * 100
        
        predicted_class = class_names[int(predicted.item())]
        max_confidence = float(confidence.item()) * 100

        display_text = f"{predicted_class}: {max_confidence:.1f}%"
        detailed_text = f"Mask: {mask_prob:.1f}% | No Mask: {no_mask_prob:.1f}%"

        color = (0, 255, 0) if predicted_class == 'mask' else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, detailed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Facemask Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
