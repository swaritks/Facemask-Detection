import cv2
import torch
import sys
sys.path.append(".")  # Add this here

from models.cnn import CNN


model = CNN()
model.load_state_dict(torch.load("models/facemask_cnn.pth"))

model.eval()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    output = model(img)
    label = "Mask" if torch.argmax(output) == 0 else "No Mask"

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Facemask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()