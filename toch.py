from xml.parsers.expat import model
import torch

# Assuming 'model' is your trained PyTorch model
# Train your model and then save it
torch.save(model.state_dict(), 'traffic_prediction_model.pth')

print("Model saved successfully.")

