from xml.parsers.expat import model
import torch

torch.save(model.state_dict(), 'traffic_prediction_model.pth')

print("Model saved successfully.")

