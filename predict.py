import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image


class maskdetection:
    def __init__(self, filename):
        self.filename = filename
        # Check if CUDA is available and set PyTorch to use GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        # Load the pre-trained model
        self.model = models.alexnet()
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=100, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=100, out_features=6, bias=True)
        )
        # Load the trained model state dictionary
        state_dict = torch.load("alexnet_trained_model.pth", map_location=self.device)
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        # Move the model to the appropriate device
        self.model.to(self.device)
        # Set the model to evaluation mode
        self.model.eval()
        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.plant_details = {
            'Bacterial Spot': {
                'desclaimer': 'Remember, these are general solutions and the effectiveness can vary depending on the specific conditions of the plant and the environment. Always consider consulting with a local agricultural extension service or a plant health expert for more personalized advice. 😊',
                'description': 'This disease is caused by bacteria and frequently appears as spots on the leaves. The spots differ in size, color, and in extreme circumstances, they may combine to destroy the entire leaf.',
                'solution': 'Early identification is crucial to effective management of bacterial spot disease. The use of disease free seeds, crop rotation, the elimination of volunteer plants and weed hosts, and the restriction of overhead watering to keep leaves dry are all effective means of preventing the spread of common blight. In addition to these, you can use a copper based fungicide on crops.'
            },
            'Brown Blight': {
                'desclaimer': 'Remember, these are general solutions and the effectiveness can vary depending on the specific conditions of the plant and the environment. Always consider consulting with a local agricultural extension service or a plant health expert for more personalized advice. 😊',
                'description': 'This disease often leads to browning, wilting, and the loss of plant tissue. It is observed that they spread quickly and harm the plant seriously.',
                'solution': 'Control of this disease can be achieved by using disease-free seeds, practicing crop rotation, removing volunteer plants and weed hosts, and limiting the use of overhead watering to keep foilage dry. Good sanitation practices are also important as the bacterium overwinters on host debris and on the soil surface.'
            },
            'Dry': {
                'desclaimer': 'Remember, these are general solutions and the effectiveness can vary depending on the specific conditions of the plant and the environment. Always consider consulting with a local agricultural extension service or a plant health expert for more personalized advice. 😊',
                'description': 'This is a condition, not a disease, in which a plant does not get enough water. In this condition, it is observed that the leaf often turns yellow and stunts the growth.',
                'solution': 'Irregular watering can lead to dry conditions in plants. Before watering your plant, check the soil with your finger, and if 2 to 3 inces of soil surface is dry, it is the time to water it. Morning hours are good for watering plants. Also, using a complete garden fertilizer at the recommended rate in spring and mulching in late winter/early spring and again in min-summer can help plants grow in dry situations.'
            },
            'Powdery Mildew': {
                'desclaimer': 'Remember, these are general solutions and the effectiveness can vary depending on the specific conditions of the plant and the environment. Always consider consulting with a local agricultural extension service or a plant health expert for more personalized advice. 😊',
                'description': 'It is a fungal disease that develops on the plant leaf as a white or gray powdery substance. It is observed that leaves suffering from powder mildew are often twisted or curled.',
                'solution': 'Powdery mildew can be treated can be treated early on with fungicides including potassium bicarbonate, neem oil, sulfur, or copper. Home remedies like baking soda and milk can also be successful treatments when applied properly.'
            },
            'Sooty Mold': {
                'desclaimer': 'Remember, these are general solutions and the effectiveness can vary depending on the specific conditions of the plant and the environment. Always consider consulting with a local agricultural extension service or a plant health expert for more personalized advice. 😊',
                'description': 'It is also a fungal disease that develops on the surface of the leaf that is black in color. The affected leaf growth gets stunted.',
                'solution': 'The best method to remove the mold is to soak affected plants in a water and detergent mixture. Use 1 tablespoon of household liquid detergent per gallon of water and spray it on the plants. Wait 15 minutes, then wash the detergent solution off with a strong stream of water. You may have to repeat this treatment a number of times over a few weeks.'
            },
            'Healthy': {
                'desclaimer': 'No desclaimer',
                'description': 'It shows that the plant is in a healthy state. In this condition, the leaf appears smooth, green, and evenly shaped.',
                'solution': 'Already healthy. No solution required.'
            }
        }

    def predictionmask(self):
        try:
            imagename = self.filename
            # Load the image
            image = Image.open(imagename)
            # Apply the transformation
            input_tensor = self.transform(image)
            # Add a batch dimension
            input_tensor = input_tensor.unsqueeze(0)
            # Move the input to the appropriate device
            input_tensor = input_tensor.to(self.device)
            # Perform the prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            # Get the predicted class
            _, predicted_class = torch.max(output, 1)
            # Map predicted class index to label
            labels = ['Bacterial Spot', 'Brown Blight', 'Dry', 'Healthy', 'Powdery Mildew', 'Sooty Mold']
            prediction = labels[predicted_class.item()]
            details = self.plant_details[prediction]
            return [{"image": prediction, "details": details}]
            return details
        except Exception as ex:
            raise ex
