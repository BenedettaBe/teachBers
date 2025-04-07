import openai
import joblib
import numpy as np
import cv2
from PIL import Image
import base64
import matplotlib.pyplot as plt
import os

def process_images(img1_path, img2_path, output_path, show_plots=True):
    """Colora le aree nere di due immagini e le sovrappone con il rosso davanti al verde"""
    # Carica le immagini in scala di grigi
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Errore: impossibile caricare le immagini. Verifica i percorsi:")
        print(f"Immagine 1: {img1_path}")
        print(f"Immagine 2: {img2_path}")
        return False
    
    # Assicura che abbiano le stesse dimensioni
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Crea immagini RGBA vuote
    height, width = img1.shape
    green_img = np.zeros((height, width, 4), dtype=np.float32)
    red_img = np.zeros((height, width, 4), dtype=np.float32)
    
    # Soglia per identificare le aree nere
    threshold = 180
    
    # Colora di verde le aree nere della prima immagine
    black_areas1 = img1 < threshold
    green_img[black_areas1, 1] = 1.0  # Verde
    green_img[black_areas1, 3] = 1.0 - (img1[black_areas1].astype(np.float32) / threshold)  # Alpha
    
    # Colora di rosso le aree nere della seconda immagine
    black_areas2 = img2 < threshold
    red_img[black_areas2, 0] = 1.0  # Rosso
    red_img[black_areas2, 3] = 1.0 - (img2[black_areas2].astype(np.float32) / threshold)  # Alpha
    
    # Sovrapponi le immagini con il rosso davanti al verde
    result = np.zeros_like(red_img)
    
    # Alpha blending (rosso davanti al verde)
    for i in range(3):  # Canali R, G, B
        result[..., i] = (red_img[..., i] * red_img[..., 3] + 
                         green_img[..., i] * green_img[..., 3] * (1 - red_img[..., 3]))
    
    # Calcola il nuovo canale alpha
    result[..., 3] = red_img[..., 3] + green_img[..., 3] * (1 - red_img[..., 3])
    
    # Normalizza dove alpha > 0
    alpha_positive = result[..., 3] > 0
    for i in range(3):  # Canali R, G, B
        result[alpha_positive, i] /= result[alpha_positive, 3]
    
    # Salva il risultato
    plt.imsave(output_path, np.clip(result, 0, 1))
    print(f"Risultato salvato in {output_path}")
    
    # Visualizza i risultati (opzionale)
    if show_plots:
        plt.figure(figsize=(12, 4))
        '''plt.subplot(1, 4, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Immagine 1')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(img2, cmap='gray')
        plt.title('Immagine 2')
        plt.axis('off')'''
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(green_img, 0, 1))
        plt.title('Aree verdi')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(red_img, 0, 1))
        plt.title('Aree rosse')
        plt.axis('off')
        
        plt.figure()
        plt.imshow(np.clip(result, 0, 1))
        plt.axis('off')
        plt.show(block=False)  # Non blocca l'esecuzione
        
        # Attendi un input dell'utente prima di chiudere
        input("Premi Enter per chiudere le finestre e terminare il programma...")
        plt.close('all')
    
    return True

def load_image(image_path):
    """Loads an image and converts it into a compatible format."""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def image_to_base64(image_path):
    """Converts an image to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = "lmstudio-community/gemma-2-2b-it-GGUF"
    
    def provide_feedback(self, image1_path, image2_path, number, number_predicted):
        """Passes two images to LM Studio to receive improvement suggestions."""
        image1_base64 = image_to_base64(image1_path)
        image2_base64 = image_to_base64(image2_path)
        
        if image1_base64 is None or image2_base64 is None:
            return "Error loading images."

        messages = [
            {"role": "system", "content": "You are an assistant helping 6-year-old children improve their handwriting for italian numbers from 0 to 9."},
            {"role": "user", "content": f"This is the number {image1_base64} written by a child, and this is the standard number {image2_base64} that you have to compare with the one written by the child. Provide clear and simple suggestions to improve the number written. Give me the differences between the number written by the child and the standard number {image2_base64}."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in feedback request: {e}")
            return "Unable to get feedback at this time."

def predict_digit(image_path, model):
    """Predicts the digit from an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (28, 28))
    img = 255 - img
    img[img < 100] = 0
    img[img >= 100] = 255
    print(img)
    img = img.flatten()
    img = img.reshape(1, -1)  # Ensure proper shape for the model
    
    return model.predict(img)[0]

def process_frame(frame):
    # Save the captured frame
    img_name = "lavoroFine/images/image.png"
    cv2.imwrite(img_name, frame)
    print(f"{img_name} written!")
    
    # Process the image
    image = cv2.imread(img_name)
    if image is None:
        print(f"Error: the image was not loaded. Check the path: {img_name}")
        return None
    
    # Convert to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return None
    
    # Process the largest contour
    contour = max(contours, key=cv2.contourArea)
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Center the crop to get a square
    if w > h:
        y -= (w - h) // 2
        h = w
    else:
        x -= (h - w) // 2
        w = h
    
    # Ensure coordinates are valid
    x, y = max(0, x), max(0, y)
    h = min(h, image.shape[0] - y)
    w = min(w, image.shape[1] - x)
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]
    
    # Prepare final 28x28 image
    # Convert and resize to 20x20
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((20, 20))
    
    # Create 28x28 image with padding
    final_array = np.ones((28, 28), dtype=np.uint8) * 255
    gray_image = np.array(pil_image.convert('L'))
    
    # Insert the 20x20 image in the center of the 28x28 canvas
    final_array[4:-4, 4:-4] = gray_image
    
    final_image = Image.fromarray(final_array)
    
    # Save and show the final image
    cropped_img_name = 'lavoroFine/images/cropped_img.png'
    final_image.save(cropped_img_name)
    print(f"Cropped image saved as {cropped_img_name}")
    final_image.show()
    
    return cropped_img_name


def main():
    # Print current working directory for debugging
    print("Current working directory:", os.getcwd())
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    captured = False
    processed_image_path = None
    
    while not captured:
        # Capture frame from webcam
        ret, frame = cam.read()
        if not ret:
            print("Unable to capture frame")
            break
            
        # Show real-time video
        cv2.imshow("test", frame)
        
        # Handle keyboard input
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Escape pressed, closing...")
            break
        elif k % 256 == 32:  # SPACE
            processed_image_path = process_frame(frame)
            if processed_image_path:
                captured = True
    
    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    
    if not processed_image_path:
        print("No image was processed. Exiting.")
        return
    
    input_image_path = processed_image_path
    
    # Define all possible model filenames to try
    model_filenames = ['lavoroFine/modello/model_cifre.joblib']
    model = None
    
    # Try to load the model from any of the possible filenames
    for filename in model_filenames:
        try:
            # Try with relative path
            model_path = os.path.join('.', filename)
            print(f"Trying to load model from: {model_path}")
            model = joblib.load(model_path)
            print(f"Successfully loaded model from {model_path}")
            break
        except FileNotFoundError:
            try:
                # Try with absolute path in the same directory as the script
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
                print(f"Trying to load model from: {model_path}")
                model = joblib.load(model_path)
                print(f"Successfully loaded model from {model_path}")
                break
            except FileNotFoundError:
                print(f"Could not find model file: {filename}")
                continue
    
    # If no model could be loaded, ask the user for the correct path
    if model is None:
        print("Could not find any model file automatically.")
        while model is None:
            custom_path = input("Please enter the full path to your model file (or type 'exit' to quit): ")
            if custom_path.lower() == 'exit':
                return
            try:
                model = joblib.load(custom_path)
                print(f"Successfully loaded model from {custom_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    # Make a prediction
    predicted_digit = predict_digit(input_image_path, model)
    print(f"I recognized a {predicted_digit}")
    
    response = input("Is it correct? (y/n) ")
    digit = None
    
    if response.lower() == "y":
        digit = predicted_digit
    else:
        while True:
            digit_input = input("Which number did you draw? (0-9) ")
            if digit_input.isdigit() and 0 <= int(digit_input) <= 9:
                digit = int(digit_input)
                break
            else:
                print("Please enter a valid digit between 0 and 9.")
    
    # Check if reference image exists
    reference_image_path = f"lavoroFine/standard/{digit}.png"
    if not os.path.exists(reference_image_path):
        print(f"Warning: Reference image for digit {digit} not found at {reference_image_path}")
        reference_image_dir = input("Please enter the directory containing reference digit images: ")
        reference_image_path = os.path.join(reference_image_dir, f"{digit}.png")
        if not os.path.exists(reference_image_path):
            print(f"Error: Reference image still not found at {reference_image_path}")
            return

    output_path = f"lavoroFine/risultati/result{digit}.png"
    success = process_images(reference_image_path, input_image_path, output_path, show_plots=True)
    
    if success:
        print(f"Elaborazione completata con successo. Risultato salvato in {output_path}")
    else:
        print("Elaborazione fallita.")
    
    # Instance of the LM Studio client and initialize it
    lm_client = LMStudioClient()
    
    # Request improvement suggestions
    feedback = lm_client.provide_feedback(input_image_path, reference_image_path, digit, predicted_digit)
    print("Suggestions to improve handwriting:")
    print(feedback)

if __name__ == "__main__":
    main()