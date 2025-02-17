from flask import Flask, request, render_template
import cv2
import numpy as np
import pytesseract
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
import base64
import time
import os
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)

# Load environment variables
load_dotenv()
FIGMA_API_TOKEN = os.getenv("FIGMA_API_TOKEN")

# Load LPIPS Model for Visual Fidelity
lpips_model = lpips.LPIPS(net='alex')

# Function to retrieve a Figma design screenshot
def get_figma_screenshot(figma_file_key, figma_node_id=None):
    headers = {"X-Figma-Token": FIGMA_API_TOKEN}

    # Determine Node ID if not provided
    if not figma_node_id:
        file_url = f"https://api.figma.com/v1/files/{figma_file_key}"
        response = requests.get(file_url, headers=headers)

        if response.status_code != 200:
            print("❌ Error retrieving Figma file details:", response.status_code, response.text)
            return None

        try:
            file_data = response.json()
            first_page = file_data["document"]["children"][0]
            figma_node_id = first_page["children"][0]["id"]
            print(f"✅ Auto-detected first frame: {figma_node_id}")
        except (KeyError, IndexError):
            print("❌ No valid frame found in Figma file.")
            return None

    # Request Image from Figma API
    image_url = f"https://api.figma.com/v1/images/{figma_file_key}?ids={figma_node_id}&format=png"
    image_response = requests.get(image_url, headers=headers)

    if image_response.status_code != 200:
        print("❌ Failed to retrieve Figma image:", image_response.text)
        return None

    image_data = image_response.json()
    image_link = image_data["images"].get(figma_node_id, None)

    if image_link:
        img_data = requests.get(image_link)
        if img_data.status_code == 200:
            with open("figma_screenshot.png", "wb") as f:
                f.write(img_data.content)
            print("✅ Figma screenshot saved successfully.")
            return "figma_screenshot.png"

    print("❌ No image URL returned from Figma API.")
    return None

# Function to capture a website screenshot
def capture_web_screenshot(web_url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280x800")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(web_url)
        time.sleep(10)  # Allow full page load
        web_screenshot_path = "web_screenshot.png"
        driver.save_screenshot(web_screenshot_path)
        print("✅ Web screenshot captured successfully.")
    finally:
        driver.quit()
    
    return web_screenshot_path

# Compute SSIM for Layout Consistency
def compute_ssim(image1, image2):
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# Extract text for text accuracy validation
def extract_text(image):
    return pytesseract.image_to_string(image)

def compare_text(text1, text2):
    return 1 - (len(set(text1.split()) ^ set(text2.split())) / max(len(text1.split()), len(text2.split())))

# Extract font details
def extract_text_and_font(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return [(data['text'][i], data['height'][i]) for i in range(len(data['text']))]

def compare_typography(image1, image2):
    fonts1 = extract_text_and_font(image1)
    fonts2 = extract_text_and_font(image2)

    matches = sum(1 if t1 == t2 and f1 == f2 else 0.5 for (t1, f1), (t2, f2) in zip(fonts1, fonts2))
    return matches / len(fonts1) if fonts1 else 0

# Compute Color Consistency
def compute_color_similarity(image1, image2):
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Compute LPIPS for Visual Fidelity
def compute_lpips(image1, image2):
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))
    image1 = torch.tensor(image1).permute(2, 0, 1).unsqueeze(0).float() / 255
    image2 = torch.tensor(image2).permute(2, 0, 1).unsqueeze(0).float() / 255
    return 1 - lpips_model(image1, image2).item()


# Function to get description based on similarity percentage
def get_description(category, percentage):
    if category == "layout_consistency":
        if 0 <= percentage <= 45:
            return "Low layout consistency. Significant differences detected."
        elif 46 <= percentage <= 84:
            return "Moderate layout consistency. Some differences detected."
        elif 85 <= percentage <= 100:
            return "High layout consistency. Minimal differences detected."
    elif category == "text_accuracy":
        if 0 <= percentage <= 45:
            return "Low text accuracy. Significant text mismatches detected."
        elif 46 <= percentage <= 84:
            return "Moderate text accuracy. Some text mismatches detected."
        elif 85 <= percentage <= 100:
            return "High text accuracy. Minimal text mismatches detected."
    elif category == "typography_consistency":
        if 0 <= percentage <= 15:
            return "Low typography consistency. Significant differences detected."
        elif 16 <= percentage <= 49:
            return "Moderate typography consistency. Some differences detected."
        elif 50 <= percentage <= 100:
            return "High typography consistency. Minimal differences detected."
    elif category == "color_consistency":
        if 0 <= percentage <= 45:
            return "Low color consistency. Significant color deviation detected."
        elif 46 <= percentage <= 84:
            return "Moderate color consistency. Some color deviation detected."
        elif 85 <= percentage <= 100:
            return "High color consistency. Minimal color deviation detected."
    elif category == "visual_fidelity":
        if 0 <= percentage <= 20:
            return "Low visual fidelity. Significant visual differences detected."
        elif 21 <= percentage <= 54:
            return "Moderate visual fidelity. Some visual differences detected."
        elif 55 <= percentage <= 100:
            return "High visual fidelity. Minimal visual differences detected."
    return "No description available."


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        figma_file_key = request.form['figma_file_key']
        web_url = request.form['web_url']
        figma_node_id = request.form.get('figma_node_id')

        figma_screenshot = get_figma_screenshot(figma_file_key, figma_node_id)
        web_screenshot = capture_web_screenshot(web_url)

        if not figma_screenshot or not web_screenshot:
            return "Error retrieving screenshots", 500

        img1 = cv2.imread(figma_screenshot)
        img2 = cv2.imread(web_screenshot)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute similarity scores
        layout_percentage = round(compute_ssim(img1, img2) * 100, 2)
        text_percentage = round(compare_text(extract_text(img1), extract_text(img2)) * 100, 2)
        typography_percentage = round(compare_typography(img1, img2) * 100, 2)
        color_percentage = round(compute_color_similarity(img1, img2) * 100, 2)
        visual_fidelity_percentage = round((1 - compute_lpips(img1, img2)) * 100, 2)

        # Initialize comments
        comments = []
        # Validate categories based on the similarity percentage  
        if layout_percentage < 85:  
            comments.append("Review spacing, alignment, and overall structure for consistency.")  
        if text_percentage < 85:  
            comments.append("Check for missing, incorrect, or misplaced text elements.")  
        if typography_percentage < 50:  
            comments.append("Ensure font type, size, weight, and spacing are consistent.")  
        if color_percentage < 85:  
            comments.append("Verify color accuracy, contrast, and consistency with the design.")  
        if visual_fidelity_percentage < 55:  
            comments.append("Check images, icons, and other graphical elements for accuracy.") 

        validation_status = "✅ Valid" if not comments else "❌ Not Valid"

        results = {
            "layout_consistency": layout_percentage,
            "text_accuracy": text_percentage,
            "typography_consistency": typography_percentage,
            "color_consistency": color_percentage,
            "visual_fidelity": visual_fidelity_percentage,
            "validation_status": validation_status,
            "Comments": comments
        }

        # Add descriptions to results
        results["Layout Description"] = get_description("layout_consistency", layout_percentage)
        results["Text Description"] = get_description("text_accuracy", text_percentage)
        results["Typography Description"] = get_description("typography_consistency", typography_percentage)
        results["Color Description"] = get_description("color_consistency", color_percentage)
        results["Visual Fidelity Description"] = get_description("visual_fidelity", visual_fidelity_percentage)

        # Convert images to displayable format
        _, buffer1 = cv2.imencode('.png', img1)
        figma_image_url = "data:image/png;base64," + base64.b64encode(buffer1).decode('utf-8')
        
        _, buffer2 = cv2.imencode('.png', img2)
        web_image_url = "data:image/png;base64," + base64.b64encode(buffer2).decode('utf-8')

        # Pass the calculated similarity percentages to the HTML template
        return render_template('result.html', results=results, figma_image_url=figma_image_url, web_image_url=web_image_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
