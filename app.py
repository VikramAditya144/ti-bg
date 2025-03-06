import streamlit as st
import torch
from PIL import Image
import requests
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
import io
import cloudinary
import cloudinary.uploader
import cloudinary.api
import uuid
import datetime
import time

# Set device to CPU
device = torch.device('cpu')

# Configure Streamlit page
st.set_page_config(
    page_title="Background Removal Tool",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state for storing app state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'upload'  # 'upload' or 'history'
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None

# Initialize Cloudinary configuration
def initialize_cloudinary():
    # Cloudinary configuration with your credentials
    cloudinary.config(
        cloud_name="dmeeik2ix",
        api_key="735119433879568",
        api_secret="gU26ckqZurDurG8V9qLtUXAtOug",
        secure=True
    )

# Model loading with caching
@st.cache_resource
def load_model():
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )
        model.to(device)  # Move model to CPU
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def transform_image():
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Resize image for display
def resize_for_display(image, max_size=(800, 800)):
    """Resize image while maintaining aspect ratio for better UI display"""
    # Get original dimensions
    width, height = image.size
    
    # Calculate the scaling factor
    scale = min(max_size[0] / width, max_size[1] / height)
    
    # Only resize if the image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

# Image processing function
def process_image(image, model):
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Transform image
        transformer = transform_image()
        input_tensor = transformer(image).unsqueeze(0).to(device)
        
        # Process with model
        with torch.no_grad():
            predictions = model(input_tensor)[-1].sigmoid().cpu()
        
        # Get mask
        mask = predictions[0].squeeze()
        mask_image = transforms.ToPILImage()(mask)
        mask_resized = mask_image.resize(original_size)
        
        # Apply mask to original image
        result = image.copy()
        result.putalpha(mask_resized)
        
        return result
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Function to load image from URL
def load_url_image(url):
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

# Function to upload image to Cloudinary
def upload_to_cloudinary(image, is_processed=False):
    try:
        # Convert PIL Image to bytes
        img_buffer = io.BytesIO()
        if is_processed:
            image.save(img_buffer, format='PNG')
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Create a unique ID for the image pair
        if not is_processed:
            image_id = str(uuid.uuid4())
            st.session_state.last_processed = image_id
        else:
            # Use the previously generated ID for the processed image
            image_id = st.session_state.last_processed
        
        folder = "processed" if is_processed else "original"
        
        # Upload to Cloudinary
        # Set transformation to resize large images
        transformation = None
        if not is_processed:
            transformation = {"width": 1200, "crop": "limit"}
        
        upload_result = cloudinary.uploader.upload(
            img_buffer,
            folder=f"background_removal/{folder}",
            public_id=image_id,
            tags=["background_removal", folder],
            resource_type="image",
            transformation=transformation
        )
        
        return upload_result.get('secure_url'), image_id
    except Exception as e:
        st.error(f"Error uploading to Cloudinary: {str(e)}")
        return None, None

# Function to fetch image history from Cloudinary
def get_image_history():
    try:
        # Use admin API to get all resources with background_removal tag
        processed_result = cloudinary.api.resources_by_tag(
            "processed",
            resource_type="image",
            max_results=100
        )
        
        processed_images = {}
        for resource in processed_result.get('resources', []):
            image_id = os.path.basename(resource['public_id'])
            processed_images[image_id] = {
                'processed_url': resource['secure_url'],
                'created_at': resource['created_at']
            }
        
        original_result = cloudinary.api.resources_by_tag(
            "original",
            resource_type="image",
            max_results=100
        )
        
        # Match original images with processed ones
        history = []
        for resource in original_result.get('resources', []):
            image_id = os.path.basename(resource['public_id'])
            if image_id in processed_images:
                history.append({
                    'id': image_id,
                    'original_url': resource['secure_url'],
                    'processed_url': processed_images[image_id]['processed_url'],
                    'created_at': processed_images[image_id]['created_at']
                })
        
        # Sort by creation date (newest first)
        history.sort(key=lambda x: x['created_at'], reverse=True)
        return history
    except Exception as e:
        st.error(f"Error fetching image history: {str(e)}")
        return []

# Switch view function
def switch_to_upload():
    st.session_state.current_view = 'upload'

def switch_to_history():
    st.session_state.current_view = 'history'

# Process and store image
def process_and_store(image, model):
    # Process image
    with st.spinner("Processing image..."):
        # Create a copy to avoid modifying the original
        img_copy = image.copy()
        
        processed_image = process_image(img_copy, model)
        
        if processed_image is not None:
            # Upload original image first
            original_url, image_id = upload_to_cloudinary(img_copy, is_processed=False)
            
            if original_url and image_id:
                # Now upload the processed image with the same ID
                processed_url, _ = upload_to_cloudinary(processed_image, is_processed=True)
                
                if processed_url:
                    # Add to history
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = {
                        'id': image_id,
                        'original_url': original_url,
                        'processed_url': processed_url,
                        'created_at': timestamp
                    }
                    st.session_state.history.insert(0, new_entry)  # Add to the beginning
                    
                    # Wait for Cloudinary processing
                    time.sleep(1)
                    
                    return processed_image, processed_url
                else:
                    st.error("Failed to upload processed image to Cloudinary")
                    return processed_image, None
            else:
                st.error("Failed to upload original image to Cloudinary")
                return processed_image, None
        else:
            st.error("Failed to process the image")
            return None, None

# Upload view
def show_upload_view(model):
    st.subheader("Upload a new image")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Image URL"],
        horizontal=True
    )
    
    image = None
    
    # Process image based on input method
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp', 'avif']
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
            
    else:  # URL input
        url = st.text_input("Enter the URL of an image")
        if url:
            image = load_url_image(url)
    
    # If we have an image, process it
    if image is not None:
        # Display a preview of the uploaded image (resized)
        display_image = resize_for_display(image)
        st.image(display_image, caption="Preview", use_container_width=True)
        
        # Create a button to process the image
        if st.button("Process Image", use_container_width=True):
            # Process and display results
            col1, col2 = st.columns(2)
            
            # Show original image
            with col1:
                st.subheader("Original Image")
                display_image = resize_for_display(image)
                st.image(display_image, use_container_width=True)
            
            # Process and show result
            with col2:
                st.subheader("Processed Image")
                processed_image, processed_url = process_and_store(image, model)
                
                if processed_image is not None:
                    # Resize for display
                    display_processed = resize_for_display(processed_image)
                    st.image(display_processed, use_container_width=True)
                    
                    # Save and provide download button
                    img_buffer = io.BytesIO()
                    processed_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="Download processed image",
                        data=img_buffer.getvalue(),
                        file_name="processed_image.png",
                        mime="image/png"
                    )
                    
                    if processed_url:
                        st.success("Image processed and stored successfully!")
                        st.markdown(f"[View on Cloudinary]({processed_url})")
                        
                        # Add a button to view history after processing
                        if st.button("View History"):
                            switch_to_history()
                            st.rerun()

# History view
def show_history_view():
    st.subheader("Previously Processed Images")
    
    if st.button("Refresh History"):
        st.experimental_rerun()
    
    # Get history from Cloudinary
    with st.spinner("Loading image history..."):
        history = get_image_history()
    
    if not history:
        st.info("No processed images found in history.")
        return
    
    # Display history
    for entry in history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(entry['original_url'], use_container_width=True)
            st.caption(f"Created: {entry['created_at']}")
        
        with col2:
            st.subheader("Processed Result")
            st.image(entry['processed_url'], use_container_width=True)
            # Add a download link
            st.markdown(f"[Download Image]({entry['processed_url']})")
        
        st.divider()

# Main application
def main():
    # Initialize Cloudinary
    initialize_cloudinary()
    
    st.title("🖼️ Background Removal Tool")
    st.write("Upload an image or provide a URL to remove the background")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Upload New Image", use_container_width=True):
            switch_to_upload()
    with col2:
        if st.button("🗃️ View History", use_container_width=True):
            switch_to_history()
    
    st.divider()
    
    # Load model for upload view
    model = None
    if st.session_state.current_view == 'upload':
        with st.spinner("Loading model... This may take a minute..."):
            model = load_model()
        
        if model is None:
            st.error("Failed to load the model. Please check the model configuration and try again.")
            return
    
    # Show the current view
    if st.session_state.current_view == 'upload':
        show_upload_view(model)
    else:
        show_history_view()

if __name__ == "__main__":
    main()



