from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob
import time


def _add_text_to_image(
    input_path, output_path, text, font_family, font_size, color, location
):
    """Adds text to an image at a specified location and saves the annotated image.
    This function opens an image file, draws the specified text on it using the provided
    font settings, and saves the resulting image to the specified output path.
        input_path (str): Path to the input image file.
        text (str): The text to be added to the image.
        font_family (str): Path to the TrueType font file to use for the text.
        font_size (int): Size of the font to be used for the text.
        color (tuple): Color of the text in RGB format, e.g., (255, 255, 255) for white.
        location (tuple): Coordinates (x, y) specifying where to place the text on the image.
    Raises:
        IOError: If the input image file cannot be opened or the font file is not found.
    """
    # Open image and create a drawing context.
    image = Image.open(input_path)
    draw = ImageDraw.Draw(image)

    # Try to load the specified TrueType font.
    try:
        font = ImageFont.truetype(font_family, font_size)
    except IOError:
        # Fallback if the font file is not found.
        font = ImageFont.load_default(font_size=font_size)

    # Draw the text at the given location.
    draw.text(location, text, font=font, fill=color)
    image.save(output_path)


def _annotate_img(input_path, output_path, text):
    """Annotates the image at the given path with the given text.
    Text will be added in one of the 4 corners, depending on the available amount of white space in each corner.
    Text is added in red color with font family 'Arial' and size 120.
    If 'arial.ttf' is not available, a fallback font is chosen.

    Args:
        input_path (str): Path to the image file.
        output_path (str): Path to save the annotated image.
        text (str): Text to label the image.
    """
    # Open the image and ensure it's in RGB mode.
    image = Image.open(input_path).convert("RGB")
    width, height = image.size

    # Set font properties.
    font_size = 120
    font_family = "arial.ttf"

    # Create a drawing context.
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_family, font_size)
    except IOError:
        # Fallback if the font file is not found.
        font = ImageFont.load_default(font_size=font_size)

    # Measure the text bounding box.
    # textbbox returns (x0, y0, x1, y1) relative to the anchor (0,0)
    bbox = draw.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = bbox
    text_width = x1 - x0
    text_height = y1 - y0

    # Define candidate regions (bounding boxes) for each corner.
    # Each box is defined as (left, top, right, bottom).
    candidates = {
        "top_left": (0, 0, text_width, text_height),
        "top_right": (width - text_width, 0, width, text_height),
        "bottom_left": (0, height - text_height, text_width, height),
        "bottom_right": (width - text_width, height - text_height, width, height),
    }

    # Count white pixels in each candidate area.
    white_counts = {}
    for corner, box in candidates.items():
        region = image.crop(box).convert("RGB")
        region_array = np.array(region)
        # Count pixels that are exactly white (255, 255, 255)
        count_white = np.sum(np.all(region_array == [255, 255, 255], axis=-1))
        white_counts[corner] = count_white

    # Determine which corner has the most white pixels.
    max_white = max(white_counts.values())
    max_candidates = [
        corner for corner, count in white_counts.items() if count == max_white
    ]

    if len(max_candidates) == 1:
        chosen_corner = max_candidates[0]
    else:
        # If there is no unique maximum, default to the bottom right.
        chosen_corner = "bottom_right"

    # Compute the adjusted drawing location based on the bounding box offsets.
    # This ensures that the drawn text's bounding box is flush with the chosen image corner
    if chosen_corner == "top_left":
        location = (-x0, -y0)
    elif chosen_corner == "top_right":
        location = (width - x1, -y0)
    elif chosen_corner == "bottom_left":
        location = (-x0, height - y1)
    else:  # bottom_right
        location = (width - x1, height - y1)

    # Define text color (red) as an RGB tuple.
    color = (255, 0, 0)

    # Call the helper function to add text to the image.
    _add_text_to_image(
        input_path, output_path, text, font_family, font_size, color, location
    )

def annotate_images(input_dir, output_dir):
    """Annotates all PNG images in the input directory with their file name (without file extension).
    The annotated images are saved in the output directory.
    The output directory will be created if non-existent.

    Args:
        input_dir (str): Path to the directory containing input PNG images.
        output_dir (str): Path to the directory to save annotated images.
    """
    # Create the output directory 
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all PNG files in the input directory.
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    for input_path in png_files:
        # Extract the base filename (without extension) to use as the annotation text.
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        
        # Annotate the image using the annotate_img function. Try 3 times as the image may be in use for a short time.
        for attempt in range(3):
            try:
                _annotate_img(input_path, output_path, base_name)
                break
            except Exception as ex:
                if attempt < 2:  # Retry for the first 2 attempts
                    time.sleep(0.3)
                else:
                    raise ex  # Raise the exception after 3 failed attempts


