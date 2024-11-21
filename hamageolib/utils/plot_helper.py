# MIT License
# 
# Copyright (c) 2024 Haoyuan Li
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
File: plot_helpers.py
Author: Haoyuan Li

Description:
This module provides utility functions and classes for generating plots 
related to geodynamic modeling. It is intended to support plot customization 
and visualization tasks, with a focus on ease of use and consistency.

Classes:
    (To be defined)
        - Descriptions will be added once the classes are implemented.

Functions:
    scale_matplotlib_params
        - Scales Matplotlib parameters proportionally based on a given scaling factor. 
        This function adjusts font sizes, line widths, marker sizes, and other visual 
        elements to ensure consistent scaling in plots. It allows fine-tuning of specific 
        parameters such as font size, line width, and legend font size through additional multipliers.
"""

import matplotlib.pyplot as plt
import fitz
import os
from PIL import Image  # Pillow for cropping and saving images
import subprocess

def scale_matplotlib_params(scaling_factor=1.0, **kwargs):
    """
    Scale Matplotlib parameters proportionally by a given scaling factor.
    
    Parameters:
        scaling_factor (float): The factor by which to scale font sizes, linewidths, and markers.
        **kwargs:
            font_scaling_multiplier (float): extra scaling multiplier for font
    """
    font_scaling_multiplier = kwargs.get("font_scaling_multiplier", 1.0)
    line_width_scaling_multiplier = kwargs.get("line_width_scaling_multiplier", 1.0)
    legend_font_scaling_multiplier = kwargs.get("legend_font_scaling_multiplier", 1.0)

    # Base font sizes and line widths
    base_fontsize = 10 * font_scaling_multiplier
    base_linewidth = 1.0
    base_markersize = 6.0
    base_tickwidth = 0.5
    base_ticklength = 3.5
    base_spinewidth = 0.5
    base_gridlinewidth = 0.5
    
    # Update rcParams
    plt.rcParams.update({
        "figure.figsize": (6 * scaling_factor, 4 * scaling_factor),  # Scale figure size
        "font.size": base_fontsize * scaling_factor,                # Scale font size
        "axes.labelsize": base_fontsize * scaling_factor,           # Scale axis label size
        "axes.titlesize": base_fontsize * scaling_factor,           # Scale title size
        "xtick.labelsize": base_fontsize * scaling_factor,          # Scale x-tick label size
        "ytick.labelsize": base_fontsize * scaling_factor,          # Scale y-tick label size
        "legend.fontsize": base_fontsize * scaling_factor * legend_font_scaling_multiplier,          # Scale legend font size
        "lines.linewidth": base_linewidth * scaling_factor * line_width_scaling_multiplier,         # Scale line width
        "lines.markersize": base_markersize * scaling_factor * line_width_scaling_multiplier,       # Scale marker size
        "xtick.major.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale major tick width
        "ytick.major.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale major tick width
        "xtick.minor.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale minor tick width
        "ytick.minor.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale minor tick width
        "xtick.major.size": base_ticklength * scaling_factor,       # Scale major tick length
        "ytick.major.size": base_ticklength * scaling_factor,       # Scale major tick length
        "grid.linewidth": base_gridlinewidth * scaling_factor      # Scale grid line width
    })


def convert_eps_to_pdf(eps_file, pdf_file):
    """
    Convert an EPS file to a PDF using Ghostscript via subprocess.
    
    Args:
        eps_file (str): Path to the EPS file.
        pdf_file (str): Path to the output PDF file.
    """
    try:
        # Ghostscript command
        command = [
            "gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={pdf_file}",
            eps_file
        ]

        # Run the command
        subprocess.run(command, check=True)
        print(f"Successfully converted {eps_file} to {pdf_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error during Ghostscript execution: {e}")
    except FileNotFoundError:
        print("Ghostscript (gs) command not found. Make sure it is installed.")


def extract_image_by_size(
    pdf_file, target_size, output_dir, crop_box=None
):
    """
    Extract an embedded image by its size from a PDF and optionally crop it.
    
    Args:
        pdf_file (str): Path to the PDF file.
        target_size (tuple): (width, height) of the target image.
        output_dir (str): Directory to save the output images.
        crop_box (tuple, optional): (x0, y0, x1, y1) crop coordinates. Defaults to None.

    Returns:
        str: Path to the saved cropped image, or None if no image is found.
    """
    # Open the PDF
    doc = fitz.open(pdf_file)
    page = doc[0]
    
    # Get embedded images
    images = page.get_images(full=True)
    if not images:
        print("No images found on the page.")
        return None

    print(f"Found {len(images)} image(s) on the page.")
    for img in images:
        xref = img[0]  # Image reference number
        pix = fitz.Pixmap(doc, xref)
        
        # Match the target size
        if (pix.width, pix.height) == target_size:
            print(f"Found matching image with dimensions: {pix.width}x{pix.height}")
            
            # Save the full image
            full_image_path = os.path.join(output_dir, "real_figure.png")
            pix.save(full_image_path)
            print(f"Saved the full figure as {full_image_path}")
            
            return full_image_path

    print("No matching image found.")
    return None


def overlay_images_on_blank_canvas(
    canvas_size,
    image_files,
    image_positions,
    cropping_regions=None,
    image_scale_factors=None,
    output_image_file="output.png"
):
    """
    Create a blank canvas, crop and overlay multiple images, then save as an image.
    
    Args:
        canvas_size (tuple): (width, height) of the blank canvas in pixels.
        image_files (list of str): List of paths to images to overlay on the canvas.
        image_positions (list of tuple): List of (x, y) positions for placing each image.
        cropping_regions (list of tuple, optional): List of cropping regions (left, upper, right, lower)
            for each image. If None, no cropping is applied.
        image_scale_factors (list of float, optional): List of scale factors to resize each image.
            If None, no scaling is applied.
        output_image_file (str): Path to save the final combined image.
    """
    # Validate inputs
    if len(image_files) != len(image_positions):
        raise ValueError("The number of image files must match the number of image positions.")
    if cropping_regions and len(cropping_regions) != len(image_files):
        raise ValueError("The number of cropping regions must match the number of image files.")
    if image_scale_factors and len(image_scale_factors) != len(image_files):
        raise ValueError("The number of scale factors must match the number of image files.")

    # Create a blank canvas
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))  # White background

    # Process each image
    for idx, image_file in enumerate(image_files):
        # Load the image
        image = Image.open(image_file).convert("RGBA")
        print(f"Original size of image {idx + 1}: {image.size}")

        # Apply cropping if specified
        if cropping_regions:
            crop_region = cropping_regions[idx]
            image = image.crop(crop_region)
            print(f"Cropped size of image {idx + 1}: {image.size}")

        # Apply scaling if specified
        if image_scale_factors:
            scale_factor = image_scale_factors[idx]
            if scale_factor:
                original_width, original_height = image.size
                new_width = int(round(original_width * scale_factor))
                new_height = int(round(original_height * scale_factor))
                image = image.resize((new_width, new_height), Image.ANTIALIAS)
                print(f"Scaled size of image {idx + 1}: {image.size}")

        # Get the position for this image
        position = image_positions[idx]

        # Overlay the image onto the canvas
        canvas.paste(image, position, image)

    # Save the final canvas as an image
    canvas.save(output_image_file, "PNG")
    print(f"Overlay completed. Final image saved as {output_image_file}")