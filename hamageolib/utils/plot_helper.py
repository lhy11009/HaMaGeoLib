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
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont  # Pillow for cropping and saving images
import subprocess

from utils.file_reader import read_simulation_log

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


def add_text_to_image(
    image_path: str,
    output_path: str,
    text: str,
    position: tuple,
    font_path: str,
    font_size: int = 60,
    text_color: str = "black",
):
    """
    Adds text to a PNG image and saves the result.

    Parameters:
        image_path (str): Path to the input PNG file.
        output_path (str): Path to save the output image with text.
        text (str): Text to insert onto the image.
        position (tuple): (x, y) coordinates for the text position.
        font_path (str): Path to the TrueType (.ttf) font file.
        font_size (int): Font size for the text. Default is 60.
        text_color (str): Color of the text. Default is "black".

    Returns:
        None
    """
    try:
        # Load the image
        image = Image.open(image_path)

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Define the font
        font = ImageFont.truetype(font_path, font_size)

        # Add text to the image
        draw.text(position, text, font=font, fill=text_color)

        # Save the edited image
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def combine_images(image_matrix, output_path):
    """
    Combines images into a single canvas based on a 2D NumPy matrix layout.

    Parameters:
        image_matrix (numpy.ndarray): A 2D NumPy array containing file paths to the PNG images or None values.
        output_path (str): Path to save the combined image.

    Returns:
        None
    """
    # Validate the matrix is 2D
    if not isinstance(image_matrix, np.ndarray) or len(image_matrix.shape) != 2:
        raise ValueError("The image matrix must be a 2D NumPy array.")

    # Validate all entries in the first row and first column are not None
    if any(path is None for path in image_matrix[0, :]):
        raise ValueError("All entries in the first row must be valid file paths (not None).")
    if any(path is None for path in image_matrix[:, 0]):
        raise ValueError("All entries in the first column must be valid file paths (not None).")

    # Validate each file exists if not None
    for path in image_matrix.flat:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load images, skipping None
    images = np.empty_like(image_matrix, dtype=object)
    for row_idx, row in enumerate(image_matrix):
        for col_idx, path in enumerate(row):
            images[row_idx, col_idx] = Image.open(path) if path is not None else None

    # Calculate the canvas size
    canvas_width = sum(img.width for img in images[0, :])  # Total width is the sum of the first row's widths
    canvas_height = sum(img.height for img in images[:, 0])  # Total height is the sum of the first column's heights

    # Create a blank canvas
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))  # Transparent canvas

    # Place images onto the canvas, skipping None
    y_offset = 0
    for row_idx in range(images.shape[0]):
        x_offset = 0
        for col_idx in range(images.shape[1]):
            img = images[row_idx, col_idx]
            if img is not None:
                canvas.paste(img, (x_offset, y_offset))
                x_offset += img.width
        if images[row_idx, 0] is not None:
            y_offset += images[row_idx, 0].height

    # Save the combined image
    canvas.save(output_path)
    print(f"Combined image saved to {output_path}")


def plot_statistic_generic(
    data, x_col, y_col, xlabel, ylabel, title, label, ax, color=None, **kwargs
):
    """
    Generic function to create a statistic plot with units included in labels and optional annotations.
    Supports plotting multiple lines on the same plot.

    Args:
        data (pd.DataFrame): DataFrame containing the simulation data.
        x_col (str): Column name for the x-axis.
        y_col (str or list): Column name(s) for the y-axis.
        xlabel (str): Label for the x-axis (before unit addition).
        ylabel (str): Label for the y-axis (before unit addition).
        title (str): Title of the plot.
        label (str or list): Legend label(s) for the data series.
        ax (matplotlib.axes.Axes): Axes object for the plot.
        color (str or list, optional): Color(s) for the plot line(s).
        **kwargs: Optional arguments for plot customization, including:
            - annotate_column (str): Column name to annotate on the plot.
            - annotate_points (int, optional): Number of points to annotate (default: 5).

    Returns:
        None

    Raises:
        ValueError: If x_col, y_col, label, and color lengths don't match (when y_col is a list).
        KeyError: If x_col or y_col is not found in the DataFrame.
    """
    # Ensure y_col, label, and color are lists for consistency
    if isinstance(y_col, str):
        y_col = [y_col]
    if isinstance(label, str):
        label = [label]
    if isinstance(color, str) or color is None:
        color = [color] * len(y_col)  # Repeat the same color or None for all lines

    # Error handling: Ensure y_col, label, and color lengths match
    if len(y_col) != len(label):
        raise ValueError(f"y_col and label must have the same length. Got {len(y_col)} and {len(label)}.")
    if color and len(color) != len(y_col):
        raise ValueError(f"y_col and color must have the same length if color is provided. Got {len(y_col)} and {len(color)}.")

    # Check that x_col exists in the DataFrame
    if x_col not in data.columns:
        raise KeyError(f"x_col '{x_col}' not found in the DataFrame. Available columns: {list(data.columns)}")

    # Check that all y_col values exist in the DataFrame
    missing_y_cols = [col for col in y_col if col not in data.columns]
    if missing_y_cols:
        raise KeyError(f"y_col(s) {missing_y_cols} not found in the DataFrame. Available columns: {list(data.columns)}")

    # Retrieve optional arguments
    annotate_column = kwargs.get("annotate_column", None)
    annotate_points = kwargs.get("annotate_points", 5)

    # Retrieve units from DataFrame metadata
    units_map = data.attrs.get("units", {})
    x_unit = units_map.get(x_col, None)
    xlabel = f"{xlabel} ({x_unit})" if x_unit else xlabel
    ylabel = f"{ylabel}"  # Do not append y_unit for multi-line plots (ambiguous)

    # Plot each line
    for y, lbl, clr in zip(y_col, label, color):
        y_unit = units_map.get(y, None)
        ylbl = f"{lbl} ({y_unit})" if y_unit else lbl
        ax.plot(data[x_col], data[y], label=ylbl, color=clr)

    # Set labels and titles
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()

    # Add optional annotations for the first y_col only
    if annotate_column:
        num_points = len(data)
        if num_points <= annotate_points:
            indices_to_annotate = range(num_points)
        else:
            indices_to_annotate = np.linspace(0, num_points - 1, annotate_points, dtype=int)
        for i in indices_to_annotate:
            row = data.iloc[i]
            ax.annotate(
                str(row[annotate_column]),
                (row[x_col], row[y_col[0]]),  # Annotate based on the first y_col
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
                color="blue",
            )


def generate_statistic_plots(file_path, output_dir="plots", **kwargs):
    """
    Generates statistical plots from the simulation log file.

    Args:
        file_path (str): Path to the simulation log file.
        output_dir (str): Directory to save the generated plots.
        **kwargs: Optional arguments for plot customization passed to plot_statistic_generic.

    Returns:
        None
    """
    # Read the data
    data = read_simulation_log(file_path)

    # Ensure the output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Updated Plot configurations with cleaned column names
    plot_configs = [
        {
            "x_col": "Time",
            "y_col": "Time step number",
            "xlabel": "Time",
            "ylabel": "Time Step Number",
            "title": "Time Step Number Over Time",
            "label": "Time Step Number",
            "color": "blue",
            "file_name": "time_step_number_over_time.png",
        },
        {
            "x_col": "Time",
            "y_col": [
                "Number of Stokes degrees of freedom",
                "Number of temperature degrees of freedom",
                "Number of degrees of freedom for all compositions",
            ],
            "xlabel": "Time",
            "ylabel": "Number of Degrees of Freedom",
            "title": "Degrees of Freedom Over Time",
            "label": [
                "Stokes Degrees of Freedom",
                "Temperature Degrees of Freedom",
                "Compositional Degrees of Freedom",
            ],
            "color": ["blue", "orange", "green"],
            "file_name": "degrees_of_freedom_over_time.png",
        },
        {
            "x_col": "Time",
            "y_col": "Number of nonlinear iterations",
            "xlabel": "Time",
            "ylabel": "Nonlinear Iterations",
            "title": "Nonlinear Iterations Over Time",
            "label": "Nonlinear Iterations",
            "color": "red",
            "file_name": "nonlinear_iterations_over_time.png",
        },
        {
            "x_col": "Time",
            "y_col": [
                "Minimal temperature",
                "Average temperature",
                "Maximal temperature",
                "Average nondimensional temperature",
            ],
            "xlabel": "Time",
            "ylabel": "Temperature",
            "title": "Temperature Metrics Over Time",
            "label": [
                "Minimal Temperature",
                "Average Temperature",
                "Maximal Temperature",
                "Average Nondimensional Temperature",
            ],
            "color": ["blue", "orange", "green", "red"],
            "file_name": "temperature_metrics_over_time.png",
        },
        {
            "x_col": "Time",
            "y_col": ["RMS velocity", "Max. velocity"],
            "xlabel": "Time",
            "ylabel": "Velocity",
            "title": "RMS and Max Velocity Over Time",
            "label": ["RMS Velocity", "Max Velocity"],
            "color": ["purple", "brown"],
            "file_name": "velocity_comparison_over_time.png",
        },
    ]

    # Generate and save plots
    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_statistic_generic(
            data=data,
            x_col=config["x_col"],
            y_col=config["y_col"],
            xlabel=config["xlabel"],
            ylabel=config["ylabel"],
            title=config["title"],
            label=config["label"],
            ax=ax,
            color=config.get("color", None),
            **kwargs,
        )
        fig.savefig(os.path.join(output_dir, config["file_name"]))
        plt.close(fig)

    print(f"Statistic plots have been saved to: {output_dir}")
