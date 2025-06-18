# HaMaGeo: A Geodynamic Research Toolkit

**Authors:** Haoyuan Li (hylli@ucdavis.edu), Magali Billen (mibillen@ucdavis.edu) 
**Version:** 0.1.0  
**License:** MIT  

hamageolib is a Python package designed to provide tools and models for geodynamic research, with a focus on simulating mantle convection, plate tectonics, and related geodynamic processes. Developed by researchers for researchers, HaMaGeo offers essential utilities, modeling capabilities, and visualization tools for geodynamics.

---

## Features

- **Core Functionality (`hamageolib_core`)**: The core module provides foundational classes and functions for geodynamic modeling and simulations, such as mantle convection and plate tectonics, designed to support a wide range of research in geodynamics.
  
- **Utility Functions (`hamageolib_utils`)**: Utility tools include parameter parsing and file management features that facilitate data processing, loading, and saving for geodynamic models. For example, the `parse_parameters_to_dict` function converts parameter files to dictionaries for easier manipulation.
  
- **Visualization Tools (`hamageolib_viz`)**: This module will include visualization utilities for interpreting simulation results, creating insightful plots of model outputs, and visualizing geological structures, assisting in the analysis and presentation of research findings.
---

## Prerequisites

Anaconda is required to manage the Python environment for this package.

To install conda environment for hamageolib

   conda env create --file environment.yml

To activate in terminal

   conda activate hmgeolib

To activate in Jupyter notebooks, select "hmgeolib" as kernel

## Installation

### Anaconda version

1. **hamageolib is currently in development. To install the latest version from anaconda**:

```bash
conda install lhy11009::hamageolib
```

2. **After this, you’ll be able to import and use hamageolib**:

```python
import hamageolib
```

### Development Version

If you’d like to work with the development version, you can clone the GitHub repository and add it to your `sys.path` in a script. This allows you to access the latest updates directly from the repository.

1. **Clone the repository**:
   
```bash
git clone https://github.com/lhy11009/HaMaGeoLib
```

2. **Add to sys.path in your script**:

In your Python script, include the following lines to add the cloned directory to your path:

```python
import sys
sys.path.insert(0, "path/to/HaMaGeoLib")  # Replace with the actual path to hamageolib
```

3. **After this, you’ll be able to import and use hamageolib as usual**:

```python
import hamageolib
```
---

4. **Sync the big test folder from cloud service**:

The `big_tests` folder contains large test files that are **not synced to GitHub** due to their size. Instead, these files are stored in a cloud-based storage solution.
These typically include files that are case outputs.

Note for myself: use the lihaoyuan81@gmail.com space for this purpose.


## Contributing

We welcome contributions to **hamageolib**! If you’re interested in adding new features or improving existing functionality, please follow the workflow below:

1. **Implement Your Feature**:
   - Develop your Python script in the appropriate submodule (e.g., `core`, `utils`, or `visualization`).
   - Ensure your code is modular, well-documented, and follows the project’s coding standards.

2. **Create Tests**:
   - Add tests to validate your new feature. Place tests in the `tests` directory following a similar structure to the main package.
   - Use `pytest` to organize and run tests efficiently.
   - Ensure all tests pass before submitting your contribution.

3. **Document with Jupyter Notebooks**:
   - Create a Jupyter notebook under `notebooks/examples` to demonstrate how to use your new feature.
   - Create a Jupyter notebook under `notebooks/experiments` to demonstrate how to work on some developing features.
   - Create a Jupyter notebook under `notebooks/tutorials` to demonstrate how to follow a workflow used in research.
   - Include explanations, code examples, and visualizations as necessary to guide users through the functionality.
   - Keep the notebook focused and clear, aiming to showcase the feature’s usage and applications.
   - As an additional step, "Clear All Outputs" in the notebook before and PR to substantially save space

4. **Submit a Pull Request**:
   - After implementing, testing, and documenting your feature, submit a pull request to the main repository on GitHub.
   - Provide a clear summary of your changes and link to relevant issues if applicable.

Following this workflow will help maintain the quality and usability of **hamageolib**. Thank you for your contributions!

