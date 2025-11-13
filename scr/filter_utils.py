import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    uniform_filter,
    minimum_filter,
    maximum_filter,
    sobel,
    prewitt,
    laplace,
    gaussian_laplace,
    gaussian_gradient_magnitude,
)
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import os


def load_sample_image():
    size = 256
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    X, Y = np.meshgrid(x, y)

    circles = np.zeros((size, size))
    for radius in [1, 2, 3]:
        circle = np.sqrt(X**2 + Y**2)
        circles += ((circle > radius - 0.1) & (circle < radius + 0.1)).astype(float)

    return np.clip(circles, 0, 1)


def load_image_from_path(image_path):
    img = Image.open(image_path).convert("L")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return np.array(img) / 255.0


def get_available_images():
    img_dir = "data/img"
    if os.path.exists(img_dir):
        return [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    return []


def apply_filter(image, filter_name, params):
    """Apply a scipy filter with given parameters"""
    if image is None:
        return None

    img = image.copy()

    if filter_name == "Gaussian Blur":
        return gaussian_filter(img, sigma=params.get("sigma", 1.0))

    elif filter_name == "Median Filter":
        return median_filter(img, size=params.get("size", 3))

    elif filter_name == "Uniform Filter":
        return uniform_filter(img, size=params.get("size", 3))

    elif filter_name == "Minimum Filter":
        return minimum_filter(img, size=params.get("size", 3))

    elif filter_name == "Maximum Filter":
        return maximum_filter(img, size=params.get("size", 3))

    elif filter_name == "Sobel X":
        return np.abs(sobel(img, axis=0))

    elif filter_name == "Sobel Y":
        return np.abs(sobel(img, axis=1))

    elif filter_name == "Sobel Combined":
        sx = sobel(img, axis=0)
        sy = sobel(img, axis=1)
        return np.sqrt(sx**2 + sy**2)

    elif filter_name == "Prewitt X":
        return np.abs(prewitt(img, axis=0))

    elif filter_name == "Prewitt Y":
        return np.abs(prewitt(img, axis=1))

    elif filter_name == "Laplace":
        return np.abs(laplace(img))

    elif filter_name == "Laplacian of Gaussian":
        return np.abs(gaussian_laplace(img, sigma=params.get("sigma", 1.0)))

    elif filter_name == "Gaussian Gradient":
        return gaussian_gradient_magnitude(img, sigma=params.get("sigma", 1.0))

    elif filter_name == "None":
        return img

    return img


def apply_operation(img1, img2, operation):
    """Apply mathematical operation between two images"""
    if img1 is None or img2 is None:
        return img1

    if operation == "None":
        return img2
    elif operation == "Add":
        return img1 + img2
    elif operation == "Subtract":
        return img1 - img2
    elif operation == "Multiply":
        return img1 * img2
    elif operation == "Divide":
        return np.divide(img1, img2 + 1e-10)
    elif operation == "Average":
        return (img1 + img2) / 2
    elif operation == "Max":
        return np.maximum(img1, img2)
    elif operation == "Min":
        return np.minimum(img1, img2)

    return img2


def normalize_image(img):
    """Normalize image to 0-1 range"""
    if img is None:
        return None
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-10:
        return img
    return (img - img_min) / (img_max - img_min)


def create_filter_controls(filter_num):
    """Create parameter controls for a single filter"""
    filter_options = [
        "None",
        "Gaussian Blur",
        "Median Filter",
        "Uniform Filter",
        "Minimum Filter",
        "Maximum Filter",
        "Sobel X",
        "Sobel Y",
        "Sobel Combined",
        "Prewitt X",
        "Prewitt Y",
        "Laplace",
        "Laplacian of Gaussian",
        "Gaussian Gradient",
    ]

    filter_dropdown = widgets.Dropdown(
        options=filter_options,
        value="None",
        description=f"Filter {filter_num}:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="250px"),
    )

    sigma_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        description="Sigma:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="300px", display="none"),
    )

    size_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=15,
        step=2,
        description="Size:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="300px", display="none"),
    )

    operation_dropdown = widgets.Dropdown(
        options=[
            "None",
            "Add",
            "Subtract",
            "Multiply",
            "Divide",
            "Average",
            "Max",
            "Min",
        ],
        value="None",
        description="Then:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )

    return {
        "filter": filter_dropdown,
        "sigma": sigma_slider,
        "size": size_slider,
        "operation": operation_dropdown,
    }


def create_filter_pipeline_widget(image):
    """Create the main filter pipeline widget"""

    if image is None:
        print("Please select an image first")
        return

    updating = False

    filter_controls = [create_filter_controls(i + 1) for i in range(5)]

    output_widget = widgets.Output()

    def update_parameter_visibility(controls):
        """Show/hide parameters based on selected filter"""
        filter_name = controls["filter"].value

        needs_sigma = filter_name in [
            "Gaussian Blur",
            "Laplacian of Gaussian",
            "Gaussian Gradient",
        ]
        needs_size = filter_name in [
            "Median Filter",
            "Uniform Filter",
            "Minimum Filter",
            "Maximum Filter",
        ]

        controls["sigma"].layout.display = "flex" if needs_sigma else "none"
        controls["size"].layout.display = "flex" if needs_size else "none"

    def apply_pipeline():
        """Apply the entire filter pipeline"""
        nonlocal updating
        if updating:
            return

        result = image.copy()

        for i, controls in enumerate(filter_controls):
            filter_name = controls["filter"].value

            if filter_name == "None":
                continue

            params = {
                "sigma": controls["sigma"].value,
                "size": controls["size"].value,
            }

            filtered = apply_filter(result, filter_name, params)

            if i < len(filter_controls) - 1:
                operation = controls["operation"].value
                result = apply_operation(result, filtered, operation)
            else:
                result = filtered

        result = normalize_image(result)

        with output_widget:
            output_widget.clear_output(wait=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(result, cmap="gray")
            axes[1].set_title("Filtered Result")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

    def on_filter_change(change, controls):
        update_parameter_visibility(controls)
        apply_pipeline()

    def on_param_change(change):
        apply_pipeline()

    ui_elements = []

    for i, controls in enumerate(filter_controls):
        controls["filter"].observe(
            lambda change, c=controls: on_filter_change(change, c), names="value"
        )
        controls["sigma"].observe(on_param_change, names="value")
        controls["size"].observe(on_param_change, names="value")
        controls["operation"].observe(on_param_change, names="value")

        update_parameter_visibility(controls)

        filter_box = widgets.VBox(
            [
                controls["filter"],
                controls["sigma"],
                controls["size"],
                controls["operation"] if i < 4 else widgets.HTML(""),
            ],
            layout=widgets.Layout(
                padding="10px",
                border="1px solid #ddd",
                margin="5px",
                border_radius="5px",
            ),
        )
        ui_elements.append(filter_box)

    apply_button = widgets.Button(
        description="Apply Pipeline",
        button_style="primary",
        layout=widgets.Layout(width="200px", margin="10px"),
    )

    apply_button.on_click(lambda b: apply_pipeline())

    controls_box = widgets.VBox(
        ui_elements + [apply_button],
        layout=widgets.Layout(padding="20px", align_items="center"),
    )

    main_widget = widgets.VBox(
        [controls_box, output_widget],
        layout=widgets.Layout(padding="20px"),
    )

    display(main_widget)

    apply_pipeline()
