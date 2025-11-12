import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def load_sample_image(size=(256, 256)):
    img = np.zeros(size)
    img[size[0] // 4 : 3 * size[0] // 4, size[1] // 4 : 3 * size[1] // 4] = 1

    for i in range(3):
        center_y = np.random.randint(size[0] // 4, 3 * size[0] // 4)
        center_x = np.random.randint(size[1] // 4, 3 * size[1] // 4)
        radius = np.random.randint(10, 30)
        y, x = np.ogrid[: size[0], : size[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        img[mask] = 0.5

    return img


def load_image_from_path(image_path, max_size=512):
    img = Image.open(image_path).convert("L")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.array(img) / 255.0


def get_available_images(data_folder="data/img"):
    if os.path.exists(data_folder):
        images = [
            f
            for f in os.listdir(data_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        return sorted(images)
    return []


def apply_convolution(image, kernel):
    kernel = np.array(kernel)
    if kernel.size == 0:
        return image
    result = signal.convolve2d(image, kernel, mode="same", boundary="symm")
    return result


def normalize_image(image):
    img_min, img_max = image.min(), image.max()
    if img_max - img_min > 0:
        return (image - img_min) / (img_max - img_min)
    return image


def create_kernel_from_list(values_list):
    return np.array(values_list)


def create_kernel_from_values(k00, k01, k02, k10, k11, k12, k20, k21, k22):
    return np.array([[k00, k01, k02], [k10, k11, k12], [k20, k21, k22]])


def get_preset_kernel(kernel_name, size=3):
    if size == 3:
        kernels = {
            "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "Box Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
            "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        }
    elif size == 5:
        kernels = {
            "Identity": np.logical_and(np.eye(size), np.eye(size)[::-1]),
            "Edge Detection": np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, 24, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                ]
            ),
            "Sharpen": np.array(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, -1, -1, 0],
                    [-1, -1, 13, -1, -1],
                    [0, -1, -1, -1, 0],
                    [0, 0, -1, 0, 0],
                ]
            ),
            "Box Blur": np.ones((5, 5)) / 25,
            "Gaussian Blur": np.array(
                [
                    [1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1],
                ]
            )
            / 256,
            "Sobel X": np.array(
                [
                    [-1, -2, 0, 2, 1],
                    [-2, -3, 0, 3, 2],
                    [-3, -5, 0, 5, 3],
                    [-2, -3, 0, 3, 2],
                    [-1, -2, 0, 2, 1],
                ]
            ),
            "Sobel Y": np.array(
                [
                    [-1, -2, -3, -2, -1],
                    [-2, -3, -5, -3, -2],
                    [0, 0, 0, 0, 0],
                    [2, 3, 5, 3, 2],
                    [1, 2, 3, 2, 1],
                ]
            ),
            "Emboss": np.array(
                [
                    [-2, -1, 0, 0, 0],
                    [-1, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 2],
                ]
            ),
        }
    elif size == 7:
        kernels = {
            "Identity": np.logical_and(np.eye(size), np.eye(size)[::-1]),
            "Edge Detection": np.ones((7, 7)) * -1,
            "Sharpen": np.ones((7, 7)) * -1,
            "Box Blur": np.ones((7, 7)) / 49,
            "Gaussian Blur": np.array(
                [
                    [0, 0, 1, 2, 1, 0, 0],
                    [0, 3, 13, 22, 13, 3, 0],
                    [1, 13, 59, 97, 59, 13, 1],
                    [2, 22, 97, 159, 97, 22, 2],
                    [1, 13, 59, 97, 59, 13, 1],
                    [0, 3, 13, 22, 13, 3, 0],
                    [0, 0, 1, 2, 1, 0, 0],
                ]
            )
            / 1003,
            "Sobel X": np.array(
                [
                    [-1, -2, -3, 0, 3, 2, 1],
                    [-2, -3, -5, 0, 5, 3, 2],
                    [-3, -5, -7, 0, 7, 5, 3],
                    [-4, -7, -9, 0, 9, 7, 4],
                    [-3, -5, -7, 0, 7, 5, 3],
                    [-2, -3, -5, 0, 5, 3, 2],
                    [-1, -2, -3, 0, 3, 2, 1],
                ]
            ),
            "Sobel Y": np.array(
                [
                    [-1, -2, -3, -4, -3, -2, -1],
                    [-2, -3, -5, -7, -5, -3, -2],
                    [-3, -5, -7, -9, -7, -5, -3],
                    [0, 0, 0, 0, 0, 0, 0],
                    [3, 5, 7, 9, 7, 5, 3],
                    [2, 3, 5, 7, 5, 3, 2],
                    [1, 2, 3, 4, 3, 2, 1],
                ]
            ),
            "Emboss": np.array(
                [
                    [-2, -1, -1, 0, 0, 0, 0],
                    [-1, -1, -1, 0, 0, 0, 0],
                    [-1, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 2],
                ]
            ),
        }
        kernels["Edge Detection"][3, 3] = 48
        kernels["Sharpen"][3, 3] = 49
    else:
        center = size // 2
        kernels = {
            "Identity": np.logical_and(np.eye(size), np.eye(size)[::-1]),
            "Edge Detection": np.ones((size, size)) * -1,
            "Sharpen": np.ones((size, size)) * -1,
            "Box Blur": np.ones((size, size)) / (size * size),
            "Gaussian Blur": np.ones((size, size)) / (size * size),
            "Sobel X": np.zeros((size, size)),
            "Sobel Y": np.zeros((size, size)),
            "Emboss": np.zeros((size, size)),
        }
        kernels["Edge Detection"][center, center] = size * size - 1
        kernels["Sharpen"][center, center] = size * size

    return kernels.get(kernel_name, np.eye(size))


def plot_convolution_result(original_image, kernel, ax1, ax2):
    convolved = apply_convolution(original_image, kernel)
    normalized = normalize_image(convolved)

    ax1.clear()
    ax2.clear()

    ax1.imshow(original_image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(normalized, cmap="gray")
    ax2.set_title("Convolved Image")
    ax2.axis("off")


def create_convolution_widget(original_image):
    import ipywidgets as widgets
    from IPython.display import display

    output = widgets.Output()

    kernel_size_dropdown = widgets.Dropdown(
        options=[3, 5, 7, 9, 11],
        value=3,
        description="Size:",
        style={"description_width": "40px"},
        layout=widgets.Layout(width="120px"),
    )

    preset_dropdown = widgets.Dropdown(
        options=[
            "Custom",
            "Identity",
            "Edge Detection",
            "Sharpen",
            "Box Blur",
            "Gaussian Blur",
            "Sobel X",
            "Sobel Y",
            "Emboss",
        ],
        value="Identity",
        description="Preset:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="300px"),
    )

    input_mode_dropdown = widgets.Dropdown(
        options=["Text Input", "Individual Boxes"],
        value="Individual Boxes",
        description="Input Mode:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="250px"),
    )

    kernel_text = widgets.Textarea(
        value="0 0 0\n0 1 0\n0 0 0",
        placeholder="Enter kernel as rows, space-separated values",
        description="Text Input:",
        layout=widgets.Layout(width="200px", height="80px"),
        style={"description_width": "80px"},
    )

    fill_all = widgets.FloatText(
        value=0.0,
        description="Fill all:",
        layout=widgets.Layout(width="150px"),
        style={"description_width": "50px"},
    )

    fill_button = widgets.Button(
        description="Apply",
        button_style="info",
        layout=widgets.Layout(width="80px"),
    )

    kernel_inputs = []
    kernel_grid = None
    updating = False

    def create_kernel_grid(size):
        nonlocal kernel_inputs, kernel_grid
        kernel_inputs = []

        for i in range(size):
            row = []
            for j in range(size):
                val = 1.0 if i == size // 2 and j == size // 2 else 0.0
                w = widgets.FloatText(
                    value=val, description="", layout=widgets.Layout(width="70px")
                )
                row.append(w)
            kernel_inputs.append(row)

        for row in kernel_inputs:
            for w in row:
                w.observe(on_individual_change, names="value")

        kernel_grid = widgets.GridBox(
            children=[item for row in kernel_inputs for item in row],
            layout=widgets.Layout(
                grid_template_columns=f"repeat({size}, 70px)", grid_gap="3px"
            ),
        )
        return kernel_grid

    def get_kernel_from_inputs():
        return np.array([[w.value for w in row] for row in kernel_inputs])

    def update_plot():
        kernel = get_kernel_from_inputs()

        with output:
            output.clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            plot_convolution_result(original_image, kernel, ax1, ax2)
            plt.tight_layout()
            plt.show()

    def update_kernel_text_from_inputs():
        nonlocal updating
        kernel = get_kernel_from_inputs()
        kernel_text.value = "\n".join(
            [
                " ".join([f"{kernel[i, j]:.2f}" for j in range(kernel.shape[1])])
                for i in range(kernel.shape[0])
            ]
        )

        char_width = 10
        line_height = 11
        padding = 100

        max_line_length = max(len(line) for line in kernel_text.value.split("\n"))
        text_width = max_line_length * char_width + padding
        text_height = kernel.shape[0] * line_height + 30

        kernel_text.layout.width = f"{text_width}px"
        kernel_text.layout.height = f"{text_height}px"

    def on_individual_change(change):
        nonlocal updating
        if updating:
            return
        updating = True
        update_kernel_text_from_inputs()
        updating = False
        update_plot()

    def on_preset_change(change):
        nonlocal updating
        if change["new"] != "Custom":
            updating = True
            size = kernel_size_dropdown.value
            kernel = get_preset_kernel(change["new"], size)
            for i in range(len(kernel_inputs)):
                for j in range(len(kernel_inputs[0])):
                    kernel_inputs[i][j].value = kernel[i, j]
            update_kernel_text_from_inputs()
            updating = False
            update_plot()

    def on_text_change(change):
        nonlocal updating
        if updating:
            return
        try:
            updating = True
            lines = change["new"].strip().split("\n")
            size = len(kernel_inputs)
            if len(lines) == size:
                for i, line in enumerate(lines):
                    values = line.split()
                    if len(values) == size:
                        for j, val in enumerate(values):
                            kernel_inputs[i][j].value = float(val)
            updating = False
            update_plot()
        except:
            updating = False

    def on_fill_button_click(b):
        nonlocal updating
        updating = True
        val = fill_all.value
        for row in kernel_inputs:
            for widget in row:
                widget.value = val
        update_kernel_text_from_inputs()
        updating = False
        update_plot()

    def on_size_change(change):
        nonlocal updating
        updating = True
        size = change["new"]
        new_grid = create_kernel_grid(size)

        kernel = get_preset_kernel("Identity", size)
        for i in range(size):
            for j in range(size):
                kernel_inputs[i][j].value = kernel[i, j]

        update_kernel_text_from_inputs()

        grid_container.children = [new_grid]

        updating = False
        update_plot()

    def on_input_mode_change(change):
        if change["new"] == "Text Input":
            kernel_editor.children = [text_container]
        elif change["new"] == "Individual Boxes":
            kernel_editor.children = [grid_container]

    preset_dropdown.observe(on_preset_change, names="value")
    kernel_text.observe(on_text_change, names="value")
    fill_button.on_click(on_fill_button_click)
    kernel_size_dropdown.observe(on_size_change, names="value")
    input_mode_dropdown.observe(on_input_mode_change, names="value")

    initial_grid = create_kernel_grid(3)

    top_controls = widgets.HBox(
        [kernel_size_dropdown, preset_dropdown, fill_all, fill_button],
        layout=widgets.Layout(gap="15px", justify_content="center"),
    )

    input_mode_row = widgets.HBox(
        [input_mode_dropdown],
        layout=widgets.Layout(justify_content="center"),
    )

    grid_container = widgets.VBox(
        [initial_grid],
        layout=widgets.Layout(align_items="center"),
    )

    text_container = widgets.VBox(
        [kernel_text],
        layout=widgets.Layout(align_items="center"),
    )

    kernel_editor = widgets.VBox(
        [grid_container],
        layout=widgets.Layout(padding="10px 0px", align_items="center"),
    )

    update_plot()

    display(
        widgets.VBox(
            [
                top_controls,
                widgets.HTML("<br>"),
                input_mode_row,
                widgets.HTML("<br>"),
                kernel_editor,
                widgets.HTML("<br>"),
                output,
            ],
            layout=widgets.Layout(padding="20px"),
        )
    )
