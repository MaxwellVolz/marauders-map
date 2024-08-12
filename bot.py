import click
import yaml
import keyboard
import cv2
import numpy as np
import os
import json

from PIL import ImageGrab

from utils.ascii_text import marauder_map_ascii

############################################
#                  KEYBINDS                #
############################################
KEYBIND = "num 2"
KEYBIND2 = "num 3"

# Examples
# KEYBIND = "ctrl+shift+d"


class MainController:
    def __init__(self, config_path="config.yaml", debug=True):
        # Load config
        self.config = self.load_config(config_path)
        self.debug = debug

        self.map_image_path = "./images/all_levels/GoblinCave-5x5-01.png"

        self.minimap = self.capture_minimap()
        self.player_location = self.get_player_location()
        self.map_displayed = False

        # Initialize any required state or resources here
        self.is_running = True

    def load_config(self, path):
        """Load configuration from a YAML file."""
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def capture_minimap(self):
        top_left = self.config["Minimap"]["TopLeft_XY"]
        bot_right = self.config["Minimap"]["BottomRight_XY"]
        minimap_region = (top_left[0], top_left[1], bot_right[0], bot_right[1])

        # Capture a specific region of the screen
        if self.debug:
            debug_img = cv2.imread("images/test/gobs_02.png")
            debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

            left, top, right, bottom = minimap_region
            minimap = debug_img_rgb[top:bottom, left:right]
        else:
            minimap = ImageGrab.grab(bbox=minimap_region)

        minimap_np = np.array(minimap)
        minimap_cv = cv2.cvtColor(minimap_np, cv2.COLOR_BGR2RGB)

        # cv2.imshow("Matched Region", minimap)
        # cv2.waitKey(0)

        return minimap_cv

    def scan(self):

        print("Scanning minimap screen...")

        # Step 1: Capture the minimap
        minimap = self.capture_minimap()

        # Adjust minimap scaling
        target_size = (550, 550)  # Example target size (width, height)
        self.minimap = resize_image(minimap, target_size=target_size)

        self.goblincave_scan()

    def get_player_location(self):
        goblin_ref = cv2.imread("./images/all_levels/GoblinCave-5x5-01.png")

        # get location
        result = cv2.matchTemplate(goblin_ref, self.minimap, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        player_location = [max_loc[0] + 270, max_loc[1] + 270]

        return player_location, max_loc

    def goblincave_scan(self):
        img_path = "./images/all_levels"
        goblin_img = "GoblinCave-5x5-01.png"
        goblin_ref = cv2.imread(os.path.join(img_path, goblin_img))

        # get location
        player_location, minimap_location = self.get_player_location()

        # DEBUGGING: minimap outline
        h, w = self.minimap.shape[:2]
        # cv2.rectangle(
        #     goblin_ref,
        #     minimap_location,
        #     (minimap_location[0] + w, minimap_location[1] + h),
        #     (0, 255, 0),
        #     2,
        # )
        # DEBUGGING: player dot
        cv2.circle(goblin_ref, player_location, 5, (0, 255, 255), 4)

        # cv2.imshow("Matched Region", goblin_ref)
        # cv2.waitKey(0)

        os.path.join(img_path, goblin_img)

        with open("./data/GoblinCave-5x5-01-N-best.json", "r") as f:
            data = json.load(f)

        trans_markers = self.translate_coords(data["markers"], goblin_ref, True)
        print(trans_markers)

        cv2.imwrite(os.path.join(img_path, "GoblinCave_with_icons.png"), goblin_ref)

        return

    def translate_coords(self, markers, goblin_ref, drawit=False):
        img_h, img_w = goblin_ref.shape[:2]

        translated_markers = []

        icons = {
            "rez": cv2.imread(
                "./images/icons/rez.png",
                cv2.IMREAD_UNCHANGED,
            ),
            "exit": cv2.imread(
                "./images/icons/exit.png",
                cv2.IMREAD_UNCHANGED,
            ),
            "boss": cv2.imread(
                "./images/icons/boss.png",
                cv2.IMREAD_UNCHANGED,
            ),
            # Add other icons here as needed
        }

        # Map and plot the coordinates
        for marker in markers:
            # Assuming lat and lng correspond to the x, y on the image
            # You might need to
            lat = marker["coordinates"]["lat"]
            lng = marker["coordinates"]["lng"]

            # Scale to maps dimensions
            scaled_lat, scaled_lng = scale_coordinates(
                lat, lng, 0, 400, 0, 400, 2048, 2048
            )

            # Transform the coordinates based on the origin
            transformed_lat, transformed_lng = transform_coordinates(
                scaled_lat, scaled_lng, img_h, img_w, origin="clockwise"
            )

            translated_markers.append(
                {
                    "marker_id": marker["id"],
                    "lat": transformed_lat,
                    "lng": transformed_lng,
                }
            )

            # TODO: save translated markers for fast live processing

            # Icons
            icon = marker.get("icon", "exit")
            icon_img = icons.get(icon)

            icon_h, icon_w = icon_img.shape[:2]

            top_left_x = int(transformed_lat - icon_w // 2)
            top_left_y = int(transformed_lng - 20 - icon_h // 2)

            overlay_img(goblin_ref, icon_img, top_left_x, top_left_y)

        cv2.imshow("Matched Region", goblin_ref)
        cv2.waitKey(0)

        return translated_markers

    def identify_map(self, minimap):
        # Compare captured minimap with reference images
        best_match = None
        best_match_score = 0
        images_path = "./images/current_level"

        for filename in os.listdir(images_path):
            if filename.endswith(".png"):
                reference_image = cv2.imread(os.path.join(images_path, filename))
                score = self.compare_images(minimap, reference_image)
                if score > best_match_score:
                    best_match_score = score
                    best_match = filename

        return best_match

    def compare_images(self, minimap, reference_image):
        # Use template matching to find the minimap in the reference image
        result = cv2.matchTemplate(reference_image, minimap, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Optionally, draw a rectangle around the matched region for debugging
        h, w = minimap.shape[:2]
        cv2.rectangle(
            reference_image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2
        )
        # cv2.imshow("Matched Region", reference_image)
        # cv2.waitKey(0)

        return max_val

    def start_key_listener(self):
        # Start listening for the keybind
        keyboard.add_hotkey(KEYBIND, self.scan)
        keyboard.add_hotkey(KEYBIND2, self.trigger_action)

    def trigger_action(self):
        # This method is triggered by the keybind
        print("Keybind triggered! Executing action...")

    def run(self):
        # Print help statements
        self.print_intro()

        # Start listening for keybind
        self.start_key_listener()

        # Keep the program running
        print(f"Waiting for keybind: {KEYBIND}")
        while self.is_running:
            pass

    def print_intro(self):
        print(marauder_map_ascii())
        print("This bot scans for minimap, and tries to help navigate.\n")
        print(f"Press '{KEYBIND}' to scan.\n")


@click.group()
def cli():
    """Main entry point for the CLI application."""
    pass


@cli.command()
def start():
    """Start the bot."""
    controller = MainController()
    controller.run()


def overlay_img(background, overlay, x, y):
    # Get dimensions
    h, w = overlay.shape[:2]

    # Check for out of bounds
    if x >= background.shape[1] or y >= background.shape[0]:
        return

    # Overlay bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, background.shape[1]), min(y + h, background.shape[0])

    # Overlay area dimensions
    overlay = overlay[(y1 - y) : (y2 - y), (x1 - x) : (x2 - x)]

    # Transparency mask (if the icon has an alpha channel)
    if overlay.shape[2] == 4:
        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (
                alpha_mask * overlay[:, :, c]
                + (1 - alpha_mask) * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay


def transform_coordinates(lat, lng, image_height, image_width, origin="top-left"):
    if origin == "bottom-left":
        # Flip vertically (y-axis)
        transformed_lat = image_height - lat
        transformed_lng = lng
    elif origin == "top-right":
        # Flip horizontally (x-axis)
        transformed_lat = lat
        transformed_lng = image_width - lng
    elif origin == "bottom-right":
        # Flip both axes
        transformed_lat = image_height - lat
        transformed_lng = image_width - lng
    elif origin == "clockwise":
        # Swap lat and lng and invert the new latitude to achieve a 90-degree clockwise rotation
        transformed_lat = lng
        transformed_lng = image_height - lat
    elif origin == "counterclockwise":
        # Swap lat and lng and invert the new longitude to achieve a 90-degree counterclockwise rotation
        transformed_lat = image_width - lng
        transformed_lng = lat
    elif origin == "flip":
        # Flip both axes (180-degree rotation)
        transformed_lat = image_height - lat
        transformed_lng = image_width - lng
    else:  # 'top-left' is the default
        transformed_lat = lat
        transformed_lng = lng

    return transformed_lat, transformed_lng


def resize_image(image, scale=None, target_size=None):
    """Resize the image to a specific scale or target size."""
    if scale:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
    elif target_size:
        resized_image = cv2.resize(image, target_size)
    else:
        resized_image = image
    return resized_image


def scale_coordinates(
    lat, lng, lat_min, lat_max, lng_min, lng_max, image_height, image_width
):
    # Normalize the lat and lng to a 0-1 range
    norm_lat = (lat - lat_min) / (lat_max - lat_min)
    norm_lng = (lng - lng_min) / (lng_max - lng_min)

    # Scale to the image dimensions
    scaled_lat = norm_lat * image_height
    scaled_lng = norm_lng * image_width

    return scaled_lat, scaled_lng


if __name__ == "__main__":
    cli()
