import click
import yaml
import keyboard
import cv2
import numpy as np
import os

from PIL import ImageGrab

from utils.ascii_text import marauder_map_ascii

############################################
#                  KEYBINDS                #
############################################
KEYBIND = "num 3"

# Examples
# KEYBIND = "ctrl+shift+d"


class MainController:
    def __init__(self, config_path="config.yaml", debug=True):

        # Load config
        self.config = self.load_config(config_path)

        self.debug = debug

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

        cv2.imshow("Matched Region", minimap)
        cv2.waitKey(0)

        return minimap_cv

    def scan(self):

        print("Scanning minimap screen...")

        # Step 1: Capture the minimap
        minimap = self.capture_minimap()

        # Adjust minimap scaling
        target_size = (550, 550)  # Example target size (width, height)
        minimap_resized = self.resize_image(minimap, target_size=target_size)

        cv2.imshow("minimap_resized", minimap_resized)
        cv2.waitKey(0)

        # Step 2: Determine which map we are on
        map_name = self.identify_map(minimap_resized)
        print(f"Map identified: {map_name}")

        # Step 3: Determine location (placeholder for further implementation)
        location = self.determine_location(minimap)
        print(f"Location determined: {location}")

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
        cv2.imshow("Matched Region", reference_image)
        cv2.waitKey(0)

        return max_val

    def resize_image(self, image, scale=None, target_size=None):
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

    def determine_location(self, minimap):
        # Placeholder method for determining location
        # Implement location detection based on specific requirements
        return "Unknown Location"

    def start_key_listener(self):
        # Start listening for the keybind
        keyboard.add_hotkey(KEYBIND, self.trigger_action)

    def trigger_action(self):
        # This method is triggered by the keybind
        print("Keybind triggered! Executing action...")
        self.scan()

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


if __name__ == "__main__":
    cli()
