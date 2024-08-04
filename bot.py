import click
import keyboard
from PIL import ImageGrab
import cv2
import numpy as np
import os

from utils.ascii_text import marauder_map_ascii

############################################
#                  KEYBINDS                #
############################################
KEYBIND = "num 3"

# Examples
# KEYBIND = "ctrl+shift+d"


class MainController:
    def __init__(self):
        # Initialize any required state or resources here
        self.is_running = True

        # Coordinates (left, top, right, bottom)
        self.minimap_region = (100, 100, 300, 300)

    def capture_minimap(self):
        # Capture a specific region of the screen
        minimap = ImageGrab.grab(bbox=self.minimap_region)
        minimap_np = np.array(minimap)
        minimap_cv = cv2.cvtColor(
            minimap_np, cv2.COLOR_BGR2RGB
        )  # Convert PIL to OpenCV format
        return minimap_cv

    def scan(self):

        print("Scanning minimap screen...")

        # Step 1: Capture the minimap
        minimap = self.capture_minimap()

        # Step 2: Determine which map we are on
        map_name = self.identify_map(minimap)
        print(f"Map identified: {map_name}")

        # Step 3: Determine location (placeholder for further implementation)
        location = self.determine_location(minimap)
        print(f"Location determined: {location}")

    def identify_map(self, minimap):
        # Compare captured minimap with reference images
        best_match = None
        best_match_score = 0
        images_path = "./images"

        for filename in os.listdir(images_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                reference_image = cv2.imread(os.path.join(images_path, filename))
                score = self.compare_images(minimap, reference_image)
                if score > best_match_score:
                    best_match_score = score
                    best_match = filename

        return best_match

    def compare_images(self, img1, img2):
        # Resize images to the same size for comparison
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Calculate similarity (using correlation coefficient or another method)
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val

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
