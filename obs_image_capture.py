# import os
from random import randint

import obspython as obs
import numpy as np
import cv2

from ctypes import *  # noqa
from ctypes.util import find_library

# Globals
source_name = "Dark and Darker"
hotkey_id = obs.OBS_INVALID_HOTKEY_ID
is_active = False


# FFI setup
ffi = CDLL(find_library("obs"))  # noqa
render_texture = obs.gs_texrender_create(obs.GS_RGBA, obs.GS_ZS_NONE)
surface = None


def wrap(funcname, restype, argtypes=None, use_lib=None):
    """Simplify wrapping ctypes functions"""
    if use_lib is not None:
        func = getattr(use_lib, funcname)
    else:
        func = getattr(ffi, funcname)
    func.restype = restype
    if argtypes is not None:
        func.argtypes = argtypes
    globals()[funcname] = func


class TexRender(Structure):  # noqa
    pass


class StageSurf(Structure):  # noqa
    pass


wrap(
    "gs_stage_texture", None, argtypes=[POINTER(StageSurf), POINTER(TexRender)]  # noqa
)  # noqa
wrap(
    "gs_stagesurface_create",
    POINTER(StageSurf),  # noqa
    argtypes=[c_uint, c_uint, c_int],  # noqa
)  # noqa
wrap(
    "gs_stagesurface_map",
    c_bool,  # noqa
    argtypes=[POINTER(StageSurf), POINTER(POINTER(c_ubyte)), POINTER(c_uint)],  # noqa
)
wrap("gs_stagesurface_destroy", None, argtypes=[POINTER(StageSurf)])  # noqa
wrap("gs_stagesurface_unmap", None, argtypes=[POINTER(StageSurf)])  # noqa


def script_description():
    return "A script to capture image data from a screen capture source."


# Called at script load
def script_load(settings):
    global hotkey_id
    hotkey_id = obs.obs_hotkey_register_frontend(
        script_path(), "Marauders Map", on_updatemap_hotkey
    )
    hotkey_save_array = obs.obs_data_get_array(settings, "Marauders Map")
    obs.obs_hotkey_load(hotkey_id, hotkey_save_array)
    obs.obs_data_array_release(hotkey_save_array)


def on_updatemap_hotkey(pressed):
    print(f"pressed: {pressed}")
    # global is_active
    # is_active = pressed


# Called every frame
# Initialize a timer variable outside the function
last_update_time = 0
debug_mode = True


def script_tick(seconds):
    global last_update_time

    last_update_time += seconds
    if last_update_time >= 3:

        last_update_time = 0  # Reset the timer

        if is_active:
            print("should be updating minimap")

            if debug_mode:
                minimap = (
                    load_debug_image()
                )  # Replace with your actual debug image loading function
            else:
                minimap = get_frame_data()

            # TODO: if debug use load image as frame_data

            target_size = (550, 550)
            minimap = resize_image(minimap, target_size=target_size)

            player_location = get_player_location(minimap)[0]

            print(f"player_location: {player_location[0]}")

            position_source_in_obs(player_location, "player_icon")

            # TODO: instead of drawing a circle we need to position a source in OBS
            # on top of an image of the map, at the correct location
            # cv2.circle(goblin_ref, player_location, 5, (0, 255, 255), 4)


def script_defaults(settings):
    obs.obs_data_set_default_string(settings, "source_name", "Dark and Darker")


def start_it():
    global is_active
    is_active = True


def stop_it():
    global is_active
    is_active = False


def script_properties():
    props = obs.obs_properties_create()

    obs.obs_properties_add_button(
        props,
        "start",
        "Start",
        lambda props, prop: (start_it()),
    )

    obs.obs_properties_add_button(
        props,
        "stop",
        "Stop",
        lambda props, prop: (stop_it()),
    )

    obs.obs_properties_add_button(
        props,
        "button2",
        "Capture minimap",
        lambda props, prop: (get_frame_data()),
    )
    return props


def load_debug_image():
    goblin_ref = cv2.imread(
        "C:/Users/narfa/Documents/_git/marauders-map/output/captured_frame_31827.png"
    )

    minimap_np = np.array(goblin_ref)
    minimap_cv = cv2.cvtColor(minimap_np, cv2.COLOR_BGR2RGB)

    return minimap_cv


def position_source_in_obs(player_location, source_name):

    print(f"player_location: {player_location} | source_name: {source_name}")
    return


def get_frame_data():
    source = obs.obs_get_source_by_name(source_name)
    if source is not None:
        print(f"Source found: {source_name}")

        # Full screen dimensions (adjust to match your source resolution)
        full_width = 3440
        full_height = 1440

        # Define the bottom-right portion dimensions
        capture_width = 800  # Example width of the bottom-right region
        capture_height = 600  # Example height of the bottom-right region

        # Calculate the top-left corner of the bottom-right region
        x = full_width - capture_width
        y = full_height - capture_height

        obs.obs_enter_graphics()
        if obs.gs_texrender_begin(render_texture, capture_width, capture_height):
            # Set the orthographic projection to match the bottom-right region
            obs.gs_ortho(
                float(x),
                float(x + capture_width),
                float(y + capture_height),
                float(y),
                -100.0,
                100.0,
            )

            # Render the source into the texture
            obs.obs_source_video_render(source)
            obs.gs_texrender_end(render_texture)

            global surface
            if not surface:
                surface = gs_stagesurface_create(  # noqa
                    c_uint(capture_width),  # noqa
                    c_uint(capture_height),  # noqa
                    c_int(obs.GS_RGBA),  # noqa
                )
            tex = obs.gs_texrender_get_texture(render_texture)
            tex = c_void_p(int(tex))  # noqa
            tex = cast(tex, POINTER(TexRender))  # noqa
            gs_stage_texture(surface, tex)  # noqa
            data = POINTER(c_ubyte)()  # noqa

            if gs_stagesurface_map(surface, byref(data), byref(c_uint(4))):  # noqa
                # Convert the raw data into a NumPy array
                np_data = np.ctypeslib.as_array(
                    data, shape=(capture_height, capture_width, 4)
                )

                # Flip the image (rotate by 180 degrees)
                np_data_flipped = np.flipud(np_data)

                # TODO: trim to these with numpy
                top_x, top_y, bot_x, bot_y = 480, 280, 740, 540
                trimmed_np_data = np_data_flipped[top_y:bot_y, top_x:bot_x]

                # Convert RGBA to BGR (OpenCV format)
                image_bgr = cv2.cvtColor(trimmed_np_data, cv2.COLOR_RGBA2BGR)
                # Save the captured image
                file_path = f"C:/Users/narfa/Documents/_git/marauders-map/output/captured_frame_{randint(10000, 99999)}.png"
                cv2.imwrite(file_path, image_bgr)
                print(f"Image saved to {file_path}")

                gs_stagesurface_unmap(surface)  # noqa

                minimap_np = np.array(trimmed_np_data)
                minimap_cv = cv2.cvtColor(minimap_np, cv2.COLOR_BGR2RGB)

            else:
                print("Failed to map the staging surface.")
            obs.gs_texrender_reset(render_texture)
        else:
            print("Failed to begin texture rendering.")
        obs.obs_source_release(source)
        obs.obs_leave_graphics()
    else:
        print(f"Source '{source_name}' not found.")

    return minimap_cv


def get_player_location(minimap):
    goblin_ref = cv2.imread(
        "C:/Users/narfa/Documents/_git/marauders-map/images/all_levels/GoblinCave-5x5-01.png"
    )

    print("read file")

    # get location
    result = cv2.matchTemplate(goblin_ref, minimap, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    player_location = [max_loc[0] + 270, max_loc[1] + 270]

    return player_location, max_loc


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
