import os

import obspython as obs
import numpy as np
import cv2

from ctypes import *
from ctypes.util import find_library

# Globals
source_name = "minimap"


# FFI setup
ffi = CDLL(find_library("obs"))
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


class TexRender(Structure):
    pass


class StageSurf(Structure):
    pass


wrap("gs_stage_texture", None, argtypes=[POINTER(StageSurf), POINTER(TexRender)])
wrap("gs_stagesurface_create", POINTER(StageSurf), argtypes=[c_uint, c_uint, c_int])
wrap(
    "gs_stagesurface_map",
    c_bool,
    argtypes=[POINTER(StageSurf), POINTER(POINTER(c_ubyte)), POINTER(c_uint)],
)
wrap("gs_stagesurface_destroy", None, argtypes=[POINTER(StageSurf)])
wrap("gs_stagesurface_unmap", None, argtypes=[POINTER(StageSurf)])


def script_description():
    return "A script to capture image data from a screen capture source."


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
                surface = gs_stagesurface_create(
                    c_uint(capture_width), c_uint(capture_height), c_int(obs.GS_RGBA)
                )
            tex = obs.gs_texrender_get_texture(render_texture)
            tex = c_void_p(int(tex))
            tex = cast(tex, POINTER(TexRender))
            gs_stage_texture(surface, tex)
            data = POINTER(c_ubyte)()

            if gs_stagesurface_map(surface, byref(data), byref(c_uint(4))):
                # Convert the raw data into a NumPy array
                np_data = np.ctypeslib.as_array(
                    data, shape=(capture_height, capture_width, 4)
                )

                # Flip the image (rotate by 180 degrees)
                np_data_flipped = np.flipud(np_data)

                # Convert RGBA to BGR (OpenCV format)
                image_bgr = cv2.cvtColor(np_data_flipped, cv2.COLOR_RGBA2BGR)

                # Save the captured image
                file_path = "C:/Users/narfa/Documents/_git/marauders-map/output/captured_frame.png"
                cv2.imwrite(file_path, image_bgr)
                print(f"Image saved to {file_path}")

                gs_stagesurface_unmap(surface)
            else:
                print("Failed to map the staging surface.")
            obs.gs_texrender_reset(render_texture)
        else:
            print("Failed to begin texture rendering.")
        obs.obs_source_release(source)
        obs.obs_leave_graphics()
    else:
        print(f"Source '{source_name}' not found.")


def script_update(settings):
    get_frame_data()


def script_defaults(settings):
    obs.obs_data_set_default_string(settings, "source_name", "minimap")


def script_properties():
    props = obs.obs_properties_create()

    # Drop-down list of sources
    list_property = obs.obs_properties_add_list(
        props,
        "source_name",
        "Source name",
        obs.OBS_COMBO_TYPE_LIST,
        obs.OBS_COMBO_FORMAT_STRING,
    )
    populate_list_property_with_source_names(list_property)

    # Button to refresh the drop-down list
    obs.obs_properties_add_button(
        props,
        "button",
        "Refresh list of sources",
        lambda props, prop: (
            True if populate_list_property_with_source_names(list_property) else True
        ),
    )

    obs.obs_properties_add_button(
        props,
        "button2",
        "Capture minimap",
        lambda props, prop: (get_frame_data()),
    )

    return props


def populate_list_property_with_source_names(list_property):
    sources = obs.obs_enum_sources()
    obs.obs_property_list_clear(list_property)
    obs.obs_property_list_add_string(list_property, "", "")
    for source in sources:
        name = obs.obs_source_get_name(source)
        obs.obs_property_list_add_string(list_property, name, name)
    obs.source_list_release(sources)
