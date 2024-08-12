import obspython as obs

source_name = "Overlay"  # Name of your overlay source
hotkey_id = obs.OBS_INVALID_HOTKEY_ID


def script_description():
    return (
        "A simple script to toggle the visibility of an overlay source with a hotkey."
    )


def script_load(settings):
    global hotkey_id
    hotkey_id = obs.obs_hotkey_register_frontend(
        "toggle_overlay_visibility", "Toggle Overlay Visibility", toggle_visibility
    )
    hotkey_save_array = obs.obs_data_get_array(
        settings, "toggle_overlay_visibility.hotkey"
    )
    obs.obs_hotkey_load(hotkey_id, hotkey_save_array)
    obs.obs_data_array_release(hotkey_save_array)


def script_unload():
    obs.obs_hotkey_unregister(toggle_visibility)


def toggle_visibility(pressed):
    if pressed:
        source = obs.obs_get_source_by_name(source_name)
        if source:
            current_visibility = obs.obs_source_visible(source)
            obs.obs_source_set_enabled(source, not current_visibility)
            obs.obs_source_release(source)


def script_update(settings):
    global source_name
    source_name = obs.obs_data_get_string(settings, "source")


def script_defaults(settings):
    obs.obs_data_set_default_string(settings, "source", "Overlay")


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_text(props, "source", "Source Name", obs.OBS_TEXT_DEFAULT)
    return props
