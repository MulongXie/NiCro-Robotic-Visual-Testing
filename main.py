from NiCro import NiCro

nicro = NiCro()

nicro.get_devices_info()

nicro.detect_gui_info_for_all_devices(is_load=False)

nicro.control_multiple_devices_through_source_device(is_replay=False)
