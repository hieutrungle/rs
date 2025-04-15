# import os
# from typing import List, Tuple, Optional
# import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# import tensorflow as tf

# policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
# tf.keras.mixed_precision.set_global_policy(policy)
# tf.config.experimental.set_memory_growth(
#     tf.config.experimental.list_physical_devices("GPU")[0], True
# )
# tf.random.set_seed(0)
# import math
# import numpy as np
# import mitsuba as mi
# import drjit as dr
# import sionna.rt
# from sionna.rt import (
#     load_scene,
#     PlanarArray,
#     Transmitter,
#     Receiver,
#     Camera,
#     PathSolver,
#     ITURadioMaterial,
#     SceneObject,
#     PathSolver,
#     RadioMapSolver,
#     DirectivePattern,
#     Paths,
#     RadioMap,
# )
# from rs.utils import utils
# import time
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors


# class SignalCoverage:
#     def __init__(self, sionna_config: dict, seed: Optional[int] = None):
#         self.sionna_config = sionna_config
#         self.seed = seed
#         self.viz_scene_path = self.sionna_config["viz_scene_path"]
#         self.compute_scene_path = self.sionna_config["compute_scene_path"]
#         self.tile_path = self.sionna_config["tile_path"]

#         self.cam = None
#         self.__prepare_camera()

#         self.viz_scene = load_scene(self.viz_scene_path, merge_shapes=True)
#         self.__prepare_radio_devices(self.viz_scene)
#         self.compute_scene = load_scene(self.compute_scene_path, merge_shapes=True)
#         self.__prepare_radio_devices(self.compute_scene)

#         self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
#         self.rt_pos = np.array(self.sionna_config["rt_positions"], dtype=np.float32)
#         self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

#         self.num_rx = len(self.rx_pos)
#         self.num_rt = len(self.rt_pos)
#         self.num_tx = len(self.tx_pos)

#         self.reflector_cols = 9  # odd value
#         self.reflector_rows = 9  # odd value
#         self.compute_reflectors = None
#         self.viz_reflectors = None
#         self.__init_reflectors()

#     def __prepare_radio_devices(self, scene: sionna.rt.Scene):
#         # in Hz; implicitly updates RadioMaterials
#         scene.frequency = self.sionna_config["frequency"]

#         # Device Setup
#         scene.tx_array = PlanarArray(
#             num_rows=self.sionna_config["tx_num_rows"],
#             num_cols=self.sionna_config["tx_num_cols"],
#             vertical_spacing=self.sionna_config["tx_vertical_spacing"],
#             horizontal_spacing=self.sionna_config["tx_horizontal_spacing"],
#             pattern=self.sionna_config["tx_pattern"],
#             polarization=self.sionna_config["tx_polarization"],
#         )
#         for i, (tx_pos, rt_pos) in enumerate(
#             zip(self.sionna_config["tx_positions"], self.sionna_config["rt_positions"])
#         ):
#             tx = Transmitter(
#                 name=f"tx_{i}",
#                 position=tx_pos,
#                 look_at=rt_pos,
#                 color=[0.05, 0.05, 0.9],
#                 display_radius=0.5,
#             )
#             scene.add(tx)

#         scene.rx_array = PlanarArray(
#             num_rows=self.sionna_config["rx_num_rows"],
#             num_cols=self.sionna_config["rx_num_cols"],
#             vertical_spacing=self.sionna_config["rx_vertical_spacing"],
#             horizontal_spacing=self.sionna_config["rx_horizontal_spacing"],
#             pattern=self.sionna_config["rx_pattern"],
#             polarization=self.sionna_config["rx_polarization"],
#         )

#         for i, (rx_pos, rx_orient) in enumerate(
#             zip(self.sionna_config["rx_positions"], self.sionna_config["rx_orientations"])
#         ):
#             rx = Receiver(
#                 name=f"rx_{i}",
#                 position=rx_pos,
#                 orientation=rx_orient,
#                 color=[0.99, 0.01, 0.99],
#                 display_radius=0.5,
#             )
#             scene.add(rx)

#     def __prepare_camera(self):
#         self.cam = Camera(
#             position=self.sionna_config["cam_position"],
#             look_at=self.sionna_config["cam_look_at"],
#         )
#         self.cam.look_at(self.sionna_config["cam_look_at"])

#     def __init_reflectors(self, color: Tuple[float] = (0.8, 0.1, 0.1)):

#         tile_material = ITURadioMaterial("tile-material", "metal", thickness=0.02, color=color)
#         self.compute_reflectors = [None for _ in range(self.num_rt)]
#         for i in range(self.num_rt):
#             tiles = [
#                 SceneObject(
#                     fname=self.tile_path,  # Simple mesh of a metal tile
#                     name=f"tile{i}-{j:03d}",
#                     radio_material=tile_material,
#                 )
#                 for j in range(self.reflector_cols * self.reflector_rows)
#             ]
#             self.compute_reflectors[i] = tiles
#             self.compute_scene.edit(add=tiles)

#         tile_material = ITURadioMaterial("viz-tile-material", "metal", thickness=0.02, color=color)
#         self.viz_reflectors = [None for _ in range(self.num_rt)]
#         for i in range(self.num_rt):
#             tiles = [
#                 SceneObject(
#                     fname=self.tile_path,  # Simple mesh of a metal tile
#                     name=f"viz_tile{i}-{j:03d}",
#                     radio_material=tile_material,
#                 )
#                 for j in range(self.reflector_cols * self.reflector_rows)
#             ]
#             self.viz_reflectors[i] = tiles
#             self.viz_scene.edit(add=tiles)

#     def update_reflectors(self, reflectors_pos, focals_pts, delta_z=0.22, visualize=False):

#         # delta_z = 22 / 100 # 20 cm

#         # Hexagonal grid
#         row_spacing = mi.Point3f(0.0, 0.0, delta_z)
#         col_spacing = mi.Point3f(delta_z * math.sin(math.radians(60)), 0.0, 0.0)

#         # reflector_orientations = [30, -30]

#         # reflectors_angles has 2 sets of angles for 2 reflectors
#         # angle set has 9 small arrays: [phi, theta0, theta1, theta2, ..., num_rows]
#         # phi is the angle of the reflector in the xy plane
#         # theta0 is the zenith angle of the first tile in the reflector
#         # theta1 is the zenith angle of the second tile in the reflector
#         # and so on
#         # num_rows is the number of rows in the reflector
#         # the first reflector is the one on the right
#         # the second reflector is the one on the left

#         angle_tilts = [-60, 60]
#         for i in range(self.num_rt):
#             default_first_tile_pos = reflectors_pos[i]
#             angle_tilt = angle_tilts[i]
#             rot_matrix = mi.Transform4f().rotate(axis=[0, 0, 1], angle=angle_tilt)
#             focal_pts = focals_pts[i]
#             for c in range(self.reflector_cols):
#                 focal_pt = focal_pts[c]
#                 # focal_pt = spherical2cartesian(focal_pt[0], focal_pt[1], focal_pt[2])
#                 focal_pt = mi.Point3f(focal_pt[0], focal_pt[1], focal_pt[2])
#                 if c % 2 == 1:
#                     first_tile_pos = default_first_tile_pos - mi.Point3f(0.0, 0.0, delta_z / 2.0)
#                 else:
#                     first_tile_pos = default_first_tile_pos
#                 for r in range(self.reflector_rows):
#                     tile_idx = c * self.reflector_rows + r
#                     if i == 0:
#                         tile_pos = -row_spacing * r + col_spacing * c
#                     else:
#                         tile_pos = -row_spacing * r - col_spacing * c
#                     tile_pos = rot_matrix @ tile_pos
#                     tile_pos += first_tile_pos

#                     self.compute_reflectors[i][tile_idx].position = tile_pos
#                     self.compute_reflectors[i][tile_idx].look_at(focal_pt)
#                     self.compute_reflectors[i][tile_idx].orientation = self.compute_reflectors[i][
#                         tile_idx
#                     ].orientation + mi.Point3f(0.0, math.radians(90), 0.0)

#         if visualize:
#             for i in range(self.num_rt):
#                 focal_pts = focals_pts[i]
#                 for c in range(self.reflector_cols):
#                     focal_pt = focal_pts[c]
#                     for r in range(self.reflector_rows):
#                         focal_pt = mi.Point3f(focal_pt[0], focal_pt[1], focal_pt[2])
#                         tile_idx = c * self.reflector_rows + r
#                         self.viz_reflectors[i][tile_idx].position = self.compute_reflectors[i][
#                             tile_idx
#                         ].position
#                         self.viz_reflectors[i][tile_idx].orientation = self.compute_reflectors[i][
#                             tile_idx
#                         ].orientation

#         # Get the center of the reflector
#         reflector_center = [None for _ in range(self.num_rt)]
#         for i in range(self.num_rt):
#             first_tile_pos = reflectors_pos[i]
#             angle_tilt = angle_tilts[i]
#             rot_matrix = mi.Transform4f().rotate(axis=[0, 0, 1], angle=angle_tilt)
#             r = self.reflector_rows // 2
#             c = self.reflector_cols // 2
#             if i == 0:
#                 tile_pos = -row_spacing * r + col_spacing * c
#             else:
#                 tile_pos = -row_spacing * r - col_spacing * c
#             tile_pos = rot_matrix @ tile_pos
#             tile_pos += first_tile_pos
#             reflector_center[i] = tile_pos

#             # change the orientation of transmitter to look at the center of the reflector
#             self.compute_scene.transmitters[f"tx_{i}"].look_at(tile_pos)
#             # self.viz_scene.transmitters[f"tx_{i}"].look_at(tile_pos)

#     def compute_cmap(self, **kwargs) -> RadioMap:
#         cm_kwargs = dict(
#             scene=self.compute_scene,
#             cell_size=self.sionna_config["rm_cell_size"],
#             max_depth=self.sionna_config["rm_max_depth"],
#             samples_per_tx=int(self.sionna_config["rm_num_samples"]),
#             diffuse_reflection=self.sionna_config["diffuse_reflection"],
#         )
#         if self.seed:
#             cm_kwargs["seed"] = self.seed
#         if kwargs:
#             cm_kwargs.update(kwargs)
#         rm_solver = RadioMapSolver()
#         cmap = rm_solver(**cm_kwargs)
#         return cmap

#     def compute_paths(self, **kwargs) -> Paths:
#         paths_kwargs = dict(
#             scene=self.compute_scene,
#             max_depth=self.sionna_config["path_max_depth"],
#             samples_per_src=int(self.sionna_config["path_num_samples"]),
#             diffuse_reflection=self.sionna_config["diffuse_reflection"],
#             synthetic_array=self.sionna_config["synthetic_array"],
#         )
#         if self.seed:
#             paths_kwargs["seed"] = self.seed
#         if kwargs:
#             paths_kwargs.update(kwargs)
#         p_solver = PathSolver()
#         paths = p_solver(**paths_kwargs)
#         return paths

#     def show(self, cmap: RadioMap = None):
#         if self.cam is None:
#             self.__prepare_camera()
#         self.scene.render(camera=self.cam)

#     def render_to_file(
#         self,
#         radio_map: RadioMap = None,
#         paths: Paths = None,
#         filename: Optional[str] = None,
#     ) -> None:

#         if filename is None:
#             img_dir = utils.get_image_dir(self.sionna_config)
#             render_filename = utils.create_filename(
#                 img_dir, f"{self.sionna_config['mitsuba_filename']}_00000.png"
#             )
#         else:
#             render_filename = filename
#         render_config = dict(
#             camera=self.cam,
#             paths=paths,
#             filename=render_filename,
#             radio_map=radio_map,
#             rm_vmin=self.sionna_config["rm_vmin"],
#             rm_vmax=self.sionna_config["rm_vmax"],
#             resolution=self.sionna_config["resolution"],
#             show_devices=True,
#         )
#         self.viz_scene.render_to_file(**render_config)

import os
from typing import List, Tuple, Optional
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import tensorflow as tf
import math
import numpy as np
import mitsuba as mi
import drjit as dr
import sionna.rt
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
    ITURadioMaterial,
    SceneObject,
    PathSolver,
    RadioMapSolver,
    DirectivePattern,
    Paths,
    RadioMap,
)
from rs.utils import utils
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class SignalCoverage:
    def __init__(self, sionna_config: dict, seed: Optional[int] = None):
        # policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        # tf.keras.mixed_precision.set_global_policy(policy)
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices("GPU")[0], True
        )
        tf.random.set_seed(seed)

        self.sionna_config = sionna_config
        self.seed = seed
        self.viz_scene_path = self.sionna_config["viz_scene_path"]
        self.compute_scene_path = self.sionna_config["compute_scene_path"]
        self.tile_path = self.sionna_config["tile_path"]

        self.cam = None
        self.__prepare_camera()

        self.viz_scene = load_scene(self.viz_scene_path, merge_shapes=True)
        self.__prepare_radio_devices(self.viz_scene)
        self.compute_scene = load_scene(self.compute_scene_path, merge_shapes=True)
        self.__prepare_radio_devices(self.compute_scene)

        self.rx_pos = np.array(self.sionna_config["rx_positions"], dtype=np.float32)
        self.rt_pos = np.array(self.sionna_config["rt_positions"], dtype=np.float32)
        self.tx_pos = np.array(self.sionna_config["tx_positions"], dtype=np.float32)

        self.num_rx = len(self.rx_pos)
        self.num_rt = len(self.rt_pos)
        self.num_tx = len(self.tx_pos)

        # self.reflector_cols = 9  # odd value
        # self.reflector_rows = 9  # odd value
        # self.compute_reflectors = None
        # self.viz_reflectors = None
        # self.tile_material = ITURadioMaterial(
        #     "tile-material", "metal", thickness=0.02, color=(0.8, 0.1, 0.1)
        # )
        # self.viz_tile_material = ITURadioMaterial(
        #     "viz-tile-material", "metal", thickness=0.02, color=(0.8, 0.1, 0.1)
        # )
        # self.__init_reflectors()

    def __prepare_radio_devices(self, scene: sionna.rt.Scene):
        # in Hz; implicitly updates RadioMaterials
        scene.frequency = self.sionna_config["frequency"]

        # Device Setup
        scene.tx_array = PlanarArray(
            num_rows=self.sionna_config["tx_num_rows"],
            num_cols=self.sionna_config["tx_num_cols"],
            vertical_spacing=self.sionna_config["tx_vertical_spacing"],
            horizontal_spacing=self.sionna_config["tx_horizontal_spacing"],
            pattern=self.sionna_config["tx_pattern"],
            polarization=self.sionna_config["tx_polarization"],
        )
        for i, (tx_pos, rt_pos) in enumerate(
            zip(self.sionna_config["tx_positions"], self.sionna_config["rt_positions"])
        ):
            tx = Transmitter(
                name=f"tx_{i}",
                position=tx_pos,
                look_at=rt_pos,
                color=[0.05, 0.05, 0.9],
                display_radius=0.5,
            )
            scene.add(tx)

        scene.rx_array = PlanarArray(
            num_rows=self.sionna_config["rx_num_rows"],
            num_cols=self.sionna_config["rx_num_cols"],
            vertical_spacing=self.sionna_config["rx_vertical_spacing"],
            horizontal_spacing=self.sionna_config["rx_horizontal_spacing"],
            pattern=self.sionna_config["rx_pattern"],
            polarization=self.sionna_config["rx_polarization"],
        )

        for i, (rx_pos, rx_orient) in enumerate(
            zip(self.sionna_config["rx_positions"], self.sionna_config["rx_orientations"])
        ):
            rx = Receiver(
                name=f"rx_{i}",
                position=rx_pos,
                orientation=rx_orient,
                color=[0.99, 0.01, 0.99],
                display_radius=0.5,
            )
            scene.add(rx)

    def __prepare_camera(self):
        self.cam = Camera(
            position=self.sionna_config["cam_position"],
            look_at=self.sionna_config["cam_look_at"],
        )
        self.cam.look_at(self.sionna_config["cam_look_at"])

    def __init_reflectors(self, color: Tuple[float] = (0.8, 0.1, 0.1)):

        self.compute_reflectors = [None for _ in range(self.num_rt)]
        for i in range(self.num_rt):
            tiles = [
                SceneObject(
                    fname=self.tile_path,  # Simple mesh of a metal tile
                    name=f"tile{i}-{j:03d}",
                    radio_material=self.tile_material,
                )
                for j in range(self.reflector_cols * self.reflector_rows)
            ]
            self.compute_reflectors[i] = tiles
            self.compute_scene.edit(add=tiles)

        # self.viz_reflectors = [None for _ in range(self.num_rt)]
        # for i in range(self.num_rt):
        #     tiles = [
        #         SceneObject(
        #             fname=self.tile_path,  # Simple mesh of a metal tile
        #             name=f"viz_tile{i}-{j:03d}",
        #             radio_material=self.viz_tile_material,
        #         )
        #         for j in range(self.reflector_cols * self.reflector_rows)
        #     ]
        #     self.viz_reflectors[i] = tiles
        #     self.viz_scene.edit(add=tiles)

    # assume that there is one focal_pt for each reflectors

    def update_reflectors(self, reflectors_pos, focals_pts, delta_z=0.22, visualize=False):

        # delta_z = 22 / 100 # 20 cm

        # Hexagonal grid
        row_spacing = mi.Point3f(0.0, 0.0, delta_z)
        col_spacing = mi.Point3f(delta_z * math.sin(math.radians(60)), 0.0, 0.0)

        # reflector_orientations = [30, -30]

        # reflectors_angles has 2 sets of angles for 2 reflectors
        # angle set has 9 small arrays: [phi, theta0, theta1, theta2, ..., num_rows]
        # phi is the angle of the reflector in the xy plane
        # theta0 is the zenith angle of the first tile in the reflector
        # theta1 is the zenith angle of the second tile in the reflector
        # and so on
        # num_rows is the number of rows in the reflector
        # the first reflector is the one on the right
        # the second reflector is the one on the left

        angle_tilts = [-60, 60]
        for i in range(self.num_rt):
            default_first_tile_pos = reflectors_pos[i]
            angle_tilt = angle_tilts[i]
            rot_matrix = mi.Transform4f().rotate(axis=[0, 0, 1], angle=angle_tilt)
            focal_pts = focals_pts[i]
            for c in range(self.reflector_cols):
                focal_pt = focal_pts[c]
                # focal_pt = spherical2cartesian(focal_pt[0], focal_pt[1], focal_pt[2])
                focal_pt = mi.Point3f(focal_pt[0], focal_pt[1], focal_pt[2])
                if c % 2 == 1:
                    first_tile_pos = default_first_tile_pos - mi.Point3f(0.0, 0.0, delta_z / 2.0)
                else:
                    first_tile_pos = default_first_tile_pos
                for r in range(self.reflector_rows):
                    tile_idx = c * self.reflector_rows + r
                    if i == 0:
                        tile_pos = -row_spacing * r + col_spacing * c
                    else:
                        tile_pos = -row_spacing * r - col_spacing * c
                    tile_pos = rot_matrix @ tile_pos
                    tile_pos += first_tile_pos

                    self.compute_reflectors[i][tile_idx].position = tile_pos
                    self.compute_reflectors[i][tile_idx].look_at(focal_pt)
                    self.compute_reflectors[i][tile_idx].orientation = self.compute_reflectors[i][
                        tile_idx
                    ].orientation + mi.Point3f(0.0, math.radians(90), 0.0)

                    # tile_orientation = self.compute_reflectors[i][tile_idx].orientation
                    # tile_orientation = mi.Point3f(
                    #     tile_orientation[0], tile_orientation[1], tile_orientation[2]
                    # )
                    # self.compute_scene.edit(remove=self.compute_reflectors[i][tile_idx])
                    # self.compute_reflectors[i][tile_idx] = SceneObject(
                    #     fname=self.tile_path,  # Simple mesh of a metal tile
                    #     name=f"tile{i}-{tile_idx:03d}",
                    #     radio_material=self.tile_material,
                    # )
                    # self.compute_scene.edit(add=self.compute_reflectors[i][tile_idx])
                    # self.compute_reflectors[i][tile_idx].position = tile_pos
                    # self.compute_reflectors[i][tile_idx].orientation = tile_orientation

        if visualize:
            for i in range(self.num_rt):
                focal_pts = focals_pts[i]
                for c in range(self.reflector_cols):
                    focal_pt = focal_pts[c]
                    for r in range(self.reflector_rows):
                        focal_pt = mi.Point3f(focal_pt[0], focal_pt[1], focal_pt[2])
                        tile_idx = c * self.reflector_rows + r
                        self.viz_reflectors[i][tile_idx].position = self.compute_reflectors[i][
                            tile_idx
                        ].position
                        self.viz_reflectors[i][tile_idx].orientation = self.compute_reflectors[i][
                            tile_idx
                        ].orientation

        # Get the center of the reflector
        reflector_center = [None for _ in range(self.num_rt)]
        for i in range(self.num_rt):
            first_tile_pos = reflectors_pos[i]
            angle_tilt = angle_tilts[i]
            rot_matrix = mi.Transform4f().rotate(axis=[0, 0, 1], angle=angle_tilt)
            r = self.reflector_rows // 2
            c = self.reflector_cols // 2
            if i == 0:
                tile_pos = -row_spacing * r + col_spacing * c
            else:
                tile_pos = -row_spacing * r - col_spacing * c
            tile_pos = rot_matrix @ tile_pos
            tile_pos += first_tile_pos
            reflector_center[i] = tile_pos

            # change the orientation of transmitter to look at the center of the reflector
            self.compute_scene.transmitters[f"tx_{i}"].look_at(tile_pos)
            # self.viz_scene.transmitters[f"tx_{i}"].look_at(tile_pos)

    def compute_cmap(self, **kwargs) -> RadioMap:
        cm_kwargs = dict(
            scene=self.compute_scene,
            cell_size=self.sionna_config["rm_cell_size"],
            max_depth=self.sionna_config["rm_max_depth"],
            samples_per_tx=int(self.sionna_config["rm_num_samples"]),
            diffuse_reflection=self.sionna_config["diffuse_reflection"],
        )
        if self.seed:
            cm_kwargs["seed"] = self.seed
        if kwargs:
            cm_kwargs.update(kwargs)
        rm_solver = RadioMapSolver()
        cmap = rm_solver(**cm_kwargs)
        return cmap

    def compute_paths(self, **kwargs) -> Paths:
        paths_kwargs = dict(
            scene=self.compute_scene,
            max_depth=self.sionna_config["path_max_depth"],
            samples_per_src=int(self.sionna_config["path_num_samples"]),
            diffuse_reflection=self.sionna_config["diffuse_reflection"],
            synthetic_array=self.sionna_config["synthetic_array"],
        )
        if self.seed:
            paths_kwargs["seed"] = self.seed
        if kwargs:
            paths_kwargs.update(kwargs)
        p_solver = PathSolver()
        paths = p_solver(**paths_kwargs)
        return paths

    def show(self, cmap: RadioMap = None):
        if self.cam is None:
            self.__prepare_camera()
        self.scene.render(camera=self.cam)

    def render_to_file(
        self,
        radio_map: RadioMap = None,
        paths: Paths = None,
        filename: Optional[str] = None,
    ) -> None:

        if filename is None:
            img_dir = utils.get_image_dir(self.sionna_config)
            render_filename = utils.create_filename(
                img_dir, f"{self.sionna_config['mitsuba_filename']}_00000.png"
            )
        else:
            render_filename = filename
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            radio_map=radio_map,
            rm_vmin=self.sionna_config["rm_vmin"],
            rm_vmax=self.sionna_config["rm_vmax"],
            resolution=self.sionna_config["resolution"],
            show_devices=True,
        )
        self.viz_scene.render_to_file(**render_config)

    def __getstate__(self):
        # Exclude non-picklable attributes
        state = self.__dict__.copy()
        non_picklable_attrs = [
            "viz_scene",
            # "compute_scene",
            "cam",
        ]
        for attr in non_picklable_attrs:
            state[attr] = None
        return state

    def __setstate__(self, state):
        # Restore the state and reinitialize non-picklable attributes
        self.__dict__.update(state)
        self.__prepare_camera()
        # self.viz_scene = load_scene(self.viz_scene_path, merge_shapes=True)
        # self.__prepare_radio_devices(self.viz_scene)
        # self.compute_scene = load_scene(self.compute_scene_path, merge_shapes=True)
        self.__prepare_radio_devices(self.compute_scene)
        # self.tile_material = ITURadioMaterial(
        #     "tile-material", "metal", thickness=0.02, color=(0.8, 0.1, 0.1)
        # )
        # self.viz_tile_material = ITURadioMaterial(
        #     "viz-tile-material", "metal", thickness=0.02, color=(0.8, 0.1, 0.1)
        # )
        # self.__init_reflectors()
