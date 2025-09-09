'''
inject the result in res.npz into model.vrm and exports as res_textured.vrm
'''
import argparse
import yaml
import os
import numpy as np
from numpy import ndarray

from typing import Tuple, Union, List

import argparse
from tqdm import tqdm
from box import Box

from scipy.spatial import cKDTree

import open3d as o3d
import itertools

import bpy
from mathutils import Vector

from ..data.raw_data import RawData, RawSkin
from ..data.extract import process_mesh, process_armature, get_arranged_bones

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('--id', type=int)
    return parser.parse_args()

def clean_bpy():
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

def load(filepath: str, return_armature: bool=False):
    if return_armature:
        old_objs = set(bpy.context.scene.objects)

    if not os.path.exists(filepath):
        raise ValueError(f'File {filepath} does not exist !')
    try:
        if filepath.endswith(".vrm"):
            # enable vrm addon and load vrm model
            bpy.ops.preferences.addon_enable(module='vrm')
            
            bpy.ops.import_scene.vrm(
                filepath=filepath,
                use_addon_preferences=True,
                extract_textures_into_folder=False,
                make_new_texture_folder=False,
                set_shading_type_to_material_on_import=False,
                set_view_transform_to_standard_on_import=True,
                set_armature_display_to_wire=True,
                set_armature_display_to_show_in_front=True,
                set_armature_bone_shape_to_default=True,
                disable_bake=True, # customized option for better performance
            )
        elif filepath.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=filepath)
        elif filepath.endswith(".fbx") or filepath.endswith(".FBX"):
            bpy.ops.import_scene.fbx(filepath=filepath, ignore_leaf_bones=False, use_image_search=False)
        elif filepath.endswith(".glb") or filepath.endswith(".gltf"):
            bpy.ops.import_scene.gltf(filepath=filepath, import_pack_images=False)
        elif filepath.endswith(".dae"):
            bpy.ops.wm.collada_import(filepath=filepath)
        elif filepath.endswith(".blend"):
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        else:
            raise ValueError(f"not suported type {filepath}")
    except:
        raise ValueError(f"failed to load {filepath}")
    if return_armature:
        armature = [x for x in set(bpy.context.scene.objects)-old_objs if x.type=="ARMATURE"]
        if len(armature)==0:
            return None
        if len(armature)>1:
            raise ValueError(f"multiple armatures found")
        armature = armature[0]
        
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in bpy.data.armatures[0].edit_bones:
            bone.roll = 0. # change all roll to 0. to prevent weird behaviour

        bpy.ops.object.mode_set(mode='OBJECT')
        armature.select_set(False)
        
        bpy.ops.object.select_all(action='DESELECT')
        return armature

def get_skin(arranged_bones):
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    index = {}
    for (id, pbone) in enumerate(arranged_bones):
        index[pbone.name] = id
    _dict_skin = {}
    total_bones = len(arranged_bones)
    for obj in meshes:
        total_vertices = len(obj.data.vertices)
        skin_weight = np.zeros((total_vertices, total_bones))
        obj_group_names = [g.name for g in obj.vertex_groups]
        obj_verts = obj.data.vertices
        for bone in arranged_bones:
            if bone.name not in obj_group_names:
                continue

            gidx = obj.vertex_groups[bone.name].index
            bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
            for v in bone_verts:
                which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                w = v.groups[which[0]].weight
                skin_weight[v.index, index[bone.name]] = w
        _dict_skin[obj.name] = {
            'skin': skin_weight,
        }
    
    skin = np.concatenate([
        _dict_skin[d]['skin'] for d in _dict_skin
    ], axis=0)
    return skin

def axis(a: np.ndarray):
    b = np.concatenate([-a[:, 0:1], -a[:, 1:2], a[:, 2:3]], axis=1)
    return b

def get_correct_orientation_kdtree(a: np.ndarray, b: np.ndarray, bones: np.ndarray, num: int=16384) -> np.ndarray:
    '''
    a: sampled_vertiecs
    b: mesh_vertices
    '''
    min_loss = float('inf')
    best_transformed = a.copy()
    axis_permutations = list(itertools.permutations([0, 1, 2]))
    sign_combinations = [(x, y, z) for x in [1, -1] 
                        for y in [1, -1] 
                        for z in [1, -1]]
    _bones = bones.copy()
    for perm in axis_permutations:
        permuted_a = a[np.random.permutation(a.shape[0])[:num]][:, perm]
        for signs in sign_combinations:
            transformed = permuted_a * np.array(signs)
            tree = cKDTree(transformed)
            distances, indices = tree.query(b)
            current_loss = distances.mean()
            if current_loss < min_loss: # prevent from mirroring
                min_loss = current_loss
                best_transformed = a[:, perm] * np.array(signs)
                bones[:, :3] = _bones[:, :3][:, perm] * np.array(signs)
                bones[:, 3:] = _bones[:, 3:][:, perm] * np.array(signs)
    
    return best_transformed, bones

def denormalize_vertices(mesh_vertices: ndarray, vertices: ndarray, bones: ndarray) -> np.ndarray:
    min_vals = np.min(mesh_vertices, axis=0)
    max_vals = np.max(mesh_vertices, axis=0)
    center = (min_vals + max_vals) / 2
    scale = np.max(max_vals - min_vals) / 2
    denormalized_vertices = vertices * scale + center
    denormalized_bones = bones * scale
    denormalized_bones[:, :3] += center
    denormalized_bones[:, 3:] += center

    return denormalized_vertices, denormalized_bones

def get_matrix(ob):
    m = np.eye(4)
    while ob:
        if hasattr(ob, 'matrix_world'):
            m = m @ np.array(ob.matrix_world)
        ob = ob.parent
    return m

def make_armature(
    vertices: ndarray,
    bones: ndarray, # (joint, tail)
    parents: list[Union[int, None]],
    names: list[str],
    skin: ndarray,
    group_per_vertex: int=4,
    add_root: bool=False,
    is_vrm: bool=False,
):
    context = bpy.context
    
    mesh_vertices = []
    local_coord = np.eye(4)
    local_parent = None
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        if ob.parent is not None:
            local_coord = get_matrix(ob.parent)
            local_parent = ob.parent
        m = np.array(ob.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        for v in ob.data.vertices:
            mesh_vertices.append(matrix_world_rot @ np.array(v.co) + matrix_world_bias)

    mesh_vertices = np.stack(mesh_vertices)
    vertices, bones = denormalize_vertices(mesh_vertices, vertices, bones)
    
    bpy.ops.object.add(type="ARMATURE", location=(0, 0, 0))
    armature = context.object
    if hasattr(armature.data, 'vrm_addon_extension'):
        armature.data.vrm_addon_extension.spec_version = "1.0"
        humanoid = armature.data.vrm_addon_extension.vrm1.humanoid
        is_vrm = True
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    if add_root:
        bone_root = edit_bones.new('Root')
        bone_root.name = 'Root'
        bone_root.head = (0., 0., 0.)
        bone_root.tail = (bones[0, 0], bones[0, 1], bones[0, 2])
    
    J = len(names)
    def extrude_bone(
        name: Union[None, str],
        parent_name: Union[None, str],
        head: Tuple[float, float, float],
        tail: Tuple[float, float, float],
    ):
        bone = edit_bones.new(name)
        bone.head = (head[0], head[1], head[2])
        bone.tail = (tail[0], tail[1], tail[2])
        bone.name = name
        if parent_name is None:
            return
        parent_bone = edit_bones.get(parent_name)
        bone.parent = parent_bone
        bone.use_connect = False # always False currently

    vertices, bones = get_correct_orientation_kdtree(vertices, mesh_vertices, bones)
    inv = np.linalg.inv(local_coord)
    bones[:, :3] = (inv[:3, :3] @ bones[:, :3].T + inv[:3, 3:4]).T
    bones[:, 3:] = (inv[:3, :3] @ bones[:, 3:].T + inv[:3, 3:4]).T
    for i in range(J):
        if add_root:
            pname = 'Root' if parents[i] is None else names[parents[i]]
        else:
            pname = None if parents[i] is None else names[parents[i]]
        extrude_bone(names[i], pname, bones[i, :3], bones[i, 3:])

    # must set to object mode to enable parent_set
    bpy.ops.object.mode_set(mode='OBJECT')
    objects = bpy.data.objects
    for o in bpy.context.selected_objects:
        o.select_set(False)
    
    argsorted = np.argsort(-skin, axis=1)
    vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]
    vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[...,None]
    vertex_group_reweight = np.nan_to_num(vertex_group_reweight)
    tree = cKDTree(vertices)
    for ob in objects:
        if ob.type != 'MESH':
            continue
        ob.select_set(True)
        armature.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')
        vis = []
        for x in ob.vertex_groups:
            vis.append(x.name)
        
        n_vertices = []
        m = local_coord @ np.array(ob.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        for v in ob.data.vertices:
            n_vertices.append(matrix_world_rot @ np.array(v.co) + matrix_world_bias)
        n_vertices = np.stack(n_vertices)

        _, index = tree.query(n_vertices)

        for v, co in enumerate(tqdm(n_vertices)):
            for ii in range(group_per_vertex):
                i = argsorted[index[v], ii]
                if i >= len(names):
                    continue
                n = names[i]
                if n not in ob.vertex_groups:
                    continue
                        
                ob.vertex_groups[n].add([v], vertex_group_reweight[index[v], ii], 'REPLACE')
        armature.select_set(False)
        ob.select_set(False)
    armature.parent = local_parent
    
    # set vrm bones link
    if is_vrm:
        armature.data.vrm_addon_extension.spec_version = "1.0"
        humanoid.human_bones.hips.node.bone_name = "J_Bip_C_Hips"
        humanoid.human_bones.spine.node.bone_name = "J_Bip_C_Spine"
        
        humanoid.human_bones.chest.node.bone_name = "J_Bip_C_Chest"
        humanoid.human_bones.neck.node.bone_name = "J_Bip_C_Neck"
        humanoid.human_bones.head.node.bone_name = "J_Bip_C_Head"
        humanoid.human_bones.left_upper_leg.node.bone_name = "J_Bip_L_UpperLeg"
        humanoid.human_bones.left_lower_leg.node.bone_name = "J_Bip_L_LowerLeg"
        humanoid.human_bones.left_foot.node.bone_name = "J_Bip_L_Foot"
        humanoid.human_bones.right_upper_leg.node.bone_name = "J_Bip_R_UpperLeg"
        humanoid.human_bones.right_lower_leg.node.bone_name = "J_Bip_R_LowerLeg"
        humanoid.human_bones.right_foot.node.bone_name = "J_Bip_R_Foot"
        humanoid.human_bones.left_upper_arm.node.bone_name = "J_Bip_L_UpperArm"
        humanoid.human_bones.left_lower_arm.node.bone_name = "J_Bip_L_LowerArm"
        humanoid.human_bones.left_hand.node.bone_name = "J_Bip_L_Hand"
        humanoid.human_bones.right_upper_arm.node.bone_name = "J_Bip_R_UpperArm"
        humanoid.human_bones.right_lower_arm.node.bone_name = "J_Bip_R_LowerArm"
        humanoid.human_bones.right_hand.node.bone_name = "J_Bip_R_Hand"
        
        bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(armature_name="Armature")

def merge(
    path: str,
    output_path: str,
    vertices: ndarray,
    joints: ndarray,
    skin: ndarray,
    parents: List[Union[None, int]],
    names: List[str],
    tails: ndarray,
    add_root: bool=False,
    is_vrm: bool=False,
):
    '''
    Merge skin and bone into original file.
    '''
    clean_bpy()
    try:
        load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    
    bones = np.concatenate([joints, tails], axis=1)
    # if the result is weired, orientation may be wrong
    make_armature(
        vertices=vertices,
        bones=bones,
        parents=parents,
        names=names,
        skin=skin,
        group_per_vertex=4,
        add_root=add_root,
        is_vrm=is_vrm,
    )
    
    dirpath = os.path.dirname(output_path)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)
    try:
        if is_vrm:
            bpy.ops.export_scene.vrm(filepath=output_path)
        elif output_path.endswith(".fbx") or output_path.endswith(".FBX"):
            bpy.ops.export_scene.fbx(filepath=output_path, add_leaf_bones=True)
        elif output_path.endswith(".glb") or output_path.endswith(".gltf"):
            bpy.ops.export_scene.gltf(filepath=output_path)
        elif output_path.endswith(".dae"):
            bpy.ops.wm.collada_export(filepath=output_path)
        elif output_path.endswith(".blend"):
            with bpy.data.libraries.load(output_path) as (data_from, data_to):
                data_to.objects = data_from.objects
        else:
            raise ValueError(f"not suported type {output_path}")
    except:
        raise ValueError(f"failed to export {output_path}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def nullable_string(val):
    if not val:
        return None
    return val

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require_suffix', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--data_config', type=str, required=False)
    parser.add_argument('--skeleton_config', type=str, required=False)
    parser.add_argument('--skin_config', type=str, required=False)
    parser.add_argument('--merge_dir', type=str, required=False)
    parser.add_argument('--merge_name', type=str, required=False)
    parser.add_argument('--add_root', type=str2bool, required=False, default=False)
    parser.add_argument('--source', type=nullable_string, required=False, default=None)
    parser.add_argument('--target', type=nullable_string, required=False, default=None)
    parser.add_argument('--output', type=nullable_string, required=False, default=None)
    return parser.parse_args()

def transfer(source: str, target: str, output: str, add_root: bool=False):
    clean_bpy()
    try:
        armature = load(filepath=source, return_armature=True)
        assert armature is not None
    except Exception as e:
        print(f"failed to load {source}")
        return

    vertices, faces, skin = process_mesh()
    arranged_bones = get_arranged_bones(armature)
    if skin is None:
        skin = get_skin(arranged_bones)

    joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)
    merge(
        path=target,
        output_path=output,
        vertices=vertices,
        joints=joints,
        skin=skin,
        parents=parents,
        names=names,
        tails=tails,
        add_root=add_root,
    )

if __name__ == "__main__":
    args = parse()
    
    if args.source is not None or args.target is not None:
        assert args.source is not None and args.target is not None
        transfer(args.source, args.target, args.output, args.add_root)
        exit()

    data_config     = Box(yaml.safe_load(open(args.data_config, "r")))
    skeleton_config = Box(yaml.safe_load(open(args.skeleton_config, "r")))
    skin_config     = Box(yaml.safe_load(open(args.skin_config, "r")))

    num_runs        = args.num_runs
    id              = args.id
    require_suffix  = args.require_suffix.split(',')
    merge_dir       = args.merge_dir
    merge_name      = args.merge_name
    add_root        = args.add_root

    input_dataset_dir   = data_config.input_dataset_dir
    dataset_name        = data_config.output_dataset_dir
    
    skin_output_dataset_dir = skin_config.writer.output_dir
    skin_name               = skin_config.writer.export_npz
    
    skeleton_output_dataset_dir = skeleton_config.writer.output_dir
    skeleton_name               = skeleton_config.writer.export_npz

    def make_path(output_dataset_dir, dataset_name, root, file_name):
        if output_dataset_dir is None:
            return os.path.join(
                dataset_name,
                os.path.relpath(root, input_dataset_dir),
                file_name,
            )
        return os.path.join(
            output_dataset_dir,
            dataset_name,
            os.path.relpath(root, input_dataset_dir),
            file_name,
        )

    files = []
    for root, dirs, f in os.walk(input_dataset_dir):
        for file in f:
            if file.split('.')[-1] in require_suffix:
                file_name = file.removeprefix("./")
                suffix = file.split('.')[-1]
                # remove suffix
                file_name = '.'.join(file_name.split('.')[:-1])
                
                skin_path = make_path(skin_output_dataset_dir, dataset_name, root, os.path.join(file_name, skin_name+'.npz'))
                skeleton_path = make_path(skeleton_output_dataset_dir, dataset_name, root, os.path.join(file_name, skeleton_name+'.npz'))
                merge_path = make_path(merge_dir, dataset_name, root, os.path.join(file_name, merge_name+"."+suffix))
                
                # check if inference result exists
                if os.path.exists(skin_path) and os.path.exists(skeleton_path):
                    files.append((os.path.join(root, file), skin_path, skeleton_path, merge_path))

    num_files = len(files)
    print("num_files", num_files)
    gap = num_files // num_runs
    start = gap * id
    end = gap * (id + 1)
    if id+1==num_runs:
        end = num_files
    
    files = sorted(files)
    if end!=-1:
        files = files[:end]
    tot = 0
    for file in tqdm(files[start:]):
        origin_file = file[0]
        skin_path = file[1]
        skeleton_path = file[2]
        merge_file = file[3]
        
        raw_skin = RawSkin.load(path=skin_path)
        raw_data = RawData.load(path=skeleton_path)
        
        try:
            merge(
                path=origin_file,
                output_path=merge_file,
                vertices=raw_skin.vertices,
                joints=raw_skin.joints,
                skin=raw_skin.skin,
                parents=raw_data.parents,
                names=raw_data.names,
                tails=raw_data.tails,
                add_root=add_root,
                is_vrm=(raw_data.cls=='vroid'),
            )
        except Exception as e:
            print(f"failed to merge {origin_file}: {e}")