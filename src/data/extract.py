import bpy, os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from typing import Dict, Tuple, List, Optional, Union
import trimesh
import fast_simplification
from scipy.spatial import KDTree

import argparse
import yaml
from box import Box
import os

from .log import new_entry, add_error, add_warning, new_log, end_log
from .raw_data import RawData

def load(filepath: str):
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
            # end bone is removed using remove_dummy_bone
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

# remove all data in bpy
def clean_bpy():
    # First try to purge orphan data
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception as e:
        print(f"Warning: Could not purge orphans: {e}")
        
    # Then remove all data by type
    data_types = [
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.cameras,
        bpy.data.collections,
        bpy.data.curves,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.objects,
        bpy.data.textures,
        bpy.data.worlds,
        bpy.data.node_groups
    ]
    
    for data_collection in data_types:
        try:
            for item in data_collection:
                try:
                    data_collection.remove(item)
                except Exception as e:
                    print(f"Warning: Could not remove {item.name} from {data_collection}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {data_collection}: {e}")
            
    # Force garbage collection to free memory
    import gc
    gc.collect()

def get_arranged_bones(armature):
    matrix_world = armature.matrix_world
    arranged_bones = []
    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent
    Q = [root]
    rot = np.array(matrix_world)[:3, :3]
    
    # dfs and sort
    while len(Q) != 0:
        b = Q.pop(0)
        arranged_bones.append(b)
        children = []
        for cb in b.children:
            head = rot @ np.array(b.head)
            children.append((cb, head[0], head[1], head[2]))
        children = sorted(children, key=lambda x: (x[3], x[1], x[2]))
        _c = [x[0] for x in children]
        Q = _c + Q
    return arranged_bones

def process_mesh(arranged_bones=None):
    meshes = []
    for v in bpy.data.objects:
        if v.type == 'MESH':
            meshes.append(v)
    
    if arranged_bones is not None:
        index = {}
        # update index first
        for (id, pbone) in enumerate(arranged_bones):
            index[pbone.name] = id
    
    _dict_mesh = {}
    _dict_skin = {}
    if arranged_bones is not None:
        total_bones = len(arranged_bones)
    else:
        total_bones = None
    for obj in meshes:
        m = np.array(obj.matrix_world)
        matrix_world_rot = m[:3, :3]
        matrix_world_bias = m[:3, 3]
        rot = matrix_world_rot
        total_vertices = len(obj.data.vertices)
        vertex = np.zeros((4, total_vertices))
        vertex_normal = np.zeros((total_vertices, 3))
        if total_bones is not None:
            skin_weight = np.zeros((total_vertices, total_bones))
        else:
            skin_weight = None
        obj_verts = obj.data.vertices
        faces = []
        normals = []
        
        for v in obj_verts:
            vertex_normal[v.index] = rot @ np.array(v.normal) # be careful !
            vv = rot @ v.co
            vv = np.array(vv) + matrix_world_bias
            vertex[0:3, v.index] = vv
            vertex[3][v.index] = 1 # affine coordinate
        
        for polygon in obj.data.polygons:
            edges = polygon.edge_keys
            nodes = []
            adj = {}
            for edge in edges:
                if adj.get(edge[0]) is None:
                    adj[edge[0]] = []
                adj[edge[0]].append(edge[1])
                if adj.get(edge[1]) is None:
                    adj[edge[1]] = []
                adj[edge[1]].append(edge[0])
                nodes.append(edge[0])
                nodes.append(edge[1])
            normal = polygon.normal
            nodes = list(set(sorted(nodes)))
            first = nodes[0]
            loop = []
            now = first
            vis = {}
            while True:
                loop.append(now)
                vis[now] = True
                if vis.get(adj[now][0]) is None:
                    now = adj[now][0]
                elif vis.get(adj[now][1]) is None:
                    now = adj[now][1]
                else:
                    break
            for (second, third) in zip(loop[1:], loop[2:]):
                faces.append((first + 1, second + 1, third + 1)) # the cursed +1
                normals.append(rot @ normal) # and the cursed normal of BLENDER
        
        obj_group_names = [g.name for g in obj.vertex_groups]
        if arranged_bones is not None:
            for bone in arranged_bones:
                if bone.name not in obj_group_names:
                    continue
                gidx = obj.vertex_groups[bone.name].index
                bone_verts = [v for v in obj_verts if gidx in [g.group for g in v.groups]]
                for v in bone_verts:
                    which = [id for id in range(len(v.groups)) if v.groups[id].group==gidx]
                    w = v.groups[which[0]].weight
                    assert(0 <= v.index < total_vertices)
                    vv = rot @ v.co
                    vv = np.array(vv) + matrix_world_bias
                    vertex[0:3, v.index] = vv
                    vertex[3][v.index] = 1 # affine coordinate
                    skin_weight[v.index, index[bone.name]] = w
        
        correct_faces = []
        for (i, face) in enumerate(faces):
            normal = normals[i]
            v0 = face[0] - 1
            v1 = face[1] - 1
            v2 = face[2] - 1
            v = np.cross(
                vertex[:3, v1] - vertex[:3, v0],
                vertex[:3, v2] - vertex[:3, v0],
            )
            if (v*normal).sum() > 0:
                correct_faces.append(face)
            else:
                correct_faces.append((face[0], face[2], face[1]))
        if len(correct_faces) > 0:
            _dict_mesh[obj.name] = {
                'vertex': vertex,
                'face': correct_faces,
            }
            if skin_weight is not None:
                _dict_skin[obj.name] = {
                    'skin': skin_weight,
                }
    
    vertex = np.concatenate([_dict_mesh[name]['vertex'] for name in _dict_mesh], axis=1)[:3, :].transpose()
    
    total_faces = 0
    now_bias = 0
    for name in _dict_mesh:
        total_faces += len(_dict_mesh[name]['face'])
    faces = np.zeros((total_faces, 3), dtype=np.int64)
    tot = 0
    for name in _dict_mesh:
        f = np.array(_dict_mesh[name]['face'], dtype=np.int64)
        faces[tot:tot+f.shape[0]] = f + now_bias
        now_bias += _dict_mesh[name]['vertex'].shape[1]
        tot += f.shape[0]

    skin = None
    if arranged_bones is not None and len(_dict_skin) > 0:
        skin = np.concatenate([
            _dict_skin[d]['skin'] for d in _dict_skin
        ], axis=0)

    return vertex, faces, skin

def process_armature(
    armature,
    arranged_bones,
) -> Tuple[np.ndarray, np.ndarray]:
    matrix_world = armature.matrix_world
    index = {}

    for (id, pbone) in enumerate(arranged_bones):
        index[pbone.name] = id
    
    root = armature.pose.bones[0]
    while root.parent is not None:
        root = root.parent
    m = np.array(matrix_world.to_4x4())
    scale_inv = np.linalg.inv(np.diag(matrix_world.to_scale()))
    rot = m[:3, :3]
    bias = m[:3, 3]
    
    s = []
    bpy.ops.object.editmode_toggle()
    edit_bones = armature.data.edit_bones
    
    J = len(arranged_bones)
    joints = np.zeros((J, 3), dtype=np.float32)
    tails = np.zeros((J, 3), dtype=np.float32)
    parents = []
    name_to_id = {}
    names = []
    matrix_local_stack = np.zeros((J, 4, 4), dtype=np.float32)
    for (id, pbone) in enumerate(arranged_bones):
        name = pbone.name
        names.append(name)
        matrix_local = np.array(pbone.bone.matrix_local)
        use_inherit_rotation = pbone.bone.use_inherit_rotation
        if use_inherit_rotation == False:
            add_warning(f"use_inherit_rotation of bone {name} is False !")
        head = rot @ matrix_local[0:3, 3] + bias
        s.append(head)
        edit_bone = edit_bones.get(name)
        tail = rot @ np.array(edit_bone.tail) + bias
        
        name_to_id[name] = id
        joints[id] = head
        tails[id] = tail
        parents.append(None if pbone.parent not in arranged_bones else name_to_id[pbone.parent.name])
        # remove scale part
        matrix_local[:, 3:4] = m @ matrix_local[:, 3:4]
        matrix_local[:3, :3] = scale_inv @ matrix_local[:3, :3]
        matrix_local_stack[id] = matrix_local
    bpy.ops.object.editmode_toggle()
    
    return joints, tails, parents, names, matrix_local_stack

def save_raw_data(
    path: str,
    vertices: ndarray,
    faces: ndarray,
    skin: Union[ndarray, None],
    joints: Union[ndarray, None],
    tails: Union[ndarray, None],
    parents: Union[List[Union[int, None]], None],
    names: Union[List[str], None],
    matrix_local: Union[ndarray, None],
    target_count: int,
):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)
    if faces.shape[0] > target_count:
        vertices, faces = fast_simplification.simplify(vertices, faces, target_count=target_count)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    new_vertices = np.array(mesh.vertices, dtype=np.float32)
    new_vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
    new_faces = np.array(mesh.faces, dtype=np.int64)
    new_face_normals = np.array(mesh.face_normals, dtype=np.float32)
    if joints is not None:
        new_joints = np.array(joints, dtype=np.float32)
    else:
        new_joints = None
    if skin is not None:
        new_skin = np.array(skin, dtype=np.float32)
        # sample nearest
        tree = KDTree(vertices)
        distances, indices = tree.query(new_vertices)
        new_skin = new_skin[indices]
    else:
        new_skin = None
    
    raw_data = RawData(
        vertices=new_vertices,
        vertex_normals=new_vertex_normals,
        faces=new_faces,
        face_normals=new_face_normals,
        joints=new_joints,
        tails=tails,
        skin=new_skin,
        no_skin=None,
        parents=parents,
        names=names,
        matrix_local=matrix_local,
    )
    raw_data.check()
    raw_data.save(path=path)

def extract_builtin(
    output_folder: str,
    target_count: int,
    num_runs: int,
    id: int,
    time: str,
    files: List[Union[str, str]],
):
    log_path = "./logs"
    log_path = os.path.join(log_path, time)

    num_files = len(files)
    gap = num_files // num_runs
    start = gap * id
    end = gap * (id + 1)
    if id+1==num_runs:
        end = num_files
    
    files = sorted(files)
    if end!=-1:
        files = files[:end]
    new_log(log_path, f"extract_builtin_{start}_{end}")
    tot = 0
    for file in tqdm(files[start:]):
        input_file = file[0]
        output_dir = file[1]
        clean_bpy()
        new_entry(input_file)
        try:
            print(f"Now processing {input_file}...")
            
            armature = load(input_file)
            
            print('save to:', output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            if armature is not None:
                arranged_bones = get_arranged_bones(armature)
            else:
                arranged_bones = None
            vertices, faces, skin = process_mesh(arranged_bones)
            
            if armature is not None:
                joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)
            else:
                joints = None
                tails = None
                parents = None
                names = None
                matrix_local = None
            
            save_file = os.path.join(output_dir, 'raw_data.npz')
            save_raw_data(
                path=save_file,
                vertices=vertices,
                faces=faces-1,
                skin=skin,
                joints=joints,
                tails=tails,
                parents=parents,
                names=names,
                matrix_local=matrix_local,
                target_count=target_count,
            )
            
            tot += 1

        except ValueError as e:
            add_error(str(e))
            print(f"ValueError: {str(e)}")
        except RuntimeError as e:
            add_error(str(e))
            print(f"RuntimeError: {str(e)}")
        except TimeoutError as e:
            add_error("time out")
            print("TimeoutError: Processing timed out")
        except Exception as e:
            add_error(f"Unexpected error: {str(e)}")
            print(f"Unexpected error: {str(e)}")
    end_log()
    print(f"{tot} models processed")

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

def get_files(
    data_name: str,
    input_dataset_dir: str,
    output_dataset_dir: str,
    inputs: Union[str, None]=None,
    require_suffix: List[str]=['obj','fbx','FBX','dae','glb','gltf','vrm'],
    force_override: bool=False,
    warning: bool=True,
) -> List[Tuple[str, str]]:
    
    files = [] # (input_file, output_dir)
    if inputs is not None: # specified input file(s)
        vis = {}
        inputs = inputs.split(',')
        for file in inputs:
            file_name = file.removeprefix("./")
            # remove suffix
            file_name = '.'.join(file_name.split('.')[:-1])
            output_dir = os.path.join(output_dataset_dir, file_name)
            raw_data_npz = os.path.join(output_dir, data_name)
            if not force_override and os.path.exists(raw_data_npz):
                continue
            if warning and output_dir in vis:
                print(f"\033[33mWARNING: duplicate output directory: {output_dir}, you need to rename prefix of files to avoid ambiguity\033[0m")
            vis[output_dir] = True
            files.append((file, output_dir))
    else:
        vis = {}
        for root, dirs, f in os.walk(input_dataset_dir):
            for file in f:
                if file.split('.')[-1] in require_suffix:
                    file_name = file.removeprefix("./")
                    # remove suffix
                    file_name = '.'.join(file_name.split('.')[:-1])
                    
                    output_dir = os.path.join(output_dataset_dir, os.path.relpath(root, input_dataset_dir), file_name)
                    raw_data_npz = os.path.join(output_dir, data_name)
                    
                    # Check if all required files exist
                    if not force_override and os.path.exists(raw_data_npz):
                        continue
                    if warning and output_dir in vis:
                        print(f"\033[33mWARNING: duplicate output directory: {output_dir}, you need to rename prefix of files to avoid ambiguity\033[0m")
                    vis[output_dir] = True
                    files.append((os.path.join(root, file), output_dir))

    return files

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--require_suffix', type=str, required=True)
    parser.add_argument('--faces_target_count', type=int, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--force_override', type=str2bool, required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--time', type=str, required=True)

    parser.add_argument('--input', type=nullable_string, required=False, default=None)
    parser.add_argument('--input_dir', type=nullable_string, required=False, default=None)
    parser.add_argument('--output_dir', type=nullable_string, required=False, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    config = Box(yaml.safe_load(open(args.config, "r")))
    
    num_runs        = args.num_runs
    id              = args.id
    timestamp       = args.time
    require_suffix  = args.require_suffix.split(',')
    force_override  = args.force_override
    target_count    = args.faces_target_count
    
    if args.input_dir:
        config.input_dataset_dir = args.input_dir
    if args.output_dir:
        config.output_dataset_dir = args.output_dir
    
    assert config.input_dataset_dir is not None or args.input is None, 'you cannot specify both input and input_dir'

    files = get_files(
        data_name='raw_data.npz',
        inputs=args.input,
        input_dataset_dir=config.input_dataset_dir,
        output_dataset_dir=config.output_dataset_dir,
        require_suffix=require_suffix,
        force_override=force_override,
        warning=True,
    )

    extract_builtin(
        output_folder=config.output_dataset_dir,
        target_count=target_count,
        num_runs=num_runs,
        id=id,
        time=timestamp,
        files=files,
    )