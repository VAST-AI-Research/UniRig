from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray

from typing import Dict, Union, List, Tuple

from .order import Order
from .raw_data import RawData
from .exporter import Exporter

from ..tokenizer.spec import TokenizeInput
from .utils import linear_blend_skinning

import trimesh


@dataclass
class Asset(Exporter):
    '''
    Dataclass to handle data parsed from raw data.
    '''
    
    # data class
    cls: str
    
    # where is this asset from
    path: str
    
    # data file name
    data_name: str
    
    # vertices of the mesh, shape (N, 3), float32
    vertices: ndarray
    
    # normals of vertices, shape (N, 3), float32
    vertex_normals: ndarray
    
    # faces of mesh, shape (F, 3), face id starts from 0 to F-1, int64
    faces: ndarray
    
    # face normal of mesh, shape (F, 3), float32
    face_normals: ndarray
    
    # joints of bones, shape (J, 3), float32
    joints: Union[ndarray, None]=None
    
    # tails of joints, shape (J, 3), float32
    tails: Union[ndarray, None]=None
    
    # skinning of joints, shape (N, J), float32
    skin: Union[ndarray, None]=None
    
    # whether the joint has skin, bool
    no_skin: Union[ndarray, None]=None
    
    # vertex groups
    vertex_groups: Union[Dict[str, ndarray], None]=None
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None]=None
    
    # names of joints
    names: Union[List[str], None]=None
    
    # sampled vertices, shape (N, 3)
    sampled_vertices: Union[ndarray, None]=None
    
    # sampled normals, shape (N, 3)
    sampled_normals: Union[ndarray, None]=None
    
    # sampled vertex groups, every vertex group should be (N, J)
    sampled_vertex_groups: Union[Dict[str, ndarray], None]=None
    
    # {id: part}, part==None -> a spring token
    parts_bias: Union[Dict[int, Union[str, None]], None]=None
    
    # local coordinate, shape (J, 4, 4)
    matrix_local: Union[ndarray, None]=None
    
    # pose matrix for skinning loss calculation, shape (J, 4, 4)
    pose_matrix: Union[ndarray, None]=None
    
    meta: Union[Dict[str, ...], None]=None
    
    @property
    def N(self):
        '''
        number of vertices
        '''
        return self.vertices.shape[0]
    
    @property
    def F(self):
        '''
        number of faces
        '''
        return self.faces.shape[0]
    
    @property
    def J(self):
        '''
        number of joints
        '''
        return self.joints.shape[0]
    
    def get_matrix(self, matrix_basis: ndarray, matrix_local: Union[ndarray, None]=None):
        '''
        get matrix
        
        matrix_basis: (J, 4, 4)
        '''
        if matrix_local is None:
            assert self.joints is not None
            matrix_local = self.matrix_local
            if matrix_local is None:
                matrix_local = np.zeros((self.J, 4, 4))
                matrix_local[:, 0, 0] = 1.
                matrix_local[:, 1, 1] = 1.
                matrix_local[:, 2, 2] = 1.
                matrix_local[:, 3, 3] = 1.
                for i in range(self.J):
                    matrix_local[i, :3, 3] = self.joints[i]
        
        matrix = np.zeros((self.J, 4, 4))
        for i in range(self.J):
            if i==0:
                matrix[i] = matrix_local[i] @ matrix_basis[i]
            else:
                pid = self.parents[i]
                matrix_parent = matrix[pid]
                matrix_local_parent = matrix_local[pid]
                
                matrix[i] = (
                    matrix_parent @
                    (np.linalg.inv(matrix_local_parent) @ matrix_local[i]) @
                    matrix_basis[i]
                )
        return matrix
    
    def apply_matrix_basis(self, matrix_basis: ndarray):
        '''
        apply a pose to armature
        
        matrix_basis: (J, 4, 4)
        '''
        matrix_local = self.matrix_local
        if matrix_local is None:
            matrix_local = np.zeros((self.J, 4, 4))
            matrix_local[:, 0, 0] = 1.
            matrix_local[:, 1, 1] = 1.
            matrix_local[:, 2, 2] = 1.
            matrix_local[:, 3, 3] = 1.
            for i in range(self.J):
                matrix_local[i, :3, 3] = self.joints[i].copy()
        
        matrix = self.get_matrix(matrix_basis=matrix_basis, matrix_local=matrix_local)
        self.joints = matrix[:, :3, 3].copy()
        vertices = linear_blend_skinning(self.vertices, matrix_local, matrix, self.skin, pad=1, value=1.)
        # update matrix_local
        self.matrix_local = matrix.copy()

        # change tails
        if self.tails is not None:
            t_skin = np.eye(self.J)
            self.tails = linear_blend_skinning(self.tails, matrix_local, matrix, t_skin, pad=1, value=1.)
        # in accordance with trimesh's normals
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        self.vertices = vertices
        self.vertex_normals = mesh.vertex_normals.copy()
        self.face_normals = mesh.face_normals.copy()
    
    def set_order_by_names(self, new_names: List[str]):
        assert len(new_names) == len(self.names)
        name_to_id = {name: id for (id, name) in enumerate(self.names)}
        new_name_to_id = {name: id for (id, name) in enumerate(new_names)}
        perm = []
        new_parents = []
        for (new_id, name) in enumerate(new_names):
            perm.append(name_to_id[name])
            pid = self.parents[name_to_id[name]]
            if new_id == 0:
                assert pid is None, 'first bone is not root bone'
            else:
                pname = self.names[pid]
                pid = new_name_to_id[pname]
                assert pid < new_id, 'new order does not form a tree'
            new_parents.append(pid)
        
        if self.joints is not None:
            self.joints = self.joints[perm]
        self.parents = new_parents
        if self.tails is not None:
            self.tails = self.tails[perm]
        if self.skin is not None:
            self.skin = self.skin[:, perm]
        if self.no_skin is not None:
            self.no_skin = self.no_skin[perm]
        if self.matrix_local is not None:
            self.matrix_local = self.matrix_local[perm]
        self.names = new_names
    
    def set_order(self, order: Order):
        if self.names is None or self.parents is None:
            return
        new_names, self.parts_bias = order.arrange_names(cls=self.cls, names=self.names, parents=self.parents)
        self.set_order_by_names(new_names=new_names)
    
    def collapse(self, keep: List[str]):
        dsu = [i for i in range(self.J)]
        
        def find(x: int) -> int:
            if dsu[x] == x:
                return x
            y = find(dsu[x])
            dsu[x] = y
            return y
        
        def merge(x: int, y: int):
            dsu[find(x)] = find(y)

        if self.tails is not None:
            new_tails = self.tails.copy()
        else:
            new_tails = None
        if self.skin is not None:
            new_skin = self.skin.copy()
        else:
            new_skin = None

        if self.no_skin is not None:
            new_no_skin = self.no_skin.copy()
        else:
            new_no_skin = None
        
        if self.matrix_local is not None:
            matrix_local = self.matrix_local.copy()
        else:
            matrix_local = None
        new_names = []
        new_parents = []
        perm = []
        new_name_to_id = {}
        tot = 0
        for (i, name) in enumerate(self.names):
            if name in keep:
                new_names.append(name)
                new_name_to_id[name] = tot
                tot += 1
                perm.append(i)
                pid = self.parents[i]
                if pid is None:
                    new_parents.append(None)
                else:
                    pid = find(pid)
                    new_parents.append(new_name_to_id[self.names[pid]])
                continue
            assert i != 0, 'cannot remove root'
            id = find(i)
            pid = find(self.parents[id])
            # be careful !
            # do not copy tail here because you dont know which child to inherit from
            if new_skin is not None:
                new_skin[:, pid] += new_skin[:, id]
            if new_no_skin is not None:
                new_no_skin[pid] &= new_no_skin[id]
            merge(id, pid)
        
        if new_tails is not None:
            new_tails = new_tails[perm]
        if new_skin is not None:
            new_skin = new_skin[:, perm]
        if new_no_skin is not None:
            new_no_skin = new_no_skin[perm]
        if matrix_local is not None:
            matrix_local = matrix_local[perm]
        
        if self.joints is not None:
            self.joints = self.joints[perm]
        self.parents        = new_parents
        self.tails          = new_tails
        self.skin           = new_skin
        self.no_skin        = new_no_skin
        self.names          = new_names
        self.matrix_local   = matrix_local
    
    def drop_part(self, keep: List[str]):
        assert self.skin is not None
        name_to_id = {n: i for (i, n) in enumerate(self.names)}
        mask = np.zeros(self.N, dtype=bool)
        for name in keep:
            if name in name_to_id:
                mask[self.skin[:, name_to_id[name]] > 1e-5] = True
        if len(keep) * 4 < len(self.names):
            return
        if np.all(~mask):
            return
        indices = np.where(mask)[0]
        mask = np.all(np.isin(self.faces, indices), axis=1)
        if np.all(~mask):
            return
        old_to_new = -np.ones(self.N, dtype=np.int64)
        self.vertices = self.vertices[indices]
        self.vertex_normals = self.vertex_normals[indices]
        new_faces = self.faces[mask]
        old_to_new[indices] = np.arange(len(indices))
        new_faces = old_to_new[new_faces]

        self.faces = new_faces
        if self.face_normals is not None:
            self.face_normals = self.face_normals[mask]
        self.skin = self.skin[indices]
        self.collapse(keep=keep)
    
    @staticmethod
    def from_raw_data(
        raw_data: RawData,
        cls: str,
        path: str,
        data_name: str,
    ) -> 'Asset':
        '''
        Return an asset initialized from raw data and do transform.
        '''
        return Asset(
            cls=cls,
            path=path,
            data_name=data_name,
            vertices=raw_data.vertices,
            vertex_normals=raw_data.vertex_normals,
            faces=raw_data.faces,
            face_normals=raw_data.face_normals,
            joints=raw_data.joints,
            tails=raw_data.tails,
            skin=raw_data.skin,
            no_skin=raw_data.no_skin,
            parents=raw_data.parents,
            names=raw_data.names,
            matrix_local=raw_data.matrix_local,
            meta={},
        )
    
    def get_tokenize_input(self) -> TokenizeInput:
        children = defaultdict(list)
        
        for (id, p) in enumerate(self.parents):
            if p is not None:
                children[p].append(id)
        bones = []
        branch = []
        is_leaf = []
        last = None
        for i in range(self.J):
            is_leaf.append(len(children[i])==0)
            if i == 0:
                bones.append(np.concatenate([self.joints[i], self.joints[i]]))
                branch.append(False)
            else:
                pid = self.parents[i]
                bones.append(np.concatenate([self.joints[pid], self.joints[i]]))
                branch.append(pid!=last)
            last = i
        bones = np.stack(bones)
        branch = np.array(branch, dtype=bool)
        is_leaf = np.array(is_leaf, dtype=bool)
        return TokenizeInput(
            bones=bones,
            tails=self.tails,
            branch=branch,
            is_leaf=is_leaf,
            no_skin=self.no_skin,
            cls=self.cls,
            parts_bias=self.parts_bias,
        )
    
    def export_pc(self, path: str, with_normal: bool=True, normal_size=0.01):
        '''
        export point cloud
        '''
        vertices    = self.vertices
        normals     = self.vertex_normals
        if self.sampled_vertices is not None:
            vertices    = self.sampled_vertices
            normals     = self.sampled_normals
        if with_normal == False:
            normals = None
        self._export_pc(vertices=vertices, path=path, vertex_normals=normals, normal_size=normal_size)
    
    def export_mesh(self, path: str):
        '''
        export mesh
        '''
        self._export_mesh(vertices=self.vertices, faces=self.faces, path=path)
    
    def export_skeleton(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_skeleton_sequence(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton_sequence(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        vertex_group_name: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        use_tail: bool=False,
        use_origin: bool=False,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices if use_origin else self.sampled_vertices,
            joints=self.joints,
            skin=self.sampled_vertex_groups[vertex_group_name],
            parents=self.parents,
            names=self.names,
            faces=self.faces if use_origin else None,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=self.tails if use_tail else None,
        )

    def export_render(self, path: str, resolution: Tuple[int, int]=[256, 256], use_tail: bool=False):
        if use_tail:
            assert self.tails is not None
            self._export_render(
                path=path,
                vertices=self.vertices,
                faces=self.faces,
                bones=np.concatenate([self.joints, self.tails], axis=-1),
                resolution=resolution,
            )
        else:
            pjoints = self.joints[self.parents[1:]]
            self._export_render(
                path=path,
                vertices=self.vertices,
                faces=self.faces,
                bones=np.concatenate([pjoints, self.joints[1:]], axis=-1),
                resolution=resolution,
            )