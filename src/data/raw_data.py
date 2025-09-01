from dataclasses import dataclass
import numpy as np
from numpy import ndarray

import os
from typing import Union, List, Tuple

from .exporter import Exporter

from ..tokenizer.spec import DetokenizeOutput
from .order import Order

@dataclass(frozen=True)
class RawData(Exporter):
    '''
    Dataclass to handle data from processed model files.
    '''
    
    # vertices of the mesh, shape (N, 3)
    vertices: Union[ndarray, None]
    
    # normals of vertices, shape (N, 3)
    vertex_normals: Union[ndarray, None]
    
    # faces of mesh, shape (F, 3), face id starts from 0 to F-1
    faces: Union[ndarray, None]
    
    # face normal of mesh, shape (F, 3)
    face_normals: Union[ndarray, None]
    
    # joints of bones, shape (J, 3)
    joints: Union[ndarray, None]
    
    # skinning of joints, shape (N, J)
    skin: Union[ndarray, None]
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None]
    
    # names of joints
    names: Union[List[str], None]
    
    # local coordinate
    matrix_local: Union[ndarray, None]
    
    # tails of joints, shape (J, 3)
    tails: Union[ndarray, None]=None
    
    # whether the joint has skin, bool
    no_skin: Union[ndarray, None]=None
    
    # path to data
    path: Union[str, None]=None
    
    # data cls
    cls: Union[str, None]=None
    
    @staticmethod
    def load(path: str, origin=np.float16, to=np.float32) -> 'RawData':
        data = np.load(path, allow_pickle=True)
        d = {name: data[name][()] for name in data}
        d['path'] = path
        skin = d.get('skin', None)
        if skin is not None:
            d['no_skin'] = ~np.any(skin>0, axis=0)
        else:
            d['no_skin'] = None
        return RawData(**d).change_dtype(origin, to)
    
    def change_dtype(self, origin, to) -> 'RawData':
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ndarray) and v.dtype == origin:
                v = v.astype(to)
            d[k] = v
        return RawData(**d)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)
    
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
    
    def check(self):
        if self.names is not None and self.joints is not None:
            assert len(self.names) == self.J
        if self.names is not None and self.parents is not None:
            assert len(self.names) == len(self.parents)
        if self.parents is not None:
            for (i, pid) in enumerate(self.parents):
                if i==0:
                    assert pid is None
                else:
                    assert pid is not None
                    assert pid < i
    
    def export_pc(self, path: str, with_normal: bool=True, normal_size=0.01):
        '''
        export point cloud
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.vertex_normals, normal_size=normal_size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, normal_size=normal_size)
    
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
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        use_tail: bool=False,
        custom_vertex_group: Union[ndarray, None]=None,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin if custom_vertex_group is None else custom_vertex_group,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=self.tails if use_tail else None,
        )
    
    def export_render(self, path: str, resolution: Tuple[int, int]=[256, 256]):
        self._export_render(
            path=path,
            vertices=self.vertices,
            faces=self.faces,
            bones=np.concatenate([self.joints, self.tails], axis=-1),
            resolution=resolution,
        )

@dataclass(frozen=True)
class RawSkeleton(Exporter):
    '''
    Dataclass to handle skeleton from AR.
    '''
    # joints of bones, shape (J, 3), float32
    joints: Union[ndarray, None]
    
    # tails of joints, shape (J, 3), float32
    tails: Union[ndarray, None]
    
    # whether the joint has skin, bool
    no_skin: Union[ndarray, None]
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None]
    
    # names of joints
    names: Union[List[str], None]
    
    @staticmethod
    def load(path: str) -> 'RawSkeleton':
        data = np.load(path, allow_pickle=True)
        return RawSkeleton(**{name: data[name][()] for name in data})
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)
    
    @staticmethod
    def from_detokenize_output(res: DetokenizeOutput, order: Union[Order, None]) -> 'RawSkeleton':
        J = len(res.bones)
        names = order.make_names(cls=res.cls, parts=res.parts, num_bones=J)
        joints = res.joints
        p_joints = res.p_joints
        parents = []
        for (i, joint) in enumerate(joints):
            if i == 0:
                parents.append(None)
                continue
            p_joint = p_joints[i]
            dis = 999999
            pid = None
            for j in reversed(range(i)):
                n_dis = ((joints[j] - p_joint)**2).sum()
                if n_dis < dis:
                    pid = j
                    dis = n_dis
            parents.append(pid)
        return RawSkeleton(
            joints=joints,
            tails=res.tails,
            no_skin=res.no_skin,
            parents=parents,
            names=names,
        )
        
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
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        use_tail: bool=False,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=None,
            joints=self.joints,
            skin=None,
            parents=self.parents,
            names=self.names,
            faces=None,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=self.tails if use_tail else None,
        )
    
    def export_render(self, path: str, resolution: Tuple[int, int]=[256, 256]):
        self._export_render(
            path=path,
            vertices=None,
            faces=None,
            bones=np.concatenate([self.joints, self.tails], axis=-1),
            resolution=resolution,
        )

@dataclass
class RawSkin(Exporter):
    '''
    Dataclass to handle skeleton from AR.
    '''
    # skin, shape (J, N)
    skin: ndarray
    
    # always sampled, shape (N, 3)
    vertices: Union[ndarray, None]=None
    
    # for future use, shape (J, 3)
    joints: Union[ndarray, None]=None
    
    @staticmethod
    def load(path: str) -> 'RawSkin':
        data = np.load(path, allow_pickle=True)
        return RawSkin(**{name: data[name][()] for name in data})
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)