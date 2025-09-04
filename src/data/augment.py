from dataclasses import dataclass
from typing import Tuple, Union, List, Dict
from numpy import ndarray
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
import random
import trimesh

from .spec import ConfigSpec
from .asset import Asset
from .utils import axis_angle_to_matrix

@dataclass(frozen=True)
class AugmentDropPartConfig(ConfigSpec):
    # probability
    p: float
    
    # drop rate
    rate: float
    
    @classmethod
    def parse(cls, config) -> Union['AugmentDropPartConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentDropPartConfig(
            p=config.get('p', 0.),
            rate=config.get('rate', 0.5),
        )

@dataclass(frozen=True)
class AugmentJointDiscreteConfig(ConfigSpec):
    # perturb the skeleton with probability p
    p: float
    
    # num of discretized coord
    discrete: int
    
    # continuous range
    continuous_range: Tuple[float, float]
    
    @classmethod
    def parse(cls, config) -> Union['AugmentJointDiscreteConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentJointDiscreteConfig(
            p=config.get('p', 0.),
            discrete=config.get('discrete', 256),
            continuous_range=config.get('continuous_range', [-1., 1.]),
        )

@dataclass(frozen=True)
class AugmentJointPerturbConfig(ConfigSpec):
    # perturb the skeleton with probability p
    p: float
    
    # jitter sigma on joints
    sigma: float
    
    # jitter clip on joints
    clip: float
    
    @classmethod
    def parse(cls, config) -> Union['AugmentJointPerturbConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentJointPerturbConfig(
            p=config.get('p', 0.),
            sigma=config.get('sigma', 0.),
            clip=config.get('clip', 0.),
        )

@dataclass(frozen=True)
class AugmentCollapseConfig(ConfigSpec):
    # collapse the skeleton with probability p
    p: float
    
    # probability to merge the bone
    rate: float
    
    # max bones
    max_bones: int
    
    @classmethod
    def parse(cls, config) -> Union['AugmentCollapseConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentCollapseConfig(
            p=config.get('p', 0.),
            rate=config.get('rate', 0.),
            max_bones=config.get('max_bones', 2147483647),
        )

@dataclass(frozen=True)
class AugmentLBSConfig(ConfigSpec):
    # apply a random pose with probability p
    random_pose_p: float
    
    # random pose angle range
    random_pose_angle: float
    
    # if true, add an additional random pose matrix for skinning loss calculation
    pose_for_skin: bool
    
    @classmethod
    def parse(cls, config) -> Union['AugmentLBSConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentLBSConfig(
            random_pose_p=config.get('random_pose_p', 0.),
            random_pose_angle=config.get('random_pose_angle', 0.),
            pose_for_skin=config.get('pose_for_skin', False),
        )

@dataclass(frozen=True)
class AugmentLinearConfig(ConfigSpec):
    # apply random rotation with probability p
    random_rotate_p: float
    
    # random rotation angle(degree)
    random_rotate_angle: float
    
    # swap x with probability p
    random_flip_x_p: float
    
    # swap y with probability p
    random_flip_y_p: float
    
    # swap z with probability p
    random_flip_z_p: float
    
    # probability to pick an angle in static_rotate_x
    static_rotate_x_p: float
    
    # rotate around x axis among given angles(degrees)
    static_rotate_x: List[float]
    
    # probability to pick an angle in static_rotate_y
    static_rotate_y_p: float
    
    # rotate around y axis among given angles(degrees)
    static_rotate_y: List[float]
    
    # probability to pick an angle in static_rotate_z
    static_rotate_z_p: float
    
    # rotate around z axis among given angles(degrees)
    static_rotate_z: List[float]
    
    # apply random scaling with probability p
    random_scale_p: float
    
    # random scaling xyz axis
    random_scale: Tuple[float, float]
    
    @classmethod
    def parse(cls, config) -> Union['AugmentLinearConfig', None]:
        if config is None:
            return None
        if config.get('random_flip_x_p', 0) > 0 or config.get('random_flip_y_p', 0) > 0 or config.get('random_flip_z_p', 0) > 0:
            print("\033[31mWARNING: random flip is enabled and is very likely to confuse ar model !\033[0m")
        cls.check_keys(config)
        return AugmentLinearConfig(
            random_rotate_p=config.get('random_rotate_p', 0.),
            random_rotate_angle=config.get('random_rotate_angle', 0.),
            random_flip_x_p=config.get('random_flip_x_p', 0.),
            random_flip_y_p=config.get('random_flip_y_p', 0.),
            random_flip_z_p=config.get('random_flip_z_p', 0.),
            static_rotate_x_p=config.get('static_rotate_x_p', 0.),
            static_rotate_x=config.get('static_rotate_x', []),
            static_rotate_y_p=config.get('static_rotate_y_p', 0.),
            static_rotate_y=config.get('static_rotate_y', []),
            static_rotate_z_p=config.get('static_rotate_z_p', 0.),
            static_rotate_z=config.get('static_rotate_z', []),
            random_scale_p=config.get('random_scale_p', 0.),
            random_scale=config.get('random_scale', [1.0, 1.0]),
        )

@dataclass(frozen=True)
class AugmentAffineConfig(ConfigSpec):
    # final normalization cube
    normalize_into: Tuple[float, float]

    # randomly scale coordinates with probability p
    random_scale_p: float
    
    # scale range (lower, upper)
    random_scale: Tuple[float, float]
    
    # randomly shift coordinates with probability p
    random_shift_p: float
    
    # shift range (lower, upper)
    random_shift: Tuple[float, float]
    
    @classmethod
    def parse(cls, config) -> Union['AugmentAffineConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentAffineConfig(
            normalize_into=config.normalize_into,
            random_scale_p=config.get('random_scale_p', 0.),
            random_scale=config.get('random_scale', [1., 1.]),
            random_shift_p=config.get('random_shift_p', 0.),
            random_shift=config.get('random_shift', [0., 0.]),
        )

@dataclass(frozen=True)
class AugmentJitterConfig(ConfigSpec):
    # probability
    p: float
    
    # jitter sigma on vertices
    vertex_sigma: float
    
    # jitter clip on vertices
    vertex_clip: float
    
    # jitter sigma on normals
    normal_sigma: float
    
    # jitter clip on normals
    normal_clip: float
    
    @classmethod
    def parse(cls, config) -> Union['AugmentJitterConfig', None]:
        if config is None:
            return None
        cls.check_keys(config)
        return AugmentJitterConfig(
            p=config.get('p', 0.5),
            vertex_sigma=config.get('vertex_sigma', 0.),
            vertex_clip=config.get('vertex_clip', 0.),
            normal_sigma=config.get('normal_sigma', 0.),
            normal_clip=config.get('normal_clip', 0.),
        )

@dataclass(frozen=True)
class AugmentConfig(ConfigSpec):
    '''
    Config to handle final easy augmentation of vertices, normals and bones before sampling.
    '''    
    augment_collapse_config: Union[AugmentCollapseConfig, None]
    augment_lbs_config: Union[AugmentLBSConfig, None]
    augment_linear_config: Union[AugmentLinearConfig, None]
    augment_affine_config: Union[AugmentAffineConfig, None]
    augment_jitter_config: Union[AugmentJitterConfig, None]
    augment_joint_perturb_config: Union[AugmentJointPerturbConfig, None]
    augment_joint_discrete_config: Union[AugmentJointDiscreteConfig, None]
    augment_drop_part_config: Union[AugmentDropPartConfig, None]
    
    @classmethod
    def parse(cls, config) -> 'AugmentConfig':
        cls.check_keys(config)
        return AugmentConfig(
            augment_collapse_config=AugmentCollapseConfig.parse(config.get('augment_collapse_config', None)),
            augment_lbs_config=AugmentLBSConfig.parse(config.get('augment_lbs_config', None)),
            augment_linear_config=AugmentLinearConfig.parse(config.get('augment_linear_config', None)),
            augment_affine_config=AugmentAffineConfig.parse(config.get('augment_affine_config', None)),
            augment_jitter_config=AugmentJitterConfig.parse(config.get('augment_jitter_config', None)),
            augment_joint_perturb_config=AugmentJointPerturbConfig.parse(config.get('augment_joint_perturb_config', None)),
            augment_joint_discrete_config=AugmentJointDiscreteConfig.parse(config.get('augment_joint_discrete_config', None)),
            augment_drop_part_config=AugmentDropPartConfig.parse(config.get('augment_drop_part_config', None)),
        )

class Augment(ABC):
    '''
    Abstract class for augmentation
    '''
    def __init__(self):
        pass
    
    @abstractmethod
    def transform(self, asset: Asset, **kwargs):
        pass

    @abstractmethod
    def inverse(self, asset: Asset):
        pass

class AugmentDropPart(Augment):
    def __init__(self, config: AugmentDropPartConfig):
        super().__init__()
        self.config = config

    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() >= self.config.p:
            return
        names = []
        id_to_name = {}
        for (id, name) in enumerate(asset.names):
            id_to_name[id] = name
            if np.random.rand() < self.config.rate:
                names.append(name)
        if len(names) == 0:
            return
        for (i, p) in reversed(list(enumerate(asset.parents))):
            if p is None:
                continue
            if id_to_name[i] in names and id_to_name[p] not in names:
                names.append(id_to_name[p])
        asset.drop_part(keep=names)
    
    def inverse(self, asset: Asset):
        pass

class AugmentCollapse(Augment):
    def __init__(self, config: AugmentCollapseConfig):
        super().__init__()
        self.config = config

    def transform(self, asset: Asset, **kwargs):
        def select_k(arr: List, k: int):
            if len(arr) <= k:
                return arr
            else:
                rest_indices = list(range(1, len(arr)))
                selected_indices = sorted(random.sample(rest_indices, k - 1))
                return [arr[0]] + [arr[i] for i in selected_indices]

        if np.random.rand() < self.config.p:
            names = asset.names
            assert names is not None, f"do not find names while doing collapse"
            keep = [names[0]]
            for name in names[1:]:
                if np.random.rand() >= self.config.rate:
                    keep.append(name)
            asset.collapse(keep=select_k(keep, self.config.max_bones))
        elif asset.J > self.config.max_bones:
            keep = select_k(asset.names, self.config.max_bones)
            asset.collapse(keep=keep)
    
    def inverse(self, asset: Asset):
        pass

class AugmentJointDiscrete(Augment):
    def __init__(self, config: AugmentJointDiscreteConfig):
        super().__init__()
        self.config = config

    def _discretize(
        self,
        t: ndarray,
        continuous_range: Tuple[float, float],
        num_discrete: int,
    ) -> ndarray:
        lo, hi = continuous_range
        assert hi >= lo
        t = (t - lo) / (hi - lo)
        t *= num_discrete
        return np.clip(t.round(), 0, num_discrete - 1).astype(np.int64)

    def _undiscretize(
        self,
        t: ndarray,
        continuous_range: Tuple[float, float],
        num_discrete: int,
    ) -> ndarray:
        lo, hi = continuous_range
        assert hi >= lo
        t = t.astype(np.float32) + 0.5
        t /= num_discrete
        return t * (hi - lo) + lo

    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() < self.config.p:
            asset.joints = self._undiscretize(self._discretize(
                    asset.joints,
                    self.config.continuous_range,
                    self.config.discrete,
                ),
                self.config.continuous_range,
                self.config.discrete,)
            if asset.tails is not None:
                asset.tails = self._undiscretize(self._discretize(
                        asset.tails,
                        self.config.continuous_range,
                        self.config.discrete,
                    ),
                    self.config.continuous_range,
                    self.config.discrete,
                )
    
    def inverse(self, asset: Asset):
        pass

class AugmentJointPerturb(Augment):
    def __init__(self, config: AugmentJointPerturbConfig):
        super().__init__()
        self.config = config

    def transform(self, asset: Asset, **kwargs):
        if np.random.rand() < self.config.p:
            asset.joints += np.clip(
                np.random.normal(0, self.config.sigma, (asset.J, 3)),
                -self.config.clip,
                self.config.clip,
            )
            if asset.tails is not None:
                asset.tails += np.clip(
                    np.random.normal(0, self.config.sigma, (asset.J, 3)),
                    -self.config.clip,
                    self.config.clip,
                )
    
    def inverse(self, asset: Asset):
        pass

class AugmentLBS(Augment):
    def __init__(self, config: AugmentLBSConfig):
        super().__init__()
        self.config = config

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        def get_matrix_basis(angle: float):
            matrix = axis_angle_to_matrix((np.random.rand(asset.J, 3) - 0.5) * angle / 180 * np.pi * 2).astype(np.float32)
            return matrix
        
        if np.random.rand() < self.config.random_pose_p:
            matrix_basis = get_matrix_basis(self.config.random_pose_angle)
            asset.apply_matrix_basis(matrix_basis=matrix_basis)
        if self.config.pose_for_skin:
            matrix_basis = get_matrix_basis(self.config.random_pose_angle)
            pose_matrix = asset.get_matrix(matrix_basis=matrix_basis)
            asset.pose_matrix = pose_matrix
    
    def inverse(self, asset: Asset):
        pass

class AugmentLinear(Augment):
    
    def __init__(self, config: AugmentLinearConfig):
        super().__init__()
        self.config = config

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        r = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.config.random_rotate_p:
            angle = self.config.random_rotate_angle
            axis_angle = (np.random.rand(3) - 0.5) * angle / 180 * np.pi * 2
            r = R.from_rotvec(axis_angle).as_matrix()
            r = np.pad(r, ((0, 1), (0, 1)), 'constant', constant_values=0.)
            r[3, 3] = 1.
        
        if np.random.uniform(0, 1) < self.config.random_flip_x_p:
            r @= np.array([
                [-1.0, 0.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0, 1.0, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.random_flip_y_p:
            r @= np.array([
                [1.0,  0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0,  0.0, 1.0, 0.0],
                [0.0,  0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.random_flip_z_p:
            r @= np.array([
                [1.0, 0.0,  0.0, 0.0],
                [0.0, 1.0,  0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0,  0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.static_rotate_x_p:
            assert len(self.config.static_rotate_x) > 0, "static rotation of x is enabled, but static_rotate_x is empty"
            angle = np.random.choice(self.config.static_rotate_x) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [ 1.0, 0.0, 0.0, 0.0],
                [ 0.0,   c,   s, 0.0],
                [ 0.0,  -s,   c, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.static_rotate_y_p:
            assert len(self.config.static_rotate_y) > 0, "static rotation of y is enabled, but static_rotate_y is empty"
            angle = np.random.choice(self.config.static_rotate_y) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [   c, 0.0,  -s, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [   s, 0.0,   c, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.static_rotate_z_p:
            assert len(self.config.static_rotate_z) > 0, "static rotation of z is enabled, but static_rotate_z is empty"
            angle = np.random.choice(self.config.static_rotate_z) / 180 * np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            r @= np.array([
                [   c,   s, 0.0, 0.0],
                [  -s,   c, 0.0, 0.0],
                [ 0.0, 0.0, 1.0, 0.0],
                [ 0.0, 0.0, 0.0, 1.0],
            ])
        
        if np.random.uniform(0, 1) < self.config.random_scale_p:
            scale_x = np.random.uniform(self.config.random_scale[0], self.config.random_scale[1])
            scale_y = np.random.uniform(self.config.random_scale[0], self.config.random_scale[1])
            scale_z = np.random.uniform(self.config.random_scale[0], self.config.random_scale[1])
            r @= np.array([
                [scale_x, 0.0, 0.0, 0.0],
                [0.0, scale_y, 0.0, 0.0],
                [0.0, 0.0, scale_z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        
        trans_vertex = r @ trans_vertex
        
        # apply transform here
        self.trans_vertex = trans_vertex
        asset.vertices = self._apply(asset.vertices, self.trans_vertex)
        
        mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
        asset.vertex_normals = mesh.vertex_normals.copy()
        asset.face_normals = mesh.face_normals.copy()
        
        if asset.matrix_local is not None:
            asset.matrix_local = trans_vertex @ asset.matrix_local
        if asset.pose_matrix is not None:
            asset.pose_matrix = trans_vertex @ asset.pose_matrix
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, self.trans_vertex)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, self.trans_vertex)
            
        # normalize normals
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        vertex_norms = np.linalg.norm(asset.vertex_normals, axis=1, keepdims=True)
        vertex_norms = np.maximum(vertex_norms, epsilon)  # Ensure no zeros
        asset.vertex_normals = asset.vertex_normals / vertex_norms
        
        face_norms = np.linalg.norm(asset.face_normals, axis=1, keepdims=True)
        face_norms = np.maximum(face_norms, epsilon)  # Ensure no zeros
        asset.face_normals = asset.face_normals / face_norms
        
        # Clean up any remaining NaN or Inf values
        asset.vertex_normals = np.nan_to_num(asset.vertex_normals, nan=0., posinf=0., neginf=0.)
        asset.face_normals = np.nan_to_num(asset.face_normals, nan=0., posinf=0., neginf=0.)
        
    def inverse(self, asset: Asset):
        m = np.linalg.inv(self.trans_vertex)
        asset.vertices = self._apply(asset.vertices, m)
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, m)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, m)

class AugmentAffine(Augment):
    
    def __init__(self, config: AugmentAffineConfig):
        super().__init__()
        self.config = config

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        bound_min = asset.vertices.min(axis=0)
        bound_max = asset.vertices.max(axis=0)
        if asset.joints is not None:
            joints_bound_min = asset.joints.min(axis=0)
            joints_bound_max = asset.joints.max(axis=0)            
            bound_min = np.minimum(bound_min, joints_bound_min)
            bound_max = np.maximum(bound_max, joints_bound_max)
        
        trans_vertex = np.eye(4, dtype=np.float32)
        
        trans_vertex = _trans_to_m(-(bound_max + bound_min)/2) @ trans_vertex
        
        # scale into the cube
        normalize_into = self.config.normalize_into
        scale = np.max((bound_max - bound_min) / (normalize_into[1] - normalize_into[0]))
        trans_vertex = _scale_to_m(1. / scale) @ trans_vertex
        
        bias = (normalize_into[0] + normalize_into[1]) / 2
        trans_vertex = _trans_to_m(np.array([bias, bias, bias], dtype=np.float32)) @ trans_vertex
        
        if np.random.rand() < self.config.random_scale_p:
            scale = _scale_to_m(np.random.uniform(self.config.random_scale[0], self.config.random_scale[1]))
            trans_vertex = scale @ trans_vertex

        if np.random.rand() < self.config.random_shift_p:
            l, r = self.config.random_shift
            shift = _trans_to_m(np.array([np.random.uniform(l, r), np.random.uniform(l, r), np.random.uniform(l, r)]))
            trans_vertex = shift @ trans_vertex
        
        asset.vertices = self._apply(asset.vertices, trans_vertex)
        # do not affect scale in matrix
        if asset.matrix_local is not None:
            asset.matrix_local[:, :, 3:4] = trans_vertex @ asset.matrix_local[:, :, 3:4]
        if asset.pose_matrix is not None:
            asset.pose_matrix[:, :, 3:4] = trans_vertex @ asset.pose_matrix[:, :, 3:4]
        # do not affect normal here
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, trans_vertex)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, trans_vertex)
        
        self.trans_vertex = trans_vertex
    
    def inverse(self, asset: Asset):
        m = np.linalg.inv(self.trans_vertex)
        asset.vertices = self._apply(asset.vertices, m)
        if asset.joints is not None:
            asset.joints = self._apply(asset.joints, m)
        if asset.tails is not None:
            asset.tails = self._apply(asset.tails, m)

class AugmentJitter(Augment):
    
    def __init__(self, config: AugmentJitterConfig):
        super().__init__()
        self.config = config

    def _apply(self, v: ndarray, trans: ndarray) -> ndarray:
        return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

    def transform(self, asset: Asset, **kwargs):
        p = self.config.p
        vertex_sigma = self.config.vertex_sigma
        vertex_clip = self.config.vertex_clip
        normal_sigma = self.config.normal_sigma
        normal_clip = self.config.normal_clip
        
        if np.random.rand() < p:
            scale = np.random.rand() + 1e-6
            vertex_sigma *= scale
            vertex_clip *= scale
            scale = np.random.rand() + 1e-6
            normal_sigma *= scale
            normal_clip *= scale
            if vertex_sigma > 0:
                noise = np.clip(np.random.randn(*asset.vertices.shape) * vertex_sigma, -vertex_clip, vertex_clip).astype(np.float32)
                asset.vertices += noise
            
            if normal_sigma > 0:
                noise = np.clip(np.random.randn(*asset.vertex_normals.shape) * normal_sigma, -normal_clip, normal_clip).astype(np.float32)
                asset.vertex_normals += noise
                
                noise = np.clip(np.random.randn(*asset.face_normals.shape) * normal_sigma, -normal_clip, normal_clip).astype(np.float32)
                asset.face_normals += noise
                
                asset.vertex_normals = asset.vertex_normals / np.linalg.norm(asset.vertex_normals, axis=1, keepdims=True)
                
                asset.face_normals = asset.face_normals / np.linalg.norm(asset.face_normals, axis=1, keepdims=True)
                
                asset.vertex_normals = np.nan_to_num(asset.vertex_normals, nan=0., posinf=0., neginf=0.)
                
                asset.face_normals = np.nan_to_num(asset.face_normals, nan=0., posinf=0., neginf=0.)
    
    def inverse(self, asset: Asset):
        pass

def _trans_to_m(v: ndarray):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v
    return m

def _scale_to_m(r: ndarray):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = r
    m[1, 1] = r
    m[2, 2] = r
    m[3, 3] = 1.
    return m

def get_augments(config: AugmentConfig) -> Tuple[List[Augment], List[Augment]]:
    first_augments  = [] # augments before sample
    second_augments = [] # augments after sample
    augment_collapse_config         = config.augment_collapse_config
    augment_lbs_config              = config.augment_lbs_config
    augment_linear_config           = config.augment_linear_config
    augment_affine_config           = config.augment_affine_config
    augment_jitter_config           = config.augment_jitter_config
    augment_joint_perturb_config    = config.augment_joint_perturb_config
    augment_joint_discrete_config   = config.augment_joint_discrete_config
    augment_drop_part_config        = config.augment_drop_part_config
    if augment_drop_part_config is not None:
        first_augments.append(AugmentDropPart(config=augment_drop_part_config))
    if augment_collapse_config is not None:
        first_augments.append(AugmentCollapse(config=augment_collapse_config))
    if augment_lbs_config is not None:
        first_augments.append(AugmentLBS(config=augment_lbs_config))
    if augment_linear_config is not None:
        first_augments.append(AugmentLinear(config=augment_linear_config))
    if augment_affine_config is not None:
        first_augments.append(AugmentAffine(config=augment_affine_config))

    if augment_jitter_config is not None:
        second_augments.append(AugmentJitter(config=augment_jitter_config))
    if augment_joint_perturb_config is not None:
        second_augments.append(AugmentJointPerturb(config=augment_joint_perturb_config))
    if augment_joint_discrete_config is not None:
        second_augments.append(AugmentJointDiscrete(config=augment_joint_discrete_config))
    return first_augments, second_augments