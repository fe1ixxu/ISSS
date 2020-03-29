from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('LeapPython', [dirname(__file__)])
        except ImportError:
            import LeapPython
            return LeapPython
        if fp is not None:
            try:
                _mod = imp.load_module('LeapPython', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    LeapPython = swig_import_helper()
    del swig_import_helper
else:
    import LeapPython
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        object.__setattr__(self, name, value)
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


try:
    import weakref
    weakref_proxy = weakref.proxy
except:
    weakref_proxy = lambda x: x


class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = LeapPython.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return LeapPython.SwigPyIterator_value(self)

    def incr(self, n=1):
        return LeapPython.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return LeapPython.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return LeapPython.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return LeapPython.SwigPyIterator_equal(self, x)

    def copy(self):
        return LeapPython.SwigPyIterator_copy(self)

    def next(self):
        return LeapPython.SwigPyIterator_next(self)

    def __next__(self):
        return LeapPython.SwigPyIterator___next__(self)

    def previous(self):
        return LeapPython.SwigPyIterator_previous(self)

    def advance(self, n):
        return LeapPython.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return LeapPython.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return LeapPython.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return LeapPython.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return LeapPython.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return LeapPython.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return LeapPython.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = LeapPython.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class byte_array(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, byte_array, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, byte_array, name)
    __repr__ = _swig_repr

    def __init__(self, nelements):
        this = LeapPython.new_byte_array(nelements)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_destroy__ = LeapPython.delete_byte_array
    __del__ = lambda self: None

    def __getitem__(self, index):
        return LeapPython.byte_array___getitem__(self, index)

    def __setitem__(self, index, value):
        return LeapPython.byte_array___setitem__(self, index, value)

    def cast(self):
        return LeapPython.byte_array_cast(self)
    __swig_getmethods__["frompointer"] = lambda x: LeapPython.byte_array_frompointer
    if _newclass:
        frompointer = staticmethod(LeapPython.byte_array_frompointer)
byte_array_swigregister = LeapPython.byte_array_swigregister
byte_array_swigregister(byte_array)

def byte_array_frompointer(t):
    return LeapPython.byte_array_frompointer(t)
byte_array_frompointer = LeapPython.byte_array_frompointer

class float_array(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, float_array, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, float_array, name)
    __repr__ = _swig_repr

    def __init__(self, nelements):
        this = LeapPython.new_float_array(nelements)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_destroy__ = LeapPython.delete_float_array
    __del__ = lambda self: None

    def __getitem__(self, index):
        return LeapPython.float_array___getitem__(self, index)

    def __setitem__(self, index, value):
        return LeapPython.float_array___setitem__(self, index, value)

    def cast(self):
        return LeapPython.float_array_cast(self)
    __swig_getmethods__["frompointer"] = lambda x: LeapPython.float_array_frompointer
    if _newclass:
        frompointer = staticmethod(LeapPython.float_array_frompointer)
float_array_swigregister = LeapPython.float_array_swigregister
float_array_swigregister(float_array)

def float_array_frompointer(t):
    return LeapPython.float_array_frompointer(t)
float_array_frompointer = LeapPython.float_array_frompointer

class Vector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = LeapPython.new_Vector(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def distance_to(self, other):
        return LeapPython.Vector_distance_to(self, other)

    def angle_to(self, other):
        return LeapPython.Vector_angle_to(self, other)

    def dot(self, other):
        return LeapPython.Vector_dot(self, other)

    def cross(self, other):
        return LeapPython.Vector_cross(self, other)

    def __neg__(self):
        return LeapPython.Vector___neg__(self)

    def __add__(self, other):
        return LeapPython.Vector___add__(self, other)

    def __sub__(self, other):
        return LeapPython.Vector___sub__(self, other)

    def __mul__(self, scalar):
        return LeapPython.Vector___mul__(self, scalar)

    def __div__(self, scalar):
        return LeapPython.Vector___div__(self, scalar)

    def __iadd__(self, other):
        return LeapPython.Vector___iadd__(self, other)

    def __isub__(self, other):
        return LeapPython.Vector___isub__(self, other)

    def __imul__(self, scalar):
        return LeapPython.Vector___imul__(self, scalar)

    def __idiv__(self, scalar):
        return LeapPython.Vector___idiv__(self, scalar)

    def __str__(self):
        return LeapPython.Vector___str__(self)

    def __eq__(self, other):
        return LeapPython.Vector___eq__(self, other)

    def __ne__(self, other):
        return LeapPython.Vector___ne__(self, other)

    def is_valid(self):
        return LeapPython.Vector_is_valid(self)

    def __getitem__(self, index):
        return LeapPython.Vector___getitem__(self, index)
    __swig_setmethods__["x"] = LeapPython.Vector_x_set
    __swig_getmethods__["x"] = LeapPython.Vector_x_get
    if _newclass:
        x = _swig_property(LeapPython.Vector_x_get, LeapPython.Vector_x_set)
    __swig_setmethods__["y"] = LeapPython.Vector_y_set
    __swig_getmethods__["y"] = LeapPython.Vector_y_get
    if _newclass:
        y = _swig_property(LeapPython.Vector_y_get, LeapPython.Vector_y_set)
    __swig_setmethods__["z"] = LeapPython.Vector_z_set
    __swig_getmethods__["z"] = LeapPython.Vector_z_get
    if _newclass:
        z = _swig_property(LeapPython.Vector_z_get, LeapPython.Vector_z_set)
    __swig_getmethods__["magnitude"] = LeapPython.Vector_magnitude_get
    if _newclass:
        magnitude = _swig_property(LeapPython.Vector_magnitude_get)
    __swig_getmethods__["magnitude_squared"] = LeapPython.Vector_magnitude_squared_get
    if _newclass:
        magnitude_squared = _swig_property(LeapPython.Vector_magnitude_squared_get)
    __swig_getmethods__["pitch"] = LeapPython.Vector_pitch_get
    if _newclass:
        pitch = _swig_property(LeapPython.Vector_pitch_get)
    __swig_getmethods__["roll"] = LeapPython.Vector_roll_get
    if _newclass:
        roll = _swig_property(LeapPython.Vector_roll_get)
    __swig_getmethods__["yaw"] = LeapPython.Vector_yaw_get
    if _newclass:
        yaw = _swig_property(LeapPython.Vector_yaw_get)
    __swig_getmethods__["normalized"] = LeapPython.Vector_normalized_get
    if _newclass:
        normalized = _swig_property(LeapPython.Vector_normalized_get)
    def to_float_array(self): return [self.x, self.y, self.z]
    def to_tuple(self): return (self.x, self.y, self.z)

    __swig_destroy__ = LeapPython.delete_Vector
    __del__ = lambda self: None
Vector_swigregister = LeapPython.Vector_swigregister
Vector_swigregister(Vector)
cvar = LeapPython.cvar
PI = cvar.PI
DEG_TO_RAD = cvar.DEG_TO_RAD
RAD_TO_DEG = cvar.RAD_TO_DEG
EPSILON = cvar.EPSILON
Vector.zero = LeapPython.cvar.Vector_zero
Vector.x_axis = LeapPython.cvar.Vector_x_axis
Vector.y_axis = LeapPython.cvar.Vector_y_axis
Vector.z_axis = LeapPython.cvar.Vector_z_axis
Vector.forward = LeapPython.cvar.Vector_forward
Vector.backward = LeapPython.cvar.Vector_backward
Vector.left = LeapPython.cvar.Vector_left
Vector.right = LeapPython.cvar.Vector_right
Vector.up = LeapPython.cvar.Vector_up
Vector.down = LeapPython.cvar.Vector_down

class Matrix(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Matrix, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Matrix, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = LeapPython.new_Matrix(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def set_rotation(self, axis, angleRadians):
        return LeapPython.Matrix_set_rotation(self, axis, angleRadians)

    def transform_point(self, arg2):
        return LeapPython.Matrix_transform_point(self, arg2)

    def transform_direction(self, arg2):
        return LeapPython.Matrix_transform_direction(self, arg2)

    def rigid_inverse(self):
        return LeapPython.Matrix_rigid_inverse(self)

    def __mul__(self, other):
        return LeapPython.Matrix___mul__(self, other)

    def __imul__(self, other):
        return LeapPython.Matrix___imul__(self, other)

    def __eq__(self, other):
        return LeapPython.Matrix___eq__(self, other)

    def __ne__(self, other):
        return LeapPython.Matrix___ne__(self, other)

    def __str__(self):
        return LeapPython.Matrix___str__(self)
    __swig_setmethods__["x_basis"] = LeapPython.Matrix_x_basis_set
    __swig_getmethods__["x_basis"] = LeapPython.Matrix_x_basis_get
    if _newclass:
        x_basis = _swig_property(LeapPython.Matrix_x_basis_get, LeapPython.Matrix_x_basis_set)
    __swig_setmethods__["y_basis"] = LeapPython.Matrix_y_basis_set
    __swig_getmethods__["y_basis"] = LeapPython.Matrix_y_basis_get
    if _newclass:
        y_basis = _swig_property(LeapPython.Matrix_y_basis_get, LeapPython.Matrix_y_basis_set)
    __swig_setmethods__["z_basis"] = LeapPython.Matrix_z_basis_set
    __swig_getmethods__["z_basis"] = LeapPython.Matrix_z_basis_get
    if _newclass:
        z_basis = _swig_property(LeapPython.Matrix_z_basis_get, LeapPython.Matrix_z_basis_set)
    __swig_setmethods__["origin"] = LeapPython.Matrix_origin_set
    __swig_getmethods__["origin"] = LeapPython.Matrix_origin_get
    if _newclass:
        origin = _swig_property(LeapPython.Matrix_origin_get, LeapPython.Matrix_origin_set)
    def to_array_3x3(self, output = None):
        if output is None:
            output = [0]*9
        output[0], output[1], output[2] = self.x_basis.x, self.x_basis.y, self.x_basis.z
        output[3], output[4], output[5] = self.y_basis.x, self.y_basis.y, self.y_basis.z
        output[6], output[7], output[8] = self.z_basis.x, self.z_basis.y, self.z_basis.z
        return output
    def to_array_4x4(self, output = None):
        if output is None:
            output = [0]*16
        output[0],  output[1],  output[2],  output[3]  = self.x_basis.x, self.x_basis.y, self.x_basis.z, 0.0
        output[4],  output[5],  output[6],  output[7]  = self.y_basis.x, self.y_basis.y, self.y_basis.z, 0.0
        output[8],  output[9],  output[10], output[11] = self.z_basis.x, self.z_basis.y, self.z_basis.z, 0.0
        output[12], output[13], output[14], output[15] = self.origin.x,  self.origin.y,  self.origin.z,  1.0
        return output

    __swig_destroy__ = LeapPython.delete_Matrix
    __del__ = lambda self: None
Matrix_swigregister = LeapPython.Matrix_swigregister
Matrix_swigregister(Matrix)
Matrix.identity = LeapPython.cvar.Matrix_identity

class Interface(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Interface, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Interface, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined")
    __repr__ = _swig_repr
Interface_swigregister = LeapPython.Interface_swigregister
Interface_swigregister(Interface)

class Pointable(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Pointable, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Pointable, name)
    __repr__ = _swig_repr
    ZONE_NONE = LeapPython.Pointable_ZONE_NONE
    ZONE_HOVERING = LeapPython.Pointable_ZONE_HOVERING
    ZONE_TOUCHING = LeapPython.Pointable_ZONE_TOUCHING

    def __init__(self):
        this = LeapPython.new_Pointable()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __eq__(self, arg2):
        return LeapPython.Pointable___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Pointable___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Pointable___str__(self)
    __swig_getmethods__["id"] = LeapPython.Pointable_id_get
    if _newclass:
        id = _swig_property(LeapPython.Pointable_id_get)
    __swig_getmethods__["hand"] = LeapPython.Pointable_hand_get
    if _newclass:
        hand = _swig_property(LeapPython.Pointable_hand_get)
    __swig_getmethods__["tip_position"] = LeapPython.Pointable_tip_position_get
    if _newclass:
        tip_position = _swig_property(LeapPython.Pointable_tip_position_get)
    __swig_getmethods__["tip_velocity"] = LeapPython.Pointable_tip_velocity_get
    if _newclass:
        tip_velocity = _swig_property(LeapPython.Pointable_tip_velocity_get)
    __swig_getmethods__["direction"] = LeapPython.Pointable_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.Pointable_direction_get)
    __swig_getmethods__["width"] = LeapPython.Pointable_width_get
    if _newclass:
        width = _swig_property(LeapPython.Pointable_width_get)
    __swig_getmethods__["length"] = LeapPython.Pointable_length_get
    if _newclass:
        length = _swig_property(LeapPython.Pointable_length_get)
    __swig_getmethods__["is_tool"] = LeapPython.Pointable_is_tool_get
    if _newclass:
        is_tool = _swig_property(LeapPython.Pointable_is_tool_get)
    __swig_getmethods__["is_finger"] = LeapPython.Pointable_is_finger_get
    if _newclass:
        is_finger = _swig_property(LeapPython.Pointable_is_finger_get)
    __swig_getmethods__["is_extended"] = LeapPython.Pointable_is_extended_get
    if _newclass:
        is_extended = _swig_property(LeapPython.Pointable_is_extended_get)
    __swig_getmethods__["is_valid"] = LeapPython.Pointable_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Pointable_is_valid_get)
    __swig_getmethods__["touch_zone"] = LeapPython.Pointable_touch_zone_get
    if _newclass:
        touch_zone = _swig_property(LeapPython.Pointable_touch_zone_get)
    __swig_getmethods__["touch_distance"] = LeapPython.Pointable_touch_distance_get
    if _newclass:
        touch_distance = _swig_property(LeapPython.Pointable_touch_distance_get)
    __swig_getmethods__["stabilized_tip_position"] = LeapPython.Pointable_stabilized_tip_position_get
    if _newclass:
        stabilized_tip_position = _swig_property(LeapPython.Pointable_stabilized_tip_position_get)
    __swig_getmethods__["time_visible"] = LeapPython.Pointable_time_visible_get
    if _newclass:
        time_visible = _swig_property(LeapPython.Pointable_time_visible_get)
    __swig_getmethods__["frame"] = LeapPython.Pointable_frame_get
    if _newclass:
        frame = _swig_property(LeapPython.Pointable_frame_get)
    __swig_destroy__ = LeapPython.delete_Pointable
    __del__ = lambda self: None
Pointable_swigregister = LeapPython.Pointable_swigregister
Pointable_swigregister(Pointable)
Pointable.invalid = LeapPython.cvar.Pointable_invalid

class Arm(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Arm, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Arm, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_Arm()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __eq__(self, arg2):
        return LeapPython.Arm___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Arm___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Arm___str__(self)
    __swig_getmethods__["width"] = LeapPython.Arm_width_get
    if _newclass:
        width = _swig_property(LeapPython.Arm_width_get)
    __swig_getmethods__["center"] = LeapPython.Arm_center_get
    if _newclass:
        center = _swig_property(LeapPython.Arm_center_get)
    __swig_getmethods__["direction"] = LeapPython.Arm_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.Arm_direction_get)
    __swig_getmethods__["basis"] = LeapPython.Arm_basis_get
    if _newclass:
        basis = _swig_property(LeapPython.Arm_basis_get)
    __swig_getmethods__["elbow_position"] = LeapPython.Arm_elbow_position_get
    if _newclass:
        elbow_position = _swig_property(LeapPython.Arm_elbow_position_get)
    __swig_getmethods__["wrist_position"] = LeapPython.Arm_wrist_position_get
    if _newclass:
        wrist_position = _swig_property(LeapPython.Arm_wrist_position_get)
    __swig_getmethods__["is_valid"] = LeapPython.Arm_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Arm_is_valid_get)
    __swig_destroy__ = LeapPython.delete_Arm
    __del__ = lambda self: None
Arm_swigregister = LeapPython.Arm_swigregister
Arm_swigregister(Arm)
Arm.invalid = LeapPython.cvar.Arm_invalid

class Bone(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Bone, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Bone, name)
    __repr__ = _swig_repr
    TYPE_METACARPAL = LeapPython.Bone_TYPE_METACARPAL
    TYPE_PROXIMAL = LeapPython.Bone_TYPE_PROXIMAL
    TYPE_INTERMEDIATE = LeapPython.Bone_TYPE_INTERMEDIATE
    TYPE_DISTAL = LeapPython.Bone_TYPE_DISTAL

    def __init__(self):
        this = LeapPython.new_Bone()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __eq__(self, arg2):
        return LeapPython.Bone___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Bone___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Bone___str__(self)
    __swig_getmethods__["prev_joint"] = LeapPython.Bone_prev_joint_get
    if _newclass:
        prev_joint = _swig_property(LeapPython.Bone_prev_joint_get)
    __swig_getmethods__["next_joint"] = LeapPython.Bone_next_joint_get
    if _newclass:
        next_joint = _swig_property(LeapPython.Bone_next_joint_get)
    __swig_getmethods__["center"] = LeapPython.Bone_center_get
    if _newclass:
        center = _swig_property(LeapPython.Bone_center_get)
    __swig_getmethods__["direction"] = LeapPython.Bone_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.Bone_direction_get)
    __swig_getmethods__["length"] = LeapPython.Bone_length_get
    if _newclass:
        length = _swig_property(LeapPython.Bone_length_get)
    __swig_getmethods__["width"] = LeapPython.Bone_width_get
    if _newclass:
        width = _swig_property(LeapPython.Bone_width_get)
    __swig_getmethods__["type"] = LeapPython.Bone_type_get
    if _newclass:
        type = _swig_property(LeapPython.Bone_type_get)
    __swig_getmethods__["basis"] = LeapPython.Bone_basis_get
    if _newclass:
        basis = _swig_property(LeapPython.Bone_basis_get)
    __swig_getmethods__["is_valid"] = LeapPython.Bone_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Bone_is_valid_get)
    __swig_destroy__ = LeapPython.delete_Bone
    __del__ = lambda self: None
Bone_swigregister = LeapPython.Bone_swigregister
Bone_swigregister(Bone)
Bone.invalid = LeapPython.cvar.Bone_invalid

class Finger(Pointable):
    __swig_setmethods__ = {}
    for _s in [Pointable]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Finger, name, value)
    __swig_getmethods__ = {}
    for _s in [Pointable]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Finger, name)
    __repr__ = _swig_repr
    JOINT_MCP = LeapPython.Finger_JOINT_MCP
    JOINT_PIP = LeapPython.Finger_JOINT_PIP
    JOINT_DIP = LeapPython.Finger_JOINT_DIP
    JOINT_TIP = LeapPython.Finger_JOINT_TIP
    TYPE_THUMB = LeapPython.Finger_TYPE_THUMB
    TYPE_INDEX = LeapPython.Finger_TYPE_INDEX
    TYPE_MIDDLE = LeapPython.Finger_TYPE_MIDDLE
    TYPE_RING = LeapPython.Finger_TYPE_RING
    TYPE_PINKY = LeapPython.Finger_TYPE_PINKY

    def __init__(self, *args):
        this = LeapPython.new_Finger(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def joint_position(self, jointIx):
        return LeapPython.Finger_joint_position(self, jointIx)

    def bone(self, boneIx):
        return LeapPython.Finger_bone(self, boneIx)

    def __str__(self):
        return LeapPython.Finger___str__(self)
    __swig_getmethods__["type"] = LeapPython.Finger_type_get
    if _newclass:
        type = _swig_property(LeapPython.Finger_type_get)
    __swig_destroy__ = LeapPython.delete_Finger
    __del__ = lambda self: None
Finger_swigregister = LeapPython.Finger_swigregister
Finger_swigregister(Finger)
Finger.invalid = LeapPython.cvar.Finger_invalid

class Tool(Pointable):
    __swig_setmethods__ = {}
    for _s in [Pointable]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Tool, name, value)
    __swig_getmethods__ = {}
    for _s in [Pointable]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Tool, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = LeapPython.new_Tool(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def __str__(self):
        return LeapPython.Tool___str__(self)
    __swig_destroy__ = LeapPython.delete_Tool
    __del__ = lambda self: None
Tool_swigregister = LeapPython.Tool_swigregister
Tool_swigregister(Tool)
Tool.invalid = LeapPython.cvar.Tool_invalid

class Hand(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Hand, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Hand, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_Hand()
        try:
            self.this.append(this)
        except:
            self.this = this

    def pointable(self, id):
        return LeapPython.Hand_pointable(self, id)

    def finger(self, id):
        return LeapPython.Hand_finger(self, id)

    def translation(self, sinceFrame):
        return LeapPython.Hand_translation(self, sinceFrame)

    def translation_probability(self, sinceFrame):
        return LeapPython.Hand_translation_probability(self, sinceFrame)

    def rotation_axis(self, sinceFrame):
        return LeapPython.Hand_rotation_axis(self, sinceFrame)

    def rotation_angle(self, *args):
        return LeapPython.Hand_rotation_angle(self, *args)

    def rotation_matrix(self, sinceFrame):
        return LeapPython.Hand_rotation_matrix(self, sinceFrame)

    def rotation_probability(self, sinceFrame):
        return LeapPython.Hand_rotation_probability(self, sinceFrame)

    def scale_factor(self, sinceFrame):
        return LeapPython.Hand_scale_factor(self, sinceFrame)

    def scale_probability(self, sinceFrame):
        return LeapPython.Hand_scale_probability(self, sinceFrame)

    def __eq__(self, arg2):
        return LeapPython.Hand___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Hand___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Hand___str__(self)
    __swig_getmethods__["id"] = LeapPython.Hand_id_get
    if _newclass:
        id = _swig_property(LeapPython.Hand_id_get)
    __swig_getmethods__["pointables"] = LeapPython.Hand_pointables_get
    if _newclass:
        pointables = _swig_property(LeapPython.Hand_pointables_get)
    __swig_getmethods__["fingers"] = LeapPython.Hand_fingers_get
    if _newclass:
        fingers = _swig_property(LeapPython.Hand_fingers_get)
    __swig_getmethods__["palm_position"] = LeapPython.Hand_palm_position_get
    if _newclass:
        palm_position = _swig_property(LeapPython.Hand_palm_position_get)
    __swig_getmethods__["palm_velocity"] = LeapPython.Hand_palm_velocity_get
    if _newclass:
        palm_velocity = _swig_property(LeapPython.Hand_palm_velocity_get)
    __swig_getmethods__["palm_normal"] = LeapPython.Hand_palm_normal_get
    if _newclass:
        palm_normal = _swig_property(LeapPython.Hand_palm_normal_get)
    __swig_getmethods__["direction"] = LeapPython.Hand_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.Hand_direction_get)
    __swig_getmethods__["basis"] = LeapPython.Hand_basis_get
    if _newclass:
        basis = _swig_property(LeapPython.Hand_basis_get)
    __swig_getmethods__["is_valid"] = LeapPython.Hand_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Hand_is_valid_get)
    __swig_getmethods__["sphere_center"] = LeapPython.Hand_sphere_center_get
    if _newclass:
        sphere_center = _swig_property(LeapPython.Hand_sphere_center_get)
    __swig_getmethods__["sphere_radius"] = LeapPython.Hand_sphere_radius_get
    if _newclass:
        sphere_radius = _swig_property(LeapPython.Hand_sphere_radius_get)
    __swig_getmethods__["grab_angle"] = LeapPython.Hand_grab_angle_get
    if _newclass:
        grab_angle = _swig_property(LeapPython.Hand_grab_angle_get)
    __swig_getmethods__["pinch_distance"] = LeapPython.Hand_pinch_distance_get
    if _newclass:
        pinch_distance = _swig_property(LeapPython.Hand_pinch_distance_get)
    __swig_getmethods__["grab_strength"] = LeapPython.Hand_grab_strength_get
    if _newclass:
        grab_strength = _swig_property(LeapPython.Hand_grab_strength_get)
    __swig_getmethods__["pinch_strength"] = LeapPython.Hand_pinch_strength_get
    if _newclass:
        pinch_strength = _swig_property(LeapPython.Hand_pinch_strength_get)
    __swig_getmethods__["palm_width"] = LeapPython.Hand_palm_width_get
    if _newclass:
        palm_width = _swig_property(LeapPython.Hand_palm_width_get)
    __swig_getmethods__["stabilized_palm_position"] = LeapPython.Hand_stabilized_palm_position_get
    if _newclass:
        stabilized_palm_position = _swig_property(LeapPython.Hand_stabilized_palm_position_get)
    __swig_getmethods__["wrist_position"] = LeapPython.Hand_wrist_position_get
    if _newclass:
        wrist_position = _swig_property(LeapPython.Hand_wrist_position_get)
    __swig_getmethods__["time_visible"] = LeapPython.Hand_time_visible_get
    if _newclass:
        time_visible = _swig_property(LeapPython.Hand_time_visible_get)
    __swig_getmethods__["confidence"] = LeapPython.Hand_confidence_get
    if _newclass:
        confidence = _swig_property(LeapPython.Hand_confidence_get)
    __swig_getmethods__["is_left"] = LeapPython.Hand_is_left_get
    if _newclass:
        is_left = _swig_property(LeapPython.Hand_is_left_get)
    __swig_getmethods__["is_right"] = LeapPython.Hand_is_right_get
    if _newclass:
        is_right = _swig_property(LeapPython.Hand_is_right_get)
    __swig_getmethods__["frame"] = LeapPython.Hand_frame_get
    if _newclass:
        frame = _swig_property(LeapPython.Hand_frame_get)
    __swig_getmethods__["arm"] = LeapPython.Hand_arm_get
    if _newclass:
        arm = _swig_property(LeapPython.Hand_arm_get)
    __swig_destroy__ = LeapPython.delete_Hand
    __del__ = lambda self: None
Hand_swigregister = LeapPython.Hand_swigregister
Hand_swigregister(Hand)
Hand.invalid = LeapPython.cvar.Hand_invalid

class Gesture(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Gesture, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Gesture, name)
    __repr__ = _swig_repr
    TYPE_INVALID = LeapPython.Gesture_TYPE_INVALID
    TYPE_SWIPE = LeapPython.Gesture_TYPE_SWIPE
    TYPE_CIRCLE = LeapPython.Gesture_TYPE_CIRCLE
    TYPE_SCREEN_TAP = LeapPython.Gesture_TYPE_SCREEN_TAP
    TYPE_KEY_TAP = LeapPython.Gesture_TYPE_KEY_TAP
    STATE_INVALID = LeapPython.Gesture_STATE_INVALID
    STATE_START = LeapPython.Gesture_STATE_START
    STATE_UPDATE = LeapPython.Gesture_STATE_UPDATE
    STATE_STOP = LeapPython.Gesture_STATE_STOP

    def __init__(self, *args):
        this = LeapPython.new_Gesture(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def __eq__(self, rhs):
        return LeapPython.Gesture___eq__(self, rhs)

    def __ne__(self, rhs):
        return LeapPython.Gesture___ne__(self, rhs)

    def __str__(self):
        return LeapPython.Gesture___str__(self)
    __swig_getmethods__["type"] = LeapPython.Gesture_type_get
    if _newclass:
        type = _swig_property(LeapPython.Gesture_type_get)
    __swig_getmethods__["state"] = LeapPython.Gesture_state_get
    if _newclass:
        state = _swig_property(LeapPython.Gesture_state_get)
    __swig_getmethods__["id"] = LeapPython.Gesture_id_get
    if _newclass:
        id = _swig_property(LeapPython.Gesture_id_get)
    __swig_getmethods__["duration"] = LeapPython.Gesture_duration_get
    if _newclass:
        duration = _swig_property(LeapPython.Gesture_duration_get)
    __swig_getmethods__["duration_seconds"] = LeapPython.Gesture_duration_seconds_get
    if _newclass:
        duration_seconds = _swig_property(LeapPython.Gesture_duration_seconds_get)
    __swig_getmethods__["frame"] = LeapPython.Gesture_frame_get
    if _newclass:
        frame = _swig_property(LeapPython.Gesture_frame_get)
    __swig_getmethods__["hands"] = LeapPython.Gesture_hands_get
    if _newclass:
        hands = _swig_property(LeapPython.Gesture_hands_get)
    __swig_getmethods__["pointables"] = LeapPython.Gesture_pointables_get
    if _newclass:
        pointables = _swig_property(LeapPython.Gesture_pointables_get)
    __swig_getmethods__["is_valid"] = LeapPython.Gesture_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Gesture_is_valid_get)
    __swig_destroy__ = LeapPython.delete_Gesture
    __del__ = lambda self: None
Gesture_swigregister = LeapPython.Gesture_swigregister
Gesture_swigregister(Gesture)
Gesture.invalid = LeapPython.cvar.Gesture_invalid

class SwipeGesture(Gesture):
    __swig_setmethods__ = {}
    for _s in [Gesture]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwipeGesture, name, value)
    __swig_getmethods__ = {}
    for _s in [Gesture]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, SwipeGesture, name)
    __repr__ = _swig_repr
    __swig_getmethods__["class_type"] = lambda x: LeapPython.SwipeGesture_class_type
    if _newclass:
        class_type = staticmethod(LeapPython.SwipeGesture_class_type)

    def __init__(self, *args):
        this = LeapPython.new_SwipeGesture(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_getmethods__["start_position"] = LeapPython.SwipeGesture_start_position_get
    if _newclass:
        start_position = _swig_property(LeapPython.SwipeGesture_start_position_get)
    __swig_getmethods__["position"] = LeapPython.SwipeGesture_position_get
    if _newclass:
        position = _swig_property(LeapPython.SwipeGesture_position_get)
    __swig_getmethods__["direction"] = LeapPython.SwipeGesture_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.SwipeGesture_direction_get)
    __swig_getmethods__["speed"] = LeapPython.SwipeGesture_speed_get
    if _newclass:
        speed = _swig_property(LeapPython.SwipeGesture_speed_get)
    __swig_getmethods__["pointable"] = LeapPython.SwipeGesture_pointable_get
    if _newclass:
        pointable = _swig_property(LeapPython.SwipeGesture_pointable_get)
    __swig_destroy__ = LeapPython.delete_SwipeGesture
    __del__ = lambda self: None
SwipeGesture_swigregister = LeapPython.SwipeGesture_swigregister
SwipeGesture_swigregister(SwipeGesture)

def SwipeGesture_class_type():
    return LeapPython.SwipeGesture_class_type()
SwipeGesture_class_type = LeapPython.SwipeGesture_class_type

class CircleGesture(Gesture):
    __swig_setmethods__ = {}
    for _s in [Gesture]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, CircleGesture, name, value)
    __swig_getmethods__ = {}
    for _s in [Gesture]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, CircleGesture, name)
    __repr__ = _swig_repr
    __swig_getmethods__["class_type"] = lambda x: LeapPython.CircleGesture_class_type
    if _newclass:
        class_type = staticmethod(LeapPython.CircleGesture_class_type)

    def __init__(self, *args):
        this = LeapPython.new_CircleGesture(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_getmethods__["center"] = LeapPython.CircleGesture_center_get
    if _newclass:
        center = _swig_property(LeapPython.CircleGesture_center_get)
    __swig_getmethods__["normal"] = LeapPython.CircleGesture_normal_get
    if _newclass:
        normal = _swig_property(LeapPython.CircleGesture_normal_get)
    __swig_getmethods__["progress"] = LeapPython.CircleGesture_progress_get
    if _newclass:
        progress = _swig_property(LeapPython.CircleGesture_progress_get)
    __swig_getmethods__["radius"] = LeapPython.CircleGesture_radius_get
    if _newclass:
        radius = _swig_property(LeapPython.CircleGesture_radius_get)
    __swig_getmethods__["pointable"] = LeapPython.CircleGesture_pointable_get
    if _newclass:
        pointable = _swig_property(LeapPython.CircleGesture_pointable_get)
    __swig_destroy__ = LeapPython.delete_CircleGesture
    __del__ = lambda self: None
CircleGesture_swigregister = LeapPython.CircleGesture_swigregister
CircleGesture_swigregister(CircleGesture)

def CircleGesture_class_type():
    return LeapPython.CircleGesture_class_type()
CircleGesture_class_type = LeapPython.CircleGesture_class_type

class ScreenTapGesture(Gesture):
    __swig_setmethods__ = {}
    for _s in [Gesture]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ScreenTapGesture, name, value)
    __swig_getmethods__ = {}
    for _s in [Gesture]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ScreenTapGesture, name)
    __repr__ = _swig_repr
    __swig_getmethods__["class_type"] = lambda x: LeapPython.ScreenTapGesture_class_type
    if _newclass:
        class_type = staticmethod(LeapPython.ScreenTapGesture_class_type)

    def __init__(self, *args):
        this = LeapPython.new_ScreenTapGesture(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_getmethods__["position"] = LeapPython.ScreenTapGesture_position_get
    if _newclass:
        position = _swig_property(LeapPython.ScreenTapGesture_position_get)
    __swig_getmethods__["direction"] = LeapPython.ScreenTapGesture_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.ScreenTapGesture_direction_get)
    __swig_getmethods__["progress"] = LeapPython.ScreenTapGesture_progress_get
    if _newclass:
        progress = _swig_property(LeapPython.ScreenTapGesture_progress_get)
    __swig_getmethods__["pointable"] = LeapPython.ScreenTapGesture_pointable_get
    if _newclass:
        pointable = _swig_property(LeapPython.ScreenTapGesture_pointable_get)
    __swig_destroy__ = LeapPython.delete_ScreenTapGesture
    __del__ = lambda self: None
ScreenTapGesture_swigregister = LeapPython.ScreenTapGesture_swigregister
ScreenTapGesture_swigregister(ScreenTapGesture)

def ScreenTapGesture_class_type():
    return LeapPython.ScreenTapGesture_class_type()
ScreenTapGesture_class_type = LeapPython.ScreenTapGesture_class_type

class KeyTapGesture(Gesture):
    __swig_setmethods__ = {}
    for _s in [Gesture]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, KeyTapGesture, name, value)
    __swig_getmethods__ = {}
    for _s in [Gesture]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, KeyTapGesture, name)
    __repr__ = _swig_repr
    __swig_getmethods__["class_type"] = lambda x: LeapPython.KeyTapGesture_class_type
    if _newclass:
        class_type = staticmethod(LeapPython.KeyTapGesture_class_type)

    def __init__(self, *args):
        this = LeapPython.new_KeyTapGesture(*args)
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_getmethods__["position"] = LeapPython.KeyTapGesture_position_get
    if _newclass:
        position = _swig_property(LeapPython.KeyTapGesture_position_get)
    __swig_getmethods__["direction"] = LeapPython.KeyTapGesture_direction_get
    if _newclass:
        direction = _swig_property(LeapPython.KeyTapGesture_direction_get)
    __swig_getmethods__["progress"] = LeapPython.KeyTapGesture_progress_get
    if _newclass:
        progress = _swig_property(LeapPython.KeyTapGesture_progress_get)
    __swig_getmethods__["pointable"] = LeapPython.KeyTapGesture_pointable_get
    if _newclass:
        pointable = _swig_property(LeapPython.KeyTapGesture_pointable_get)
    __swig_destroy__ = LeapPython.delete_KeyTapGesture
    __del__ = lambda self: None
KeyTapGesture_swigregister = LeapPython.KeyTapGesture_swigregister
KeyTapGesture_swigregister(KeyTapGesture)

def KeyTapGesture_class_type():
    return LeapPython.KeyTapGesture_class_type()
KeyTapGesture_class_type = LeapPython.KeyTapGesture_class_type

class Device(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Device, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Device, name)
    __repr__ = _swig_repr
    TYPE_PERIPHERAL = LeapPython.Device_TYPE_PERIPHERAL
    TYPE_HP_LEGACY = LeapPython.Device_TYPE_HP_LEGACY
    TYPE_KEYBOARD = LeapPython.Device_TYPE_KEYBOARD
    TYPE_LAPTOP = LeapPython.Device_TYPE_LAPTOP

    def __init__(self):
        this = LeapPython.new_Device()
        try:
            self.this.append(this)
        except:
            self.this = this

    def distance_to_boundary(self, position):
        return LeapPython.Device_distance_to_boundary(self, position)

    def __eq__(self, arg2):
        return LeapPython.Device___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Device___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Device___str__(self)
    __swig_getmethods__["horizontal_view_angle"] = LeapPython.Device_horizontal_view_angle_get
    if _newclass:
        horizontal_view_angle = _swig_property(LeapPython.Device_horizontal_view_angle_get)
    __swig_getmethods__["vertical_view_angle"] = LeapPython.Device_vertical_view_angle_get
    if _newclass:
        vertical_view_angle = _swig_property(LeapPython.Device_vertical_view_angle_get)
    __swig_getmethods__["range"] = LeapPython.Device_range_get
    if _newclass:
        range = _swig_property(LeapPython.Device_range_get)
    __swig_getmethods__["baseline"] = LeapPython.Device_baseline_get
    if _newclass:
        baseline = _swig_property(LeapPython.Device_baseline_get)
    __swig_getmethods__["is_valid"] = LeapPython.Device_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Device_is_valid_get)
    __swig_getmethods__["is_embedded"] = LeapPython.Device_is_embedded_get
    if _newclass:
        is_embedded = _swig_property(LeapPython.Device_is_embedded_get)
    __swig_getmethods__["is_streaming"] = LeapPython.Device_is_streaming_get
    if _newclass:
        is_streaming = _swig_property(LeapPython.Device_is_streaming_get)
    __swig_getmethods__["is_smudged"] = LeapPython.Device_is_smudged_get
    if _newclass:
        is_smudged = _swig_property(LeapPython.Device_is_smudged_get)
    __swig_getmethods__["is_lighting_bad"] = LeapPython.Device_is_lighting_bad_get
    if _newclass:
        is_lighting_bad = _swig_property(LeapPython.Device_is_lighting_bad_get)
    __swig_getmethods__["type"] = LeapPython.Device_type_get
    if _newclass:
        type = _swig_property(LeapPython.Device_type_get)
    __swig_getmethods__["serial_number"] = LeapPython.Device_serial_number_get
    if _newclass:
        serial_number = _swig_property(LeapPython.Device_serial_number_get)
    __swig_getmethods__["position"] = LeapPython.Device_position_get
    if _newclass:
        position = _swig_property(LeapPython.Device_position_get)
    __swig_getmethods__["orientation"] = LeapPython.Device_orientation_get
    if _newclass:
        orientation = _swig_property(LeapPython.Device_orientation_get)
    __swig_destroy__ = LeapPython.delete_Device
    __del__ = lambda self: None
Device_swigregister = LeapPython.Device_swigregister
Device_swigregister(Device)
Device.invalid = LeapPython.cvar.Device_invalid

class FailedDevice(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FailedDevice, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, FailedDevice, name)
    __repr__ = _swig_repr
    FAIL_UNKNOWN = LeapPython.FailedDevice_FAIL_UNKNOWN
    FAIL_CALIBRATION = LeapPython.FailedDevice_FAIL_CALIBRATION
    FAIL_FIRMWARE = LeapPython.FailedDevice_FAIL_FIRMWARE
    FAIL_TRANSPORT = LeapPython.FailedDevice_FAIL_TRANSPORT
    FAIL_CONTROL = LeapPython.FailedDevice_FAIL_CONTROL
    FAIL_COUNT = LeapPython.FailedDevice_FAIL_COUNT

    def __init__(self):
        this = LeapPython.new_FailedDevice()
        try:
            self.this.append(this)
        except:
            self.this = this

    def is_valid(self):
        return LeapPython.FailedDevice_is_valid(self)
    __swig_getmethods__["invalid"] = lambda x: LeapPython.FailedDevice_invalid
    if _newclass:
        invalid = staticmethod(LeapPython.FailedDevice_invalid)

    def __eq__(self, arg2):
        return LeapPython.FailedDevice___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.FailedDevice___ne__(self, arg2)
    __swig_getmethods__["pnp_id"] = LeapPython.FailedDevice_pnp_id_get
    if _newclass:
        pnp_id = _swig_property(LeapPython.FailedDevice_pnp_id_get)
    __swig_getmethods__["failure"] = LeapPython.FailedDevice_failure_get
    if _newclass:
        failure = _swig_property(LeapPython.FailedDevice_failure_get)
    __swig_destroy__ = LeapPython.delete_FailedDevice
    __del__ = lambda self: None
FailedDevice_swigregister = LeapPython.FailedDevice_swigregister
FailedDevice_swigregister(FailedDevice)

def FailedDevice_invalid():
    return LeapPython.FailedDevice_invalid()
FailedDevice_invalid = LeapPython.FailedDevice_invalid

class Image(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Image, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Image, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_Image()
        try:
            self.this.append(this)
        except:
            self.this = this

    def data(self, dst):
        return LeapPython.Image_data(self, dst)

    def distortion(self, dst):
        return LeapPython.Image_distortion(self, dst)
    INFRARED = LeapPython.Image_INFRARED
    IBRG = LeapPython.Image_IBRG

    def rectify(self, uv):
        return LeapPython.Image_rectify(self, uv)

    def warp(self, xy):
        return LeapPython.Image_warp(self, xy)

    def __eq__(self, arg2):
        return LeapPython.Image___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Image___ne__(self, arg2)

    def __str__(self):
        return LeapPython.Image___str__(self)
    __swig_getmethods__["sequence_id"] = LeapPython.Image_sequence_id_get
    if _newclass:
        sequence_id = _swig_property(LeapPython.Image_sequence_id_get)
    __swig_getmethods__["id"] = LeapPython.Image_id_get
    if _newclass:
        id = _swig_property(LeapPython.Image_id_get)
    __swig_getmethods__["width"] = LeapPython.Image_width_get
    if _newclass:
        width = _swig_property(LeapPython.Image_width_get)
    __swig_getmethods__["height"] = LeapPython.Image_height_get
    if _newclass:
        height = _swig_property(LeapPython.Image_height_get)
    __swig_getmethods__["bytes_per_pixel"] = LeapPython.Image_bytes_per_pixel_get
    if _newclass:
        bytes_per_pixel = _swig_property(LeapPython.Image_bytes_per_pixel_get)
    __swig_getmethods__["format"] = LeapPython.Image_format_get
    if _newclass:
        format = _swig_property(LeapPython.Image_format_get)
    __swig_getmethods__["distortion_width"] = LeapPython.Image_distortion_width_get
    if _newclass:
        distortion_width = _swig_property(LeapPython.Image_distortion_width_get)
    __swig_getmethods__["distortion_height"] = LeapPython.Image_distortion_height_get
    if _newclass:
        distortion_height = _swig_property(LeapPython.Image_distortion_height_get)
    __swig_getmethods__["ray_offset_x"] = LeapPython.Image_ray_offset_x_get
    if _newclass:
        ray_offset_x = _swig_property(LeapPython.Image_ray_offset_x_get)
    __swig_getmethods__["ray_offset_y"] = LeapPython.Image_ray_offset_y_get
    if _newclass:
        ray_offset_y = _swig_property(LeapPython.Image_ray_offset_y_get)
    __swig_getmethods__["ray_scale_x"] = LeapPython.Image_ray_scale_x_get
    if _newclass:
        ray_scale_x = _swig_property(LeapPython.Image_ray_scale_x_get)
    __swig_getmethods__["ray_scale_y"] = LeapPython.Image_ray_scale_y_get
    if _newclass:
        ray_scale_y = _swig_property(LeapPython.Image_ray_scale_y_get)
    __swig_getmethods__["timestamp"] = LeapPython.Image_timestamp_get
    if _newclass:
        timestamp = _swig_property(LeapPython.Image_timestamp_get)
    __swig_getmethods__["is_valid"] = LeapPython.Image_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Image_is_valid_get)
    def data(self):
        ptr = byte_array(self.width * self.height * self.bytes_per_pixel)
        LeapPython.Image_data(self, ptr)
        return ptr
    def distortion(self):
        ptr = float_array(self.distortion_width * self.distortion_height)
        LeapPython.Image_distortion(self, ptr)
        return ptr
    __swig_getmethods__["data"] = data
    if _newclass:data = _swig_property(data)
    __swig_getmethods__["distortion"] = distortion
    if _newclass:distortion = _swig_property(distortion)

    __swig_getmethods__["data_pointer"] = LeapPython.Image_data_pointer_get
    if _newclass:
        data_pointer = _swig_property(LeapPython.Image_data_pointer_get)
    __swig_getmethods__["distortion_pointer"] = LeapPython.Image_distortion_pointer_get
    if _newclass:
        distortion_pointer = _swig_property(LeapPython.Image_distortion_pointer_get)
    __swig_destroy__ = LeapPython.delete_Image
    __del__ = lambda self: None
Image_swigregister = LeapPython.Image_swigregister
Image_swigregister(Image)
Image.invalid = LeapPython.cvar.Image_invalid

class PointableList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, PointableList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, PointableList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_PointableList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.PointableList___len__(self)

    def __getitem__(self, index):
        return LeapPython.PointableList___getitem__(self, index)

    def append(self, *args):
        return LeapPython.PointableList_append(self, *args)

    def extended(self):
        return LeapPython.PointableList_extended(self)
    __swig_getmethods__["is_empty"] = LeapPython.PointableList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.PointableList_is_empty_get)
    __swig_getmethods__["leftmost"] = LeapPython.PointableList_leftmost_get
    if _newclass:
        leftmost = _swig_property(LeapPython.PointableList_leftmost_get)
    __swig_getmethods__["rightmost"] = LeapPython.PointableList_rightmost_get
    if _newclass:
        rightmost = _swig_property(LeapPython.PointableList_rightmost_get)
    __swig_getmethods__["frontmost"] = LeapPython.PointableList_frontmost_get
    if _newclass:
        frontmost = _swig_property(LeapPython.PointableList_frontmost_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_PointableList
    __del__ = lambda self: None
PointableList_swigregister = LeapPython.PointableList_swigregister
PointableList_swigregister(PointableList)

class FingerList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FingerList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, FingerList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_FingerList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.FingerList___len__(self)

    def __getitem__(self, index):
        return LeapPython.FingerList___getitem__(self, index)

    def append(self, other):
        return LeapPython.FingerList_append(self, other)

    def extended(self):
        return LeapPython.FingerList_extended(self)

    def finger_type(self, type):
        return LeapPython.FingerList_finger_type(self, type)
    __swig_getmethods__["is_empty"] = LeapPython.FingerList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.FingerList_is_empty_get)
    __swig_getmethods__["leftmost"] = LeapPython.FingerList_leftmost_get
    if _newclass:
        leftmost = _swig_property(LeapPython.FingerList_leftmost_get)
    __swig_getmethods__["rightmost"] = LeapPython.FingerList_rightmost_get
    if _newclass:
        rightmost = _swig_property(LeapPython.FingerList_rightmost_get)
    __swig_getmethods__["frontmost"] = LeapPython.FingerList_frontmost_get
    if _newclass:
        frontmost = _swig_property(LeapPython.FingerList_frontmost_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_FingerList
    __del__ = lambda self: None
FingerList_swigregister = LeapPython.FingerList_swigregister
FingerList_swigregister(FingerList)

class ToolList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ToolList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ToolList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_ToolList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.ToolList___len__(self)

    def __getitem__(self, index):
        return LeapPython.ToolList___getitem__(self, index)

    def append(self, other):
        return LeapPython.ToolList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.ToolList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.ToolList_is_empty_get)
    __swig_getmethods__["leftmost"] = LeapPython.ToolList_leftmost_get
    if _newclass:
        leftmost = _swig_property(LeapPython.ToolList_leftmost_get)
    __swig_getmethods__["rightmost"] = LeapPython.ToolList_rightmost_get
    if _newclass:
        rightmost = _swig_property(LeapPython.ToolList_rightmost_get)
    __swig_getmethods__["frontmost"] = LeapPython.ToolList_frontmost_get
    if _newclass:
        frontmost = _swig_property(LeapPython.ToolList_frontmost_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_ToolList
    __del__ = lambda self: None
ToolList_swigregister = LeapPython.ToolList_swigregister
ToolList_swigregister(ToolList)

class HandList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, HandList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, HandList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_HandList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.HandList___len__(self)

    def __getitem__(self, index):
        return LeapPython.HandList___getitem__(self, index)

    def append(self, other):
        return LeapPython.HandList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.HandList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.HandList_is_empty_get)
    __swig_getmethods__["leftmost"] = LeapPython.HandList_leftmost_get
    if _newclass:
        leftmost = _swig_property(LeapPython.HandList_leftmost_get)
    __swig_getmethods__["rightmost"] = LeapPython.HandList_rightmost_get
    if _newclass:
        rightmost = _swig_property(LeapPython.HandList_rightmost_get)
    __swig_getmethods__["frontmost"] = LeapPython.HandList_frontmost_get
    if _newclass:
        frontmost = _swig_property(LeapPython.HandList_frontmost_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_HandList
    __del__ = lambda self: None
HandList_swigregister = LeapPython.HandList_swigregister
HandList_swigregister(HandList)

class GestureList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, GestureList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, GestureList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_GestureList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.GestureList___len__(self)

    def __getitem__(self, index):
        return LeapPython.GestureList___getitem__(self, index)

    def append(self, other):
        return LeapPython.GestureList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.GestureList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.GestureList_is_empty_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_GestureList
    __del__ = lambda self: None
GestureList_swigregister = LeapPython.GestureList_swigregister
GestureList_swigregister(GestureList)

class DeviceList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, DeviceList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, DeviceList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_DeviceList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.DeviceList___len__(self)

    def __getitem__(self, index):
        return LeapPython.DeviceList___getitem__(self, index)

    def append(self, other):
        return LeapPython.DeviceList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.DeviceList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.DeviceList_is_empty_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_DeviceList
    __del__ = lambda self: None
DeviceList_swigregister = LeapPython.DeviceList_swigregister
DeviceList_swigregister(DeviceList)

class FailedDeviceList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, FailedDeviceList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, FailedDeviceList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_FailedDeviceList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.FailedDeviceList___len__(self)

    def __getitem__(self, index):
        return LeapPython.FailedDeviceList___getitem__(self, index)

    def append(self, other):
        return LeapPython.FailedDeviceList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.FailedDeviceList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.FailedDeviceList_is_empty_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_FailedDeviceList
    __del__ = lambda self: None
FailedDeviceList_swigregister = LeapPython.FailedDeviceList_swigregister
FailedDeviceList_swigregister(FailedDeviceList)

class ImageList(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ImageList, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ImageList, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_ImageList()
        try:
            self.this.append(this)
        except:
            self.this = this

    def __len__(self):
        return LeapPython.ImageList___len__(self)

    def __getitem__(self, index):
        return LeapPython.ImageList___getitem__(self, index)

    def append(self, other):
        return LeapPython.ImageList_append(self, other)
    __swig_getmethods__["is_empty"] = LeapPython.ImageList_is_empty_get
    if _newclass:
        is_empty = _swig_property(LeapPython.ImageList_is_empty_get)
    def __iter__(self):
      _pos = 0
      while _pos < len(self):
        yield self[_pos]
        _pos += 1

    __swig_destroy__ = LeapPython.delete_ImageList
    __del__ = lambda self: None
ImageList_swigregister = LeapPython.ImageList_swigregister
ImageList_swigregister(ImageList)

class InteractionBox(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, InteractionBox, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, InteractionBox, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_InteractionBox()
        try:
            self.this.append(this)
        except:
            self.this = this

    def normalize_point(self, position, clamp=True):
        return LeapPython.InteractionBox_normalize_point(self, position, clamp)

    def denormalize_point(self, normalizedPosition):
        return LeapPython.InteractionBox_denormalize_point(self, normalizedPosition)

    def __eq__(self, arg2):
        return LeapPython.InteractionBox___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.InteractionBox___ne__(self, arg2)

    def __str__(self):
        return LeapPython.InteractionBox___str__(self)
    __swig_getmethods__["center"] = LeapPython.InteractionBox_center_get
    if _newclass:
        center = _swig_property(LeapPython.InteractionBox_center_get)
    __swig_getmethods__["width"] = LeapPython.InteractionBox_width_get
    if _newclass:
        width = _swig_property(LeapPython.InteractionBox_width_get)
    __swig_getmethods__["height"] = LeapPython.InteractionBox_height_get
    if _newclass:
        height = _swig_property(LeapPython.InteractionBox_height_get)
    __swig_getmethods__["depth"] = LeapPython.InteractionBox_depth_get
    if _newclass:
        depth = _swig_property(LeapPython.InteractionBox_depth_get)
    __swig_getmethods__["is_valid"] = LeapPython.InteractionBox_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.InteractionBox_is_valid_get)
    __swig_destroy__ = LeapPython.delete_InteractionBox
    __del__ = lambda self: None
InteractionBox_swigregister = LeapPython.InteractionBox_swigregister
InteractionBox_swigregister(InteractionBox)
InteractionBox.invalid = LeapPython.cvar.InteractionBox_invalid

class Frame(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Frame, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Frame, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_Frame()
        try:
            self.this.append(this)
        except:
            self.this = this

    def hand(self, id):
        return LeapPython.Frame_hand(self, id)

    def pointable(self, id):
        return LeapPython.Frame_pointable(self, id)

    def finger(self, id):
        return LeapPython.Frame_finger(self, id)

    def tool(self, id):
        return LeapPython.Frame_tool(self, id)

    def gesture(self, id):
        return LeapPython.Frame_gesture(self, id)

    def gestures(self, *args):
        return LeapPython.Frame_gestures(self, *args)

    def translation(self, sinceFrame):
        return LeapPython.Frame_translation(self, sinceFrame)

    def translation_probability(self, sinceFrame):
        return LeapPython.Frame_translation_probability(self, sinceFrame)

    def rotation_axis(self, sinceFrame):
        return LeapPython.Frame_rotation_axis(self, sinceFrame)

    def rotation_angle(self, *args):
        return LeapPython.Frame_rotation_angle(self, *args)

    def rotation_matrix(self, sinceFrame):
        return LeapPython.Frame_rotation_matrix(self, sinceFrame)

    def rotation_probability(self, sinceFrame):
        return LeapPython.Frame_rotation_probability(self, sinceFrame)

    def scale_factor(self, sinceFrame):
        return LeapPython.Frame_scale_factor(self, sinceFrame)

    def scale_probability(self, sinceFrame):
        return LeapPython.Frame_scale_probability(self, sinceFrame)

    def __eq__(self, arg2):
        return LeapPython.Frame___eq__(self, arg2)

    def __ne__(self, arg2):
        return LeapPython.Frame___ne__(self, arg2)

    def serialize(self, ptr):
        return LeapPython.Frame_serialize(self, ptr)

    def deserialize(self, ptr, length):
        return LeapPython.Frame_deserialize(self, ptr, length)

    def __str__(self):
        return LeapPython.Frame___str__(self)
    __swig_getmethods__["id"] = LeapPython.Frame_id_get
    if _newclass:
        id = _swig_property(LeapPython.Frame_id_get)
    __swig_getmethods__["timestamp"] = LeapPython.Frame_timestamp_get
    if _newclass:
        timestamp = _swig_property(LeapPython.Frame_timestamp_get)
    __swig_getmethods__["current_frames_per_second"] = LeapPython.Frame_current_frames_per_second_get
    if _newclass:
        current_frames_per_second = _swig_property(LeapPython.Frame_current_frames_per_second_get)
    __swig_getmethods__["pointables"] = LeapPython.Frame_pointables_get
    if _newclass:
        pointables = _swig_property(LeapPython.Frame_pointables_get)
    __swig_getmethods__["fingers"] = LeapPython.Frame_fingers_get
    if _newclass:
        fingers = _swig_property(LeapPython.Frame_fingers_get)
    __swig_getmethods__["tools"] = LeapPython.Frame_tools_get
    if _newclass:
        tools = _swig_property(LeapPython.Frame_tools_get)
    __swig_getmethods__["hands"] = LeapPython.Frame_hands_get
    if _newclass:
        hands = _swig_property(LeapPython.Frame_hands_get)
    __swig_getmethods__["images"] = LeapPython.Frame_images_get
    if _newclass:
        images = _swig_property(LeapPython.Frame_images_get)
    __swig_getmethods__["raw_images"] = LeapPython.Frame_raw_images_get
    if _newclass:
        raw_images = _swig_property(LeapPython.Frame_raw_images_get)
    __swig_getmethods__["is_valid"] = LeapPython.Frame_is_valid_get
    if _newclass:
        is_valid = _swig_property(LeapPython.Frame_is_valid_get)
    __swig_getmethods__["interaction_box"] = LeapPython.Frame_interaction_box_get
    if _newclass:
        interaction_box = _swig_property(LeapPython.Frame_interaction_box_get)
    __swig_getmethods__["serialize_length"] = LeapPython.Frame_serialize_length_get
    if _newclass:
        serialize_length = _swig_property(LeapPython.Frame_serialize_length_get)
    def serialize(self):
        length = self.serialize_length
        str = byte_array(length)
        LeapPython.Frame_serialize(self, str)
        return (str, length)
    def deserialize(self, tup):
        LeapPython.Frame_deserialize(self, tup[0], tup[1])
    __swig_getmethods__["serialize"] = serialize
    if _newclass:serialize = _swig_property(serialize)

    __swig_destroy__ = LeapPython.delete_Frame
    __del__ = lambda self: None
Frame_swigregister = LeapPython.Frame_swigregister
Frame_swigregister(Frame)
Frame.invalid = LeapPython.cvar.Frame_invalid

class BugReport(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, BugReport, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, BugReport, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_BugReport()
        try:
            self.this.append(this)
        except:
            self.this = this

    def begin_recording(self):
        return LeapPython.BugReport_begin_recording(self)

    def end_recording(self):
        return LeapPython.BugReport_end_recording(self)
    __swig_getmethods__["is_active"] = LeapPython.BugReport_is_active_get
    if _newclass:
        is_active = _swig_property(LeapPython.BugReport_is_active_get)
    __swig_getmethods__["progress"] = LeapPython.BugReport_progress_get
    if _newclass:
        progress = _swig_property(LeapPython.BugReport_progress_get)
    __swig_getmethods__["duration"] = LeapPython.BugReport_duration_get
    if _newclass:
        duration = _swig_property(LeapPython.BugReport_duration_get)
    __swig_destroy__ = LeapPython.delete_BugReport
    __del__ = lambda self: None
BugReport_swigregister = LeapPython.BugReport_swigregister
BugReport_swigregister(BugReport)

class Config(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Config, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Config, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = LeapPython.new_Config()
        try:
            self.this.append(this)
        except:
            self.this = this
    TYPE_UNKNOWN = LeapPython.Config_TYPE_UNKNOWN
    TYPE_BOOLEAN = LeapPython.Config_TYPE_BOOLEAN
    TYPE_INT32 = LeapPython.Config_TYPE_INT32
    TYPE_FLOAT = LeapPython.Config_TYPE_FLOAT
    TYPE_STRING = LeapPython.Config_TYPE_STRING










    def save(self):
        return LeapPython.Config_save(self)
    def get(self, *args):
      type = LeapPython.Config_type(self, *args)
      if type == LeapPython.Config_TYPE_BOOLEAN:
        return LeapPython.Config_get_bool(self, *args)
      elif type == LeapPython.Config_TYPE_INT32:
        return LeapPython.Config_get_int_32(self, *args)
      elif type == LeapPython.Config_TYPE_FLOAT:
        return LeapPython.Config_get_float(self, *args)
      elif type == LeapPython.Config_TYPE_STRING:
        return LeapPython.Config_get_string(self, *args)
      return None
    def set(self, *args):
      type = LeapPython.Config_type(self, *args[:-1])  # Do not pass value through
      if type == LeapPython.Config_TYPE_BOOLEAN:
        return LeapPython.Config_set_bool(self, *args)
      elif type == LeapPython.Config_TYPE_INT32:
        return LeapPython.Config_set_int_32(self, *args)
      elif type == LeapPython.Config_TYPE_FLOAT:
        return LeapPython.Config_set_float(self, *args)
      elif type == LeapPython.Config_TYPE_STRING:
        return LeapPython.Config_set_string(self, *args)
      return False

    __swig_destroy__ = LeapPython.delete_Config
    __del__ = lambda self: None
Config_swigregister = LeapPython.Config_swigregister
Config_swigregister(Config)

class Controller(Interface):
    __swig_setmethods__ = {}
    for _s in [Interface]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, Controller, name, value)
    __swig_getmethods__ = {}
    for _s in [Interface]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, Controller, name)
    __repr__ = _swig_repr
    __swig_destroy__ = LeapPython.delete_Controller
    __del__ = lambda self: None

    def __init__(self, *args):
        this = LeapPython.new_Controller(*args)
        try:
            self.this.append(this)
        except:
            self.this = this

    def is_service_connected(self):
        return LeapPython.Controller_is_service_connected(self)
    POLICY_DEFAULT = LeapPython.Controller_POLICY_DEFAULT
    POLICY_BACKGROUND_FRAMES = LeapPython.Controller_POLICY_BACKGROUND_FRAMES
    POLICY_IMAGES = LeapPython.Controller_POLICY_IMAGES
    POLICY_OPTIMIZE_HMD = LeapPython.Controller_POLICY_OPTIMIZE_HMD
    POLICY_ALLOW_PAUSE_RESUME = LeapPython.Controller_POLICY_ALLOW_PAUSE_RESUME
    POLICY_RAW_IMAGES = LeapPython.Controller_POLICY_RAW_IMAGES

    def set_policy_flags(self, flags):
        return LeapPython.Controller_set_policy_flags(self, flags)

    def set_policy(self, policy):
        return LeapPython.Controller_set_policy(self, policy)

    def clear_policy(self, policy):
        return LeapPython.Controller_clear_policy(self, policy)

    def is_policy_set(self, policy):
        return LeapPython.Controller_is_policy_set(self, policy)

    def add_listener(self, listener):
        return LeapPython.Controller_add_listener(self, listener)

    def remove_listener(self, listener):
        return LeapPython.Controller_remove_listener(self, listener)

    def frame(self, history=0):
        return LeapPython.Controller_frame(self, history)

    def failed_devices(self):
        return LeapPython.Controller_failed_devices(self)

    def enable_gesture(self, type, enable=True):
        return LeapPython.Controller_enable_gesture(self, type, enable)

    def is_gesture_enabled(self, type):
        return LeapPython.Controller_is_gesture_enabled(self, type)

    def set_paused(self, pause):
        return LeapPython.Controller_set_paused(self, pause)

    def is_paused(self):
        return LeapPython.Controller_is_paused(self)

    def now(self):
        return LeapPython.Controller_now(self)
    __swig_getmethods__["is_connected"] = LeapPython.Controller_is_connected_get
    if _newclass:
        is_connected = _swig_property(LeapPython.Controller_is_connected_get)
    __swig_getmethods__["has_focus"] = LeapPython.Controller_has_focus_get
    if _newclass:
        has_focus = _swig_property(LeapPython.Controller_has_focus_get)
    __swig_getmethods__["policy_flags"] = LeapPython.Controller_policy_flags_get
    if _newclass:
        policy_flags = _swig_property(LeapPython.Controller_policy_flags_get)
    __swig_getmethods__["config"] = LeapPython.Controller_config_get
    if _newclass:
        config = _swig_property(LeapPython.Controller_config_get)
    __swig_getmethods__["images"] = LeapPython.Controller_images_get
    if _newclass:
        images = _swig_property(LeapPython.Controller_images_get)
    __swig_getmethods__["raw_images"] = LeapPython.Controller_raw_images_get
    if _newclass:
        raw_images = _swig_property(LeapPython.Controller_raw_images_get)
    __swig_getmethods__["devices"] = LeapPython.Controller_devices_get
    if _newclass:
        devices = _swig_property(LeapPython.Controller_devices_get)
    __swig_getmethods__["bug_report"] = LeapPython.Controller_bug_report_get
    if _newclass:
        bug_report = _swig_property(LeapPython.Controller_bug_report_get)
Controller_swigregister = LeapPython.Controller_swigregister
Controller_swigregister(Controller)

MESSAGE_UNKNOWN = LeapPython.MESSAGE_UNKNOWN
MESSAGE_CRITICAL = LeapPython.MESSAGE_CRITICAL
MESSAGE_WARNING = LeapPython.MESSAGE_WARNING
MESSAGE_INFORMATION = LeapPython.MESSAGE_INFORMATION
class Listener(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Listener, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Listener, name)
    __repr__ = _swig_repr

    def __init__(self):
        if self.__class__ == Listener:
            _self = None
        else:
            _self = self
        this = LeapPython.new_Listener(_self, )
        try:
            self.this.append(this)
        except:
            self.this = this
    __swig_destroy__ = LeapPython.delete_Listener
    __del__ = lambda self: None

    def on_init(self, arg0):
        return LeapPython.Listener_on_init(self, arg0)

    def on_connect(self, arg0):
        return LeapPython.Listener_on_connect(self, arg0)

    def on_disconnect(self, arg0):
        return LeapPython.Listener_on_disconnect(self, arg0)

    def on_exit(self, arg0):
        return LeapPython.Listener_on_exit(self, arg0)

    def on_frame(self, arg0):
        return LeapPython.Listener_on_frame(self, arg0)

    def on_focus_gained(self, arg0):
        return LeapPython.Listener_on_focus_gained(self, arg0)

    def on_focus_lost(self, arg0):
        return LeapPython.Listener_on_focus_lost(self, arg0)

    def on_service_connect(self, arg0):
        return LeapPython.Listener_on_service_connect(self, arg0)

    def on_service_disconnect(self, arg0):
        return LeapPython.Listener_on_service_disconnect(self, arg0)

    def on_device_change(self, arg0):
        return LeapPython.Listener_on_device_change(self, arg0)

    def on_images(self, arg0):
        return LeapPython.Listener_on_images(self, arg0)

    def on_service_change(self, arg0):
        return LeapPython.Listener_on_service_change(self, arg0)

    def on_device_failure(self, arg0):
        return LeapPython.Listener_on_device_failure(self, arg0)

    def on_log_message(self, arg0, severity, timestamp, msg):
        return LeapPython.Listener_on_log_message(self, arg0, severity, timestamp, msg)
    def __disown__(self):
        self.this.disown()
        LeapPython.disown_Listener(self)
        return weakref_proxy(self)
Listener_swigregister = LeapPython.Listener_swigregister
Listener_swigregister(Listener)
