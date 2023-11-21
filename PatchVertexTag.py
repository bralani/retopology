from enum import Enum

class PatchVertexTag(Enum):
    NoneTag = -1
    C0 = 0
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    V0 = 6
    V1 = 7
    V2 = 8
    V3 = 9
    V4 = 10
    V5 = 11
    V6 = 12
    V7 = 13
    V8 = 14


class PatchT:
    class VHandle:
        pass

    class Point:
        pass

    def add_vertex(self, point):
        pass

    def data(self, vhandle):
        pass

    def vertices(self):
        pass


def add_tagged_vertex(patch, index, is_corner):
    v = patch.add_vertex(patch.Point())
    vdata = patch.data(v)
    vdata.patchgen.corner_index = index if is_corner else -1
    tag = PatchVertexTag.C0 if is_corner else PatchVertexTag.V0
    vdata.patchgen.tag = tag.value + index if is_corner else tag.value
    return v


def find_tagged_vertex(patch, tag):
    for v in patch.vertices():
        if patch.data(v).patchgen.tag == tag.value:
            return v
    return patch.VHandle() # not sure