"""VTU (VTK XML UnstructuredGrid) の読み書き。numpy だけを使う。

読み込みは、1 つの Piece を持つファイルの次の形式に対応する。
  - appended（raw または base64）、inline の binary（base64）、ascii
  - 非圧縮、または zlib 圧縮（vtkZLibDataCompressor）
  - ヘッダ型 UInt32 / UInt64、リトルエンディアン・ビッグエンディアン
書き出しは、非圧縮の raw appended（UInt64 ヘッダ、リトルエンディアン）だけを行う。
"""
import base64
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TYPES = {"Float32": "f4", "Float64": "f8", "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2",
         "Int32": "i4", "UInt32": "u4", "Int64": "i8", "UInt64": "u8"}
NAMES = {np.dtype(v).str[1:]: k for k, v in TYPES.items()}
VTK_TRIANGLE, VTK_QUAD = 5, 9


@dataclass
class Mesh:
    points: np.ndarray  # (N, 3) float64
    connectivity: np.ndarray  # 全セルの節点番号を並べたもの
    offsets: np.ndarray  # 各セルの終端位置（connectivity 内）
    types: np.ndarray  # VTK のセル種別
    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)

    def triangles(self):
        """三角形セルと、四角形を 2 つに割った三角形を返す。(三角形 (M, 3), 元のセル番号 (M,))。"""
        start = np.concatenate([[0], self.offsets[:-1]])
        t = np.flatnonzero(self.types == VTK_TRIANGLE)
        q = np.flatnonzero(self.types == VTK_QUAD)
        c = self.connectivity
        tri = c[start[t][:, None] + np.arange(3)]
        quad = c[start[q][:, None] + np.arange(4)]
        tris = np.concatenate([tri, quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
        return tris.astype(np.int64), np.concatenate([t, q, q])


def _b64len(nbytes):
    return 4 * ((nbytes + 2) // 3)


class _Reader:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        cut = raw.find(b"<AppendedData")
        self.appended = None
        if cut >= 0:
            close = raw.index(b">", cut)
            self.appended_raw = b'encoding="raw"' in raw[cut:close] or b"encoding='raw'" in raw[cut:close]
            begin = raw.index(b"_", close) + 1
            self.appended = raw[begin:]
            raw = raw[:cut] + b"</VTKFile>"
        self.root = ET.fromstring(raw)
        if self.root.get("type") != "UnstructuredGrid":
            raise ValueError(f"{path}: UnstructuredGrid ではない（type={self.root.get('type')}）")
        self.order = "<" if self.root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
        self.header = np.dtype(self.order + TYPES[self.root.get("header_type", "UInt32")])
        comp = self.root.get("compressor")
        if comp not in (None, "vtkZLibDataCompressor"):
            raise ValueError(f"{path}: 圧縮形式 {comp} には対応していない（非圧縮か zlib で書き出す）")
        self.zlib = comp is not None

    def _ints(self, buf, n):
        return np.frombuffer(buf[:n * self.header.itemsize], self.header, n).astype(np.int64)

    def _raw_block(self, buf):
        """先頭にヘッダを持つバイト列から、データ本体を取り出す。"""
        h = self.header.itemsize
        if not self.zlib:
            return buf[h:h + int(self._ints(buf, 1)[0])]
        nblocks = int(self._ints(buf, 1)[0])
        sizes = self._ints(buf[3 * h:], nblocks)
        pos, out = (3 + nblocks) * h, []
        for s in sizes:
            out.append(zlib.decompress(buf[pos:pos + int(s)]))
            pos += int(s)
        return b"".join(out)

    def _b64_block(self, text):
        """base64 文字列から、データ本体を取り出す。ヘッダと本体は別々に符号化されていることがある。"""
        h = self.header.itemsize
        if self.zlib:
            nblocks = int(self._ints(base64.b64decode(text[:_b64len(3 * h)]), 1)[0])
            hlen = _b64len((3 + nblocks) * h)
            head = base64.b64decode(text[:hlen])
            sizes = self._ints(head[3 * h:], nblocks)
            body = base64.b64decode(text[hlen:hlen + _b64len(int(sizes.sum()))])
            pos, out = 0, []
            for s in sizes:
                out.append(zlib.decompress(body[pos:pos + int(s)]))
                pos += int(s)
            return b"".join(out)
        first = text[:_b64len(h)]
        nbytes = int(self._ints(base64.b64decode(first), 1)[0])
        if nbytes and first.endswith(b"="):  # ヘッダだけで符号化が閉じている（本体は別に符号化）
            return base64.b64decode(text[_b64len(h):_b64len(h) + _b64len(nbytes)])
        return base64.b64decode(text[:_b64len(h + nbytes)])[h:h + nbytes]  # ヘッダと本体をまとめて符号化

    def array(self, el):
        dtype = np.dtype(self.order + TYPES[el.get("type")])
        fmt = el.get("format", "ascii")
        if fmt == "ascii":
            values = np.array((el.text or "").split(), dtype=dtype.newbyteorder("="))
        else:
            if fmt == "appended":
                if self.appended is None:
                    raise ValueError("appended のデータがない")
                buf = self.appended[int(el.get("offset")):]
                data = self._raw_block(buf) if self.appended_raw else self._b64_block(buf)
            else:
                data = self._b64_block("".join((el.text or "").split()).encode())
            values = np.frombuffer(data, dtype)
        comps = int(el.get("NumberOfComponents", "1").strip())
        values = values.astype(dtype.newbyteorder("="))
        return values.reshape(-1, comps) if comps > 1 else values


def read_vtu(path):
    r = _Reader(path)
    pieces = r.root.findall("./UnstructuredGrid/Piece")
    if len(pieces) != 1:
        raise ValueError(f"{path}: Piece が {len(pieces)} 個ある（1 個のファイルだけに対応）")
    piece = pieces[0]
    n_points, n_cells = int(piece.get("NumberOfPoints")), int(piece.get("NumberOfCells"))
    points = r.array(piece.find("./Points/DataArray")).astype(np.float64).reshape(-1, 3)
    cells = {el.get("Name"): r.array(el) for el in piece.findall("./Cells/DataArray")}

    def data(tag):
        node = piece.find(tag)
        return {} if node is None else {el.get("Name"): r.array(el) for el in node.findall("DataArray")}

    mesh = Mesh(points, cells["connectivity"].astype(np.int64), cells["offsets"].astype(np.int64),
                cells["types"].astype(np.int64), data("PointData"), data("CellData"))
    if len(points) != n_points or len(mesh.types) != n_cells:
        raise ValueError(f"{path}: 節点数またはセル数がヘッダと合わない")
    return mesh


def write_vtu(path, mesh):
    """非圧縮の raw appended で書き出す。ParaView でそのまま開ける。"""
    blocks, offset = [], 0

    def tag(name, arr, indent):
        nonlocal offset
        arr = np.ascontiguousarray(arr)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        comps = arr.shape[1] if arr.ndim > 1 else 1
        raw = arr.tobytes()
        line = (f'{indent}<DataArray type="{NAMES[arr.dtype.str[1:]]}" Name="{name}" NumberOfComponents="{comps}" '
                f'format="appended" offset="{offset}"/>')
        blocks.append(np.uint64(len(raw)).tobytes() + raw)
        offset += 8 + len(raw)
        return line

    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           " <UnstructuredGrid>",
           f'  <Piece NumberOfPoints="{len(mesh.points)}" NumberOfCells="{len(mesh.types)}">',
           "   <Points>", tag("Points", mesh.points.astype(np.float64), "    "), "   </Points>",
           "   <Cells>", tag("connectivity", mesh.connectivity.astype(np.int64), "    "),
           tag("offsets", mesh.offsets.astype(np.int64), "    "),
           tag("types", mesh.types.astype(np.uint8), "    "), "   </Cells>"]
    for section, arrays in (("PointData", mesh.point_data), ("CellData", mesh.cell_data)):
        if arrays:
            out.append(f"   <{section}>")
            out += [tag(name, arr, "    ") for name, arr in arrays.items()]
            out.append(f"   </{section}>")
    out += ["  </Piece>", " </UnstructuredGrid>", ' <AppendedData encoding="raw">', "_"]
    with open(path, "wb") as f:
        f.write("\n".join(out).encode())
        f.write(b"".join(blocks))
        f.write(b"\n </AppendedData>\n</VTKFile>\n")


def triangle_mesh(points, tri, point_data=None, cell_data=None):
    tri = np.asarray(tri)
    return Mesh(np.asarray(points, float), tri.ravel(), 3 * np.arange(1, len(tri) + 1),
                np.full(len(tri), VTK_TRIANGLE), dict(point_data or {}), dict(cell_data or {}))
