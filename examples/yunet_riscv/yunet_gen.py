#!/usr/bin/env python3
"""Generate a compact, fast, portable C implementation of a block-quantized
(weight-only INT8) YuNet-style ONNX model.

Usage: yunet_gen.py model.onnx out.c

The generated C keeps weights as INT8 (+ per-block scale / zero point) inside a
base64 string, dequantizes them ONCE on first call, and runs vectorizable
float kernels (auto-vectorized, explicit RVV intrinsics when __riscv_vector is
defined, OpenMP when compiled with -fopenmp).  entry() keeps onnx2c's
signature so it is a drop-in replacement.
"""
import sys, os, base64, collections
import numpy as np, onnx
from onnx import numpy_helper as nh, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)

# ---- constants: initializers, DequantizeLinear records, reshapes of those ----
const = {t.name: ("f", nh.to_array(t)) for t in m.graph.initializer}
for n in m.graph.node:
    if n.op_type == "DequantizeLinear":
        a = {x.name: x.i for x in n.attribute}
        q, s = const[n.input[0]][1], const[n.input[1]][1]
        z = const[n.input[2]][1]
        assert q.dtype == np.int8 and z.dtype == np.int8
        bs = a["block_size"]; assert a["axis"] == q.ndim - 1
        q2 = q.reshape(-1, q.shape[-1])
        assert s.reshape(q2.shape[0], -1).shape[1] == -(-q2.shape[1] // bs)
        const[n.output[0]] = ("dq", dict(q=q2, s=s.astype(np.float32).ravel(),
                                         z=z.ravel(), bs=bs, shape=list(q.shape)))
    elif n.op_type == "Reshape" and n.input[0] in const and const[n.input[0]][0] == "dq":
        r = dict(const[n.input[0]][1]); r["shape"] = [int(v) for v in const[n.input[1]][1]]
        const[n.output[0]] = ("dq", r)

# ---- activation shapes ----
mi = shape_inference.infer_shapes(m)
SH = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
      for v in list(mi.graph.value_info) + list(mi.graph.output) + list(mi.graph.input)}

blob = bytearray()


def put_const(name):
    """Append constant to the blob; return (element count, dq info or None)."""
    kind, v = const[name]
    if kind == "f":
        a = np.ascontiguousarray(v.astype(np.float32)); blob.extend(a.tobytes())
        return a.size, None
    q, s, z = v["q"], v["s"], v["z"]
    blob.extend(q.astype(np.int8).tobytes()); blob.extend(z.astype(np.int8).tobytes())
    blob.extend(s.tobytes())
    return q.size, (q.shape[0], q.shape[1], v["bs"])


# ---- walk the graph ----
nodes = list(m.graph.node)
consumers = collections.defaultdict(list)
for i, n in enumerate(nodes):
    for x in n.input:
        consumers[x].append(i)
alias = {}          # relu-fused tensor aliasing
def A(x): return alias.get(x, x)

loads = []          # weight loads in blob order: (varname, count, dqinfo, transform)
wvars = {}          # (const name, transform) -> C variable
def weight(name, transform=None):
    if (name, transform) in wvars: return wvars[(name, transform)]
    cnt, dq = put_const(name)
    var = "w%d" % len(wvars)
    wvars[(name, transform)] = var
    loads.append((var, cnt, dq, transform))
    return var

inp = m.graph.input[0].name
outs = [o.name for o in m.graph.output]
steps = []          # (op, inputs[list], output, params)
for i, n in enumerate(nodes):
    t = n.op_type
    if t == "DequantizeLinear" or (t == "Reshape" and n.output[0] in const):
        continue
    at = {x.name: (list(x.ints) if x.ints else x.i) for x in n.attribute if x.type in (2, 7)}
    if t == "Conv":
        x, w, b = n.input; y = n.output[0]
        relu = False
        c = consumers[y]
        if len(c) == 1 and nodes[c[0]].op_type == "Relu":
            relu = True; alias[nodes[c[0]].output[0]] = y
        ci, h, wd = SH[A(x)][1:]; co = SH[y][1]; k = at["kernel_shape"][0]; g = at.get("group", 1)
        if k == 1:
            assert g == 1 and at["strides"] == [1, 1]
            steps.append(("pw", [A(x)], y, dict(w=weight(w), b=weight(b), ci=ci, co=co, hw=h * wd, relu=int(relu))))
        elif g == 1 and ci == 3 and co == 16 and at["strides"] == [2, 2]:
            steps.append(("c3s2", [A(x)], y, dict(w=weight(w, "t16"), b=weight(b), h=h, w_=wd, relu=int(relu))))
        else:
            assert at["strides"] == [1, 1] and (g == ci == co or (ci == co == 1))
            steps.append(("dw", [A(x)], y, dict(w=weight(w), b=weight(b), c=ci, h=h, w_=wd, relu=int(relu))))
    elif t == "Relu":
        if n.output[0] in alias: continue
        raise SystemExit("unfused Relu")
    elif t == "MaxPool":
        assert at["kernel_shape"] == [2, 2] and at["strides"] == [2, 2]
        c, h, wd = SH[A(n.input[0])][1:]
        steps.append(("mp", [A(n.input[0])], n.output[0], dict(c=c, h=h, w_=wd)))
    elif t == "Resize":
        assert list(const[n.input[2]][1]) == [1, 1, 2, 2]
        c, h, wd = SH[A(n.input[0])][1:]
        steps.append(("up", [A(n.input[0])], n.output[0], dict(c=c, h=h, w_=wd)))
    elif t == "Add":
        sz = int(np.prod(SH[A(n.input[0])]))
        steps.append(("add", [A(n.input[0]), A(n.input[1])], n.output[0], dict(n=sz)))
    elif t in ("Transpose", "Reshape", "Sigmoid"):
        pass   # handled below as head ops
    else:
        raise SystemExit("unsupported op " + t)

# head ops: Transpose(NCHW->NHWC) -> Reshape -> [Sigmoid]
prod = {n.output[0]: n for n in nodes}
head = []
for o in outs:
    n = prod[o]; sig = n.op_type == "Sigmoid"
    if sig: n = prod[n.input[0]]
    assert n.op_type == "Reshape"; n = prod[n.input[0]]; assert n.op_type == "Transpose"
    src_t = A(n.input[0]); c, h, wd = SH[src_t][1:]
    head.append((src_t, o, c, h * wd, int(sig)))

# ---- liveness-based arena allocation (floats, 16-float aligned) ----
sizes = {y: int(np.prod(SH[y])) for op, ins, y, p in steps}
lu = {}
for i, (op, ins, y, p) in enumerate(steps):
    for x in ins: lu[x] = i
for (s_, o, c, hw, sg) in head: lu[s_] = len(steps)
live = {}       # name -> (start,size)
off = {}
arena_end = 0
def alloc(sz):
    global arena_end
    sz = (sz + 15) & ~15
    pos = 0
    for st, s2 in sorted(live.values()):
        if st - pos >= sz: break
        pos = max(pos, st + s2)
    arena_end = max(arena_end, pos + sz)
    return pos, sz
for i, (op, ins, y, p) in enumerate(steps):
    pos, sz = alloc(sizes[y]); live[y] = (pos, sz); off[y] = pos
    for x in ins:
        if lu.get(x) == i and x in live: del live[x]   # freed AFTER y was placed -> never aliases its own input
    if lu.get(y, -1) < i: live.pop(y, None)

# ---- emit C ----
b64 = base64.b64encode(bytes(blob)).decode()
b64lines = "\n".join('"%s"' % b64[i:i + 120] for i in range(0, len(b64), 120))

def ptr(x): return "in" if x == inp else "A+%d" % off[x]
body = []
for op, ins, y, p in steps:
    a = ",".join(str(p[k]) for k in p if k not in ("w", "b"))
    if op in ("pw", "c3s2", "dw"):
        body.append("%s(%s,A+%d,%s,%s,%s);" % (op, ptr(ins[0]), off[y], p["w"], p["b"], a))
    elif op == "add":
        body.append("add(%s,%s,A+%d,%d);" % (ptr(ins[0]), ptr(ins[1]), off[y], p["n"]))
    else:
        body.append("%s(%s,A+%d,%s);" % (op, ptr(ins[0]), off[y], a))
outargs = ["float %s[1][%d][%d]" % (o, SH[o][1], SH[o][2]) for o in outs]
for (s_, o, c, hw, sg) in head:
    body.append("head(%s,&%s[0][0][0],%d,%d,%d);" % (ptr(s_), o, c, hw, sg))

wdecl = "\n".join("static float %s[%d];" % (v, cnt) for v, cnt, dq, t in loads)
winit = []
for v, cnt, dq, t in loads:
    winit.append("ldf(%s,%d);" % (v, cnt) if dq is None else "ldq(%s,%d,%d,%d);" % (v, dq[0], dq[1], dq[2]))
    if t == "t16": winit.append("tr16(%s,%d);" % (v, cnt // 16))
code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime.c")).read()
for k, v in {"@@B64@@": b64lines, "@@WDECL@@": wdecl, "@@WINIT@@": "\n  ".join(winit),
             "@@ARENA@@": str(arena_end), "@@BODY@@": "\n  ".join(body),
             "@@OUTS@@": ", ".join(outargs), "@@MAXQ@@": str(max([d[0]*d[1] for v,c,d,t in loads if d])), "@@MAXZ@@": str(max([d[0]*-(-d[1]//d[2]) for v,c,d,t in loads if d]))}.items():
    code = code.replace(k, v)
open(dst, "w").write(code)
print("blob %d B, base64 %d B, arena %.2f MB, C file %d B" % (len(blob), len(b64), arena_end * 4 / 1e6, len(code)))
