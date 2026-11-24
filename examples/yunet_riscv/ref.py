import onnx, numpy as np, onnxruntime as ort
from onnx import numpy_helper as nh
m=onnx.load("yunet_int8bq.onnx")
I={t.name:t for t in m.graph.initializer}
keep=[];new=[]
for n in m.graph.node:
    if n.op_type!="DequantizeLinear": keep.append(n);continue
    a={x.name:x.i for x in n.attribute}
    x=nh.to_array(I[n.input[0]]).astype(np.float32)
    s=nh.to_array(I[n.input[1]]).astype(np.float32)
    z=nh.to_array(I[n.input[2]]).astype(np.float32) if len(n.input)>2 else 0
    ax,bs=a["axis"],a["block_size"]
    s=np.repeat(s,bs,axis=ax)[tuple(slice(0,d) for d in x.shape)]
    z=np.repeat(z,bs,axis=ax)[tuple(slice(0,d) for d in x.shape)] if np.ndim(z) else z
    new.append(nh.from_array(((x-z)*s).astype(np.float32),n.output[0]))
del m.graph.node[:]; m.graph.node.extend(keep); m.graph.initializer.extend(new)
onnx.save(m,"yunet_folded.onnx")
np.random.seed(0)
x=(np.random.rand(1,3,640,640).astype(np.float32)*255); x.tofile("input.bin")
so=ort.SessionOptions(); so.log_severity_level=3
out=ort.InferenceSession("yunet_folded.onnx",so).run(None,{"input":x})
np.concatenate([o.ravel() for o in out]).tofile("ref.bin"); print([o.shape for o in out])
