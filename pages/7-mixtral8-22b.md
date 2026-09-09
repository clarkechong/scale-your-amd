llama70b was to demonstrate mixed precision training (no constraints on dtype to fit on node)
this can be used to demonstrate MoE specifics

groupgemm/ragged dot (token routing, all-to-all collective)

try out jax-aiter kernels

Baseline MoE

+ GroupGEMM

+ Token sorting

+ All-to-all overlap

+ Ragged Dot

+ Fused routing

---

jax-aiter vs jax-triton vs te attention