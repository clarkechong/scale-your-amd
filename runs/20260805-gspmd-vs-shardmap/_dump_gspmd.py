
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

n = 8
mesh = Mesh(jax.devices()[:n], axis_names=("Y",))
dtype = jnp.bfloat16
B, D, F = 4096, 4096, 14336
xs = NamedSharding(mesh, P(None, "Y"))
ws = NamedSharding(mesh, P("Y", None))
os_ = NamedSharding(mesh, P(None, None))
x = jax.device_put(jnp.ones((B, D), dtype), xs)
w = jax.device_put(jnp.ones((D, F), dtype), ws)

if "gspmd" == "gspmd":
    fn = jax.jit(lambda a, b: a @ b, in_shardings=(xs, ws), out_shardings=os_)
else:
    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(P(None, "Y"), P("Y", None)), out_specs=P(None, None))
    def fn(a, b):
        return jax.lax.psum(a @ b, "Y")

jax.block_until_ready(fn(x, w))
