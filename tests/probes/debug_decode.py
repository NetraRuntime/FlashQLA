# tests/probes/debug_decode.py
"""Debug the decode kernel vs reference on the simplest case (B=1,Hk=Hv=1,D=1)."""
import torch
from ref_gdr import decode_recur
from flash_qla import recurrent_gated_delta_rule
from flash_qla.utils import l2norm

torch.manual_seed(0)
B, D, Hk, Hv = 1, 1, 1, 1
q = l2norm(torch.randn(B, D, Hk, 128, device="cuda", dtype=torch.bfloat16))
k = l2norm(torch.randn(B, D, Hk, 128, device="cuda", dtype=torch.bfloat16))
v = torch.randn(B, D, Hv, 128, device="cuda", dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.randn(B, D, Hv, device="cuda")) / 16
beta = torch.randn(B, D, Hv, device="cuda").sigmoid()

o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
o_qla, s_qla = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True)

print("g:", g.item(), "beta:", beta.item())
print("o_ref[0,0,0,:6]:", o_ref[0, 0, 0, :6].tolist())
print("o_qla[0,0,0,:6]:", o_qla[0, 0, 0, :6].float().tolist())
print("o rel err:", ((o_qla.float() - o_ref).abs().max() / o_ref.abs().max()).item())
print()
print("s_ref[0,0,:3,:4]:\n", s_ref[0, 0, :3, :4])
print("s_qla[0,0,:3,:4]:\n", s_qla[0, 0, :3, :4].float())
print("s rel err:", ((s_qla - s_ref).abs().max() / s_ref.abs().max()).item())
sd = (s_qla - s_ref).abs()
idx = sd.argmax()
dk_i, dv_i = (idx % (128 * 128)) // 128, idx % 128
print(f"s max-err at [dk={dk_i}, dv={dv_i}]: qla={s_qla[0,0,dk_i,dv_i].item():.5f} ref={s_ref[0,0,dk_i,dv_i].item():.5f}")
print("s per-col-tile max err (cols 0-31,32-63,64-95,96-127):",
      [round((s_qla[0,0,:,c:c+32]-s_ref[0,0,:,c:c+32]).abs().max().item(), 4) for c in range(0, 128, 32)])

# manual: S should be k (x) (beta*v); o = scale*(q.k)*(beta*v)
manual_S = torch.outer(k[0, 0, 0].float(), beta.item() * v[0, 0, 0].float())
print("\nmanual_S[0,:4]:", manual_S[0, :4].tolist())
print("s_ref[0,0,0,:4]:", s_ref[0, 0, 0, :4].tolist())
qk = (q[0, 0, 0].float() * k[0, 0, 0].float()).sum().item()
print("q.k:", qk, "| manual o[0]:", 128 ** -0.5 * qk * (beta.item() * v[0, 0, 0, 0].float()).item())
