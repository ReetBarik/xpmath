# Classify every function in a gfx90a device .s.
# Columns: KIND  bytes  relax_sites  poisoned_sites  symbol
#   KIND    KERNEL (ends s_endpgm, immune) or callee (returns via s[30:31])
#   bytes   instruction count x BPI (default 4.71, override with -v BPI=)
#   relax   BranchRelaxation long-jump expansions (any scratch register)
#   poison  ... of those, ones that scavenged s[30:31]  <-- the defect
#
# THE ESTIMATE IS DELIBERATELY LOW AND YOU MUST NOT GATE ON THE HARD REACH
# ALONE.  Function bodies here are `.p2align 2` and `.size` is
# `.Lfunc_end - start`, so where a FUNC symbol carries a nonzero st_size that
# value is the exact code size and can be used to calibrate.  Measured that way
# over three gfx90a builds (26 / 72 / 34 comparable symbols):
#
#       estimate/st_size   all functions        functions > 50 KB
#       mean                0.865 .. 0.919       0.908 .. 0.940
#       min                 0.678                0.865
#
# i.e. 4.71 under-reports by roughly 6-13 % on the large functions that tier 3
# actually judges, because it averages 4-byte SOP/VOP1/VOP2 encodings against
# 8-byte VOP3 and literal-carrying forms.  A callee estimated at the 131,068 B
# reach could really be ~151 KB.  That is why xpm_lint_device_asm.sh warns at
# 98,304 B (0.75 x reach): 98,304 / 0.865 = 113,646 B, still inside the reach
# even at the worst observed ratio.  Treat the WARN as the operational gate and
# the FATAL as the backstop.
#
# The discriminator is the .Lpost_getpc label: emitted ONLY by relaxation.
# A bare `s_getpc_b64 s[30:31]` is also how the backend spells an ordinary call
#
#     s_getpc_b64 s[30:31]
#     s_add_u32   s30, s30, callee@rel32@lo+4
#     s_addc_u32  s31, s31, callee@rel32@hi+12
#     s_swappc_b64 s[30:31], s[30:31]
#
# and a PC-relative constant address
#
#     s_getpc_b64 s[30:31]
#     s_add_u32   s30, s30, __const.foo@rel32@lo+4
#     ...
#     v_mov_b32_e32 v0, s30
#
# both of which are benign and both of which a naive grep reports as poisoned.
# That naive grep is what produced round 3's phantom 27-site "regression", every
# one of which was the address of the Payne-Hanek 2/pi table inside an immune
# kernel.  Only the relaxation form plants a label:
#
#     s_getpc_b64 s[30:31]
#   .Lpost_getpc156:                         <-- this, and only this
#     s_add_u32   s30, s30, (.LBB46_10344-.Lpost_getpc156)&4294967295
#     s_addc_u32  s31, s31, (.LBB46_10344-.Lpost_getpc156)>>32
#     s_setpc_b64 s[30:31]
#
# Sizes come from instruction lines, NEVER from readelf: on this target 27 of 61
# FUNC symbols report st_size == 0 and they are exactly the large ones, so an
# ELF-derived census understates the interesting functions by ~20x.

BEGIN { if (BPI + 0 <= 0) BPI = 4.71 }

# Function label: `name:` followed by the `; @name` comment the asm printer emits.
/^[A-Za-z_][A-Za-z0-9_$.]*:[ \t]*;[ \t]*@/ {
    fn = $1; sub(/:$/, "", fn); n = 0; ke = 0; relax = 0; poison = 0; pend = 0; next
}
/^\.Lfunc_end[0-9]+:/ {
    if (fn != "")
        printf "%-6s %9d %4d %4d %s\n", (ke ? "KERNEL" : "callee"),
               int(n * BPI), relax, poison, fn
    fn = ""; next
}
fn == "" { next }
/^\t[a-z]/ { n++; if ($1 == "s_endpgm") ke = 1 }
/s_getpc_b64 s\[30:31\]/ { pend = 1; next }
/^\.Lpost_getpc/ { relax++; if (pend) poison++ }
{ pend = 0 }
