#!/usr/bin/env python3
"""
Growbot — a TARS-inspired bipedal robot.  This build is the INTERNAL-VISUALISATION
model: the chassis is hollowed into real shells and the drivetrain + battery +
electronics are modelled inside so you can take cross-sections through it.

Drive architecture — direct-drive hobby servos (MG996R), no worm/gears:
  * HIP  : MG996R in the torso, output spline + horn straight out to the leg
           at the hip axis (X, Y=230).  Body proximal -> low leg-swing inertia.
  * ANKLE: MG996R stands in the LOWER LEG; its crank drives a PUSHROD down to a
           rocker on the foot, so the foot pitches about a real hinge pin (the
           ankle axis, X).  This 4-bar linkage keeps the foot slim and the rod
           visible/serviceable.  The leg is 44 mm wide to house the servo
           (its body is 37 mm along the shaft, so it won't fit a 38 mm leg).
  * NOT self-locking (no worm): the servos hold position under power; the wide
    feet + low CoM keep it stable, and the central rest foot parks it.

Units: mm, +Y up, +Z forward.  Output: Growbot_TARS.obj + .mtl
Every part is its own `o` group.  See render_sections.py for the cutaways.
"""

import math

# ---------------------------------------------------------------- parameters --
STAND_H = 290.0         # 305->290 (~11.4 in): shorten to offset the wider torso, hold mass
WALL    = 3.0            # chassis wall thickness (CF/ABS shell)

T_W, T_D, T_BOT = 82.0, 58.0, 20.0     # torso (W 76->82 so the 2 hip servos go COAXIAL)
L_W, L_D = 44.0, 44.0                  # legs (W 38->44 to house the ankle MG996R)
GAP   = 6.0
HIP_Y = 230.0
SPLIT_Y = 150.0                        # bed-split plane (in the dark band) — bolted seam

F_W, F_D = 55.0, 95.0                  # feet
F_BLK_TOP = 34.0                       # raised so the ankle wheel fits inside
PAD_H, PAD_OVER = 6.0, 1.0
CF_CLEAR, CF_W, CF_D = 8.0, 60.0, 50.0

CYL_SEG = 24

# ankle pushrod linkage — MG996R high in the leg drives a crank -> pushrod ->
# foot rocker; the foot pitches about a real hinge pin (the ankle axis, X).
ANK_SERVO_Y = 80.0     # ankle-servo shaft height inside the leg
ANK_PIVOT_Y = 43.0     # ankle hinge axis (foot pitches about this, on X)
LEG_BOT     = 52.0     # leg shell stops here; Y 34..52 is the open ankle gap

# panel aesthetic (unchanged)
PANEL_COL, ROW_UNIT = 9.5, 5.0
ROW_PATTERN = [2, 3, 2, 4, 2, 3, 3, 2, 4, 2]
PANEL_GROOVE, PANEL_RELIEF, PANEL_BORDER, PANEL_OVERLAP = 1.1, 0.5, 1.8, 0.5
CAP_TGT = 9.0
ROUND = 4


# -------------------------------------------------------------------- writer --
class Obj:
    def __init__(self):
        self.v = []
        self.lines = []
        self.vol = {}                 # accumulated solid volume per material (mm^3)
        self.cur = "mat_body"
    def _addvol(self, dv): self.vol[self.cur] = self.vol.get(self.cur, 0.0) + dv
    def _vadd(self, x, y, z):
        self.v.append((round(x, ROUND), round(y, ROUND), round(z, ROUND)))
        return len(self.v)
    def group(self, n): self.lines.append(("o", n))
    def mtl(self, n):   self.cur = n; self.lines.append(("usemtl", n))
    def face(self, i):  self.lines.append(("f", i))

    def _corners(self, p):
        a,b,c,d,e,f,g,h = [self._vadd(*q) for q in p]
        self.face([a,d,c,b]); self.face([e,f,g,h])
        self.face([a,e,h,d]); self.face([b,c,g,f])
        self.face([a,b,f,e]); self.face([d,h,g,c])

    def box(self, x0,x1,y0,y1,z0,z1):
        self._corners([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                       (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
        self._addvol(abs((x1-x0)*(y1-y0)*(z1-z0)))

    # cylinders along each axis ------------------------------------------------
    def cyl_x(self, cx,cy,cz,r,length,seg=CYL_SEG):
        x0,x1=cx-length/2,cx+length/2; r0=[];r1=[]; self._addvol(math.pi*r*r*length)
        for i in range(seg):
            t=2*math.pi*i/seg; y=cy+r*math.cos(t); z=cz+r*math.sin(t)
            r0.append(self._vadd(x0,y,z)); r1.append(self._vadd(x1,y,z))
        c0=self._vadd(x0,cy,cz); c1=self._vadd(x1,cy,cz)
        for i in range(seg):
            j=(i+1)%seg
            self.face([r0[i],r0[j],r1[j],r1[i]])
            self.face([c0,r0[j],r0[i]]); self.face([c1,r1[i],r1[j]])
    def cyl_y(self, cx,cy,cz,r,length,seg=CYL_SEG):
        y0,y1=cy-length/2,cy+length/2; r0=[];r1=[]; self._addvol(math.pi*r*r*length)
        for i in range(seg):
            t=2*math.pi*i/seg; x=cx+r*math.cos(t); z=cz+r*math.sin(t)
            r0.append(self._vadd(x,y0,z)); r1.append(self._vadd(x,y1,z))
        c0=self._vadd(cx,y0,cz); c1=self._vadd(cx,y1,cz)
        for i in range(seg):
            j=(i+1)%seg
            self.face([r0[i],r0[j],r1[j],r1[i]])
            self.face([c0,r0[j],r0[i]]); self.face([c1,r1[i],r1[j]])

    # flat link bar between two (z,y) points at a given X (crank/rod/rocker) ---
    def link_bar(self, x, z0,y0, z1,y1, w=4.0, tx=4.0):
        dz,dy=z1-z0, y1-y0; L=math.hypot(dz,dy) or 1.0; self._addvol(w*tx*L)
        pz,py=-dy/L*w/2, dz/L*w/2                 # perpendicular in the Z-Y plane
        pts=[(z0+pz,y0+py),(z0-pz,y0-py),(z1-pz,y1-py),(z1+pz,y1+py)]
        x0,x1=x-tx/2,x+tx/2
        self._corners([(x0,y,z) for (z,y) in pts]+[(x1,y,z) for (z,y) in pts])

    # ----- panel-grid helpers (uniform cols, varying rows) -------------------
    @staticmethod
    def _uniform(lo,hi,count,groove,border):
        span=(hi-lo)-2*border
        if count<1 or span<=0: return []
        cell=(span-(count-1)*groove)/count
        return [(lo+border+i*(cell+groove),lo+border+i*(cell+groove)+cell)
                for i in range(count)] if cell>0 else []
    @staticmethod
    def _varying(lo,hi,unit,pattern,groove,border,phase=0):
        avail=(hi-lo)-2*border
        if avail<=0: return []
        n,used=0,0.0
        while True:
            w=pattern[(n+phase)%len(pattern)]
            add=w*unit+(groove if n>0 else 0)
            if used+add>avail and n>=1: break
            used+=add; n+=1
            if n>800: break
        wts=[pattern[(i+phase)%len(pattern)] for i in range(n)]
        eff=(avail-(n-1)*groove)/sum(wts)
        out,y=[],lo+border
        for w in wts:
            h=w*eff; out.append((y,y+h)); y+=h+groove
        return out
    def panels(self, face, x0,x1,y0,y1,z0,z1, mtl,
               col_target=PANEL_COL, groove=PANEL_GROOVE, relief=PANEL_RELIEF,
               border=PANEL_BORDER, overlap=PANEL_OVERLAP,
               unit=ROW_UNIT, pattern=ROW_PATTERN, phase=0,
               force_cols=None, force_rows=None, uniform=False):
        if face in ("+Z","-Z"):
            a0,a1=x0,x1; vert=(y0,y1)
            tile=(lambda u0,u1,v0,v1:self.box(u0,u1,v0,v1,z1-overlap,z1+relief)) if face=="+Z" \
                 else (lambda u0,u1,v0,v1:self.box(u0,u1,v0,v1,z0-relief,z0+overlap))
        elif face in ("+X","-X"):
            a0,a1=z0,z1; vert=(y0,y1)
            tile=(lambda u0,u1,v0,v1:self.box(x1-overlap,x1+relief,v0,v1,u0,u1)) if face=="+X" \
                 else (lambda u0,u1,v0,v1:self.box(x0-relief,x0+overlap,v0,v1,u0,u1))
        else:
            a0,a1=x0,x1; vert=(z0,z1)
            tile=lambda u0,u1,v0,v1:self.box(u0,u1,y1-overlap,y1+relief,v0,v1)
        cols=force_cols or max(1,round((a1-a0)/col_target))
        col_b=self._uniform(a0,a1,cols,groove,border)
        if face=="+Y":
            nr=force_rows or max(1,round((vert[1]-vert[0])/CAP_TGT))
            row_b=self._uniform(vert[0],vert[1],nr,groove,border)
        elif uniform:
            row_b=self._uniform(vert[0],vert[1],force_rows or 1,groove,border)
        else:
            row_b=self._varying(vert[0],vert[1],unit,pattern,groove,border,phase)
        for (u0,u1) in col_b:
            for (v0,v1) in row_b: tile(u0,u1,v0,v1)

    def write(self, obj_path, mtl_name):
        with open(obj_path,"w") as fh:
            fh.write("# Growbot internal-visualisation model — generated\n")
            fh.write(f"# units: mm   standing height: {STAND_H} mm   wall: {WALL} mm\n")
            fh.write(f"mtllib {mtl_name}\n")
            for (x,y,z) in self.v: fh.write(f"v {x} {y} {z}\n")
            for k,p in self.lines:
                if k=="o": fh.write(f"o {p}\n")
                elif k=="usemtl": fh.write(f"usemtl {p}\n")
                else: fh.write("f "+" ".join(str(i) for i in p)+"\n")


O = Obj()

# ------------------------------------------------------- chassis sub-builders --
def shell(x0,x1,y0,y1,z0,z1, t=WALL, cap_top=True, cap_bot=True):
    O.mtl("mat_body")
    O.box(x0,x0+t, y0,y1, z0,z1)            # -X wall
    O.box(x1-t,x1, y0,y1, z0,z1)            # +X wall
    O.box(x0+t,x1-t, y0,y1, z0,z0+t)        # -Z wall
    O.box(x0+t,x1-t, y0,y1, z1-t,z1)        # +Z wall
    if cap_bot: O.box(x0+t,x1-t, y0,y0+t, z0+t,z1-t)
    if cap_top: O.box(x0+t,x1-t, y1-t,y1, z0+t,z1-t)

def decorate_gray(x0,x1,y0,y1,z0,z1, phase=0, top=False):
    O.mtl("mat_panel_line")
    for s in ("+Z","-Z","+X","-X"):
        O.panels(s,x0,x1,y0,y1,z0,z1,"mat_panel_line",phase=phase)
    if top: O.panels("+Y",x0,x1,y0,y1,z0,z1,"mat_panel_line")

def band_collar(x0,x1,y0,y1,z0,z1):
    """Dark ribbed grip band as a thin proud collar around the shell."""
    O.mtl("mat_accent")
    p=0.4
    O.box(x0,x1, y0,y1, z1-0.1,z1+p)        # front plate
    O.box(x0,x1, y0,y1, z0-p,z0+0.1)        # back plate
    O.box(x0-p,x0+0.1, y0,y1, z0,z1)        # left
    O.box(x1-0.1,x1+p, y0,y1, z0,z1)        # right
    O.mtl("mat_accent_rib")
    rows=max(2,int((y1-y0)/3.2))
    for s in ("+Z","-Z"):
        O.panels(s,x0,x1,y0,y1,z0-p,z1+p,"mat_accent_rib",
                 groove=1.0,relief=0.4,border=1.2,force_cols=1,force_rows=rows,uniform=True)

def nameplate(cx,cy,zface):
    O.group("nameplate"); O.mtl("mat_panel")
    O.box(cx-26,cx+26, cy-12,cy+12, zface-0.4,zface+0.8)
    O.mtl("mat_accent")
    for i in range(5):
        xc=cx-16+i*8
        O.box(xc-1.6,xc+1.6, cy-8-1.6,cy-8+1.6, zface+0.5,zface+1.8)

def tpu_pad(cx,sole_y,w,d):
    x0,x1=cx-w/2-PAD_OVER,cx+w/2+PAD_OVER
    z0,z1=-d/2-PAD_OVER,d/2+PAD_OVER
    O.mtl("mat_tpu"); O.box(x0,x1,sole_y+1.6,sole_y+PAD_H,z0,z1)
    O.mtl("mat_tpu_tread")
    span=z1-z0; nb=max(3,int(span/10)); bw=(span-(nb-1)*4)/nb
    for i in range(nb):
        zb0=z0+i*(bw+4); O.box(x0+2,x1-2,sole_y,sole_y+2.1,zb0,zb0+bw)

# ------------------------------------------------- MG996R servo (real dims) ---
# body 40.9(L=B) x 20(D) x 37(C); tabs span 54(E); boss+spline on top; 55 g each.
SV_B, SV_D, SV_C = 40.9, 20.0, 37.0
SV_BOSS, SV_TABSPAN, SV_TABT = 5.7, 54.0, 2.5
SV_SPL_R, SV_SPL_H, SV_HORN_R, SV_HORN_T, SV_SPL_OFF = 3.0, 4.0, 10.0, 3.0, 10.0

def boxc(cx, cy, cz, ex, ey, ez):
    O.box(cx-ex/2, cx+ex/2, cy-ey/2, cy+ey/2, cz-ez/2, cz+ez/2)

def servo(group, sx, sy, sz, sdir, long_axis, horn=True):
    """Direct-drive MG996R. Output spline at (sx,sy,sz) on the X (joint) axis,
    pointing sdir(±1) toward the driven limb; body sits inboard. long_axis
    ('y'/'z') is the 40.9 mm body length. horn=True draws the disc horn (hip);
    horn=False leaves a bare spline for the ankle crank to attach to."""
    cx = sx - sdir*(SV_BOSS + SV_C/2)              # case-body centre on X
    tx = sx - sdir*(SV_BOSS + SV_TABT/2)           # mounting-tab plane
    ov = (SV_TABSPAN - SV_B)/2
    O.group(group); O.mtl("mat_servo")
    if long_axis == 'y':
        cy, cz = sy - SV_SPL_OFF, sz
        boxc(cx, cy, cz, SV_C, SV_B, SV_D)
        boxc(tx, cy+SV_B/2+ov/2, cz, SV_TABT, ov, SV_D)
        boxc(tx, cy-SV_B/2-ov/2, cz, SV_TABT, ov, SV_D)
    else:                                          # body length along Z
        cy, cz = sy, sz - SV_SPL_OFF
        boxc(cx, cy, cz, SV_C, SV_D, SV_B)
        boxc(tx, cy, cz+SV_B/2+ov/2, SV_TABT, SV_D, ov)
        boxc(tx, cy, cz-SV_B/2-ov/2, SV_TABT, SV_D, ov)
    boxc(sx - sdir*SV_BOSS/2, sy, sz, SV_BOSS, 16, 16)         # gear boss
    O.mtl("mat_steel")                                          # output spline
    O.cyl_x(sx + sdir*SV_SPL_H/2, sy, sz, SV_SPL_R, SV_SPL_H)
    if horn:
        O.mtl("mat_horn")                                       # servo horn
        O.cyl_x(sx + sdir*(SV_SPL_H+SV_HORN_T/2), sy, sz, SV_HORN_R, SV_HORN_T)


# ============================================================ BUILD ===========
t_half = T_W/2
leg_inner = t_half + GAP
leg_x = leg_inner + L_W/2               # ±63

# ---- TORSO shell + exterior bands ----
torso_segs=[("gray",T_BOT,70),("band",70,92),("gray",92,150),
            ("band",150,172),("gray",172,STAND_H)]
O.group("central_body")
shell(-t_half,t_half, T_BOT,STAND_H, -T_D/2,T_D/2, cap_top=True, cap_bot=True)
for k,(kind,y0,y1) in enumerate(torso_segs):
    if kind=="gray":
        decorate_gray(-t_half,t_half,y0,y1,-T_D/2,T_D/2,phase=k*2,top=(y1>=STAND_H))
    else:
        band_collar(-t_half,t_half,y0,y1,-T_D/2,T_D/2)
nameplate(0.0,250.0,T_D/2)

# ---- TORSO internals: real battery + electronics ----
O.group("battery");  O.mtl("mat_battery")          # Flipo 2S 3300 mAh (~105x35x24)
boxc(0, 72.5, 0, 35, 105, 24)                       # vertical, low -> low CoM
O.group("controller_stack"); O.mtl("mat_pcb")       # Pi Zero 2W + PCA9685
boxc(0, 158, -13, 66, 46, 14)
O.group("power_buck"); O.mtl("mat_motor")           # XL4016 (+ MP1584)
boxc(0, 150, 16, 52, 34, 12)

# ---- bed-split BOLT JOINTS: screw the two printed halves together at SPLIT_Y ----
# Each is a printed boss spanning the seam (lower half = heat-set insert, upper half =
# clearance hole) + an M3 screw.  SPLIT_Y sits in the dark accent band (hides the seam).
def bolt_joint(cx, cz, ysplit=SPLIT_Y, boss_r=4.5, span=18.0):
    O.mtl("mat_body"); O.cyl_y(cx, ysplit, cz, boss_r, span)     # printed boss (both halves)
    O.mtl("mat_steel")
    O.cyl_y(cx, ysplit, cz, 1.6, span+4)                         # M3 screw shank
    O.cyl_y(cx, ysplit+span/2+2.0, cz, 3.2, 3.0)                 # screw head
O.group("screws_torso")
for cx in (-33.5, 33.5):
    for cz in (-21.5, 21.5): bolt_joint(cx, cz)

# ---- HIP servos: 2x MG996R, COAXIAL & SYMMETRIC (both on the X axis at Z=0) ----
# back-to-back in the widened torso so the left/right hip axes coincide -> symmetric
# load + symmetric control (no fore/aft stagger to compensate for in software).
for side in (-1,+1):
    tag = 'L' if side<0 else 'R'
    servo(f"hip_servo_{tag}", sx=side*43.0, sy=HIP_Y-5, sz=0.0,
          sdir=side, long_axis='y')                 # coaxial, body inboard, spline to leg

# ---- LEGS shell + bands + ANKLE drives + feet + pads ----
def build_leg(side):
    sx=side*leg_x; x0,x1=sx-L_W/2,sx+L_W/2; z0,z1=-L_D/2,L_D/2
    tag="L" if side<0 else "R"; word="left" if side<0 else "right"
    lo=lambda a,b:(min(a,b),max(a,b))

    # --- leg shell: stops at LEG_BOT; Y 34..52 below it is the open ankle joint
    O.group(f"leg_{word}")
    shell(x0,x1, LEG_BOT,STAND_H, z0,z1, cap_top=True, cap_bot=True)
    decorate_gray(x0,x1,LEG_BOT,150,z0,z1,phase=1)
    band_collar(x0,x1,150,168,z0,z1)
    decorate_gray(x0,x1,168,STAND_H,z0,z1,phase=4,top=True)
    O.mtl("mat_body")                                   # clevis: 2 ears carry the pin
    iw, ow = leg_x-L_W/2, leg_x+L_W/2                    # inner / outer wall distances
    O.box(*lo(side*iw, side*(iw+WALL)), 40,LEG_BOT, -8,8)   # inner ear (toward torso)
    O.box(*lo(side*(ow-WALL), side*ow), 40,LEG_BOT, -8,8)   # outer ear

    # --- foot shell + central hinge lug + sole pad ---
    O.group(f"foot_{word}")
    shell(sx-F_W/2,sx+F_W/2, PAD_H,F_BLK_TOP, -F_D/2,F_D/2, cap_top=False, cap_bot=True)
    decorate_gray(sx-F_W/2,sx+F_W/2,PAD_H,F_BLK_TOP,-F_D/2,F_D/2,top=False)
    O.mtl("mat_body")
    O.box(*lo(sx-8,sx+8), F_BLK_TOP,49, -8,8)           # central lug rides the pin
    O.group(f"foot_pad_{word}"); tpu_pad(sx,0.0,F_W,F_D)

    # --- ankle drive: MG996R in the lower leg -> crank -> pushrod -> rocker ---
    # All linkage parts are 3D-PRINTED (CF-ABS); the pin is a printed CF stub
    # (drop in an M3 bolt for long life).  Linkage runs just outboard of the leg.
    Xl = side*(ow+4.0)                                  # linkage plane (outboard)
    servo(f"ankle_servo_{tag}", sx=side*(leg_x+SV_BOSS+SV_C/2), sy=ANK_SERVO_Y, sz=0.0,
          sdir=side, long_axis='y', horn=False)         # body centred in the leg
    O.group(f"ankle_pivot_{tag}"); O.mtl("mat_print_cf")   # hinge pin = ankle axis (X)
    O.cyl_x(side*((iw+ow)/2+2), ANK_PIVOT_Y, 0.0, 4.0, (ow+4-iw)+4)  # Ø8 across the clevis
    O.group(f"ankle_crank_{tag}"); O.mtl("mat_print_cf")   # servo crank (printed)
    O.link_bar(Xl, 0.0,ANK_SERVO_Y, -9.0,70.0, w=8, tx=6)
    O.group(f"ankle_pushrod_{tag}"); O.mtl("mat_print_cf") # rigid pushrod (printed)
    O.link_bar(Xl, -9.0,70.0, -9.0,57.0, w=6, tx=6)
    O.group(f"foot_rocker_{tag}"); O.mtl("mat_print_cf")   # foot lever (printed)
    O.link_bar(Xl, 0.0,F_BLK_TOP, -9.0,57.0, w=8, tx=6)

    # --- ankle-servo wiring: clean dog-leg that AVOIDS every solid part --------
    # up the leg cavity (z=15 sits behind the servo, z<=10) -> across ABOVE the hip
    # servos (y=248 > 245) -> down the torso side wall -> in to the PCA9685.
    O.group(f"wiring_{tag}"); O.mtl("mat_wire")
    O.cyl_y(sx, 174.0, 15.0, 1.8, 148.0)                   # 1) up the leg cavity (y100..248)
    O.cyl_x(side*(leg_x+37)/2, 248.0, 15.0, 1.8, leg_x-37) # 2) cross above the hip servos
    O.cyl_y(side*36.0, 188.0, 15.0, 1.8, 120.0)            # 3) down the torso side wall
    O.cyl_x(side*24.0, 128.0, 15.0, 1.8, 24.0)             # 4) in to the PCA9685 (clear gap)

    O.group(f"screws_leg_{tag}")                           # bed-split bolts at SPLIT_Y
    for ex in (-14,14):
        for ez in (-10,10): bolt_joint(sx+ex, ez)

build_leg(-1); build_leg(+1)

# central rest foot
O.group("central_rest_foot")
shell(-CF_W/2,CF_W/2, CF_CLEAR+PAD_H,T_BOT, -CF_D/2,CF_D/2, cap_top=False, cap_bot=True)
decorate_gray(-CF_W/2,CF_W/2,CF_CLEAR+PAD_H,T_BOT,-CF_D/2,CF_D/2)
O.group("central_rest_pad"); tpu_pad(0.0,CF_CLEAR,CF_W,CF_D)

O.write("Growbot_TARS.obj","Growbot_TARS.mtl")

# --------------------------------------------------------------- materials ---
mtl="""# Growbot materials
newmtl mat_body
Kd 0.72 0.73 0.74
Ks 0.10 0.10 0.10
Ns 20
newmtl mat_panel_line
Kd 0.66 0.67 0.68
Ns 20
newmtl mat_accent
Kd 0.13 0.13 0.14
Ns 8
newmtl mat_accent_rib
Kd 0.18 0.18 0.19
Ns 8
newmtl mat_tpu
Kd 0.06 0.06 0.06
Ns 4
newmtl mat_tpu_tread
Kd 0.10 0.10 0.10
Ns 4
newmtl mat_panel
Kd 0.55 0.56 0.58
Ns 20
newmtl mat_motor
Kd 0.24 0.26 0.30
Ks 0.30 0.30 0.30
Ns 40
newmtl mat_print_cf
Kd 0.16 0.16 0.18
Ks 0.18 0.18 0.20
Ns 25
newmtl mat_wire
Kd 0.70 0.12 0.12
Ns 10
newmtl mat_steel
Kd 0.62 0.64 0.68
Ks 0.55 0.55 0.55
Ns 60
newmtl mat_battery
Kd 0.14 0.42 0.34
Ns 15
newmtl mat_pcb
Kd 0.08 0.34 0.16
Ns 15
newmtl mat_servo
Kd 0.10 0.13 0.24
Ks 0.20 0.20 0.22
Ns 30
newmtl mat_horn
Kd 0.85 0.85 0.80
Ns 20
"""
open("Growbot_TARS.mtl","w").write(mtl)

xs=[p[0] for p in O.v]; ys=[p[1] for p in O.v]; zs=[p[2] for p in O.v]
print(f"vertices {len(O.v)}  faces {sum(1 for k,_ in O.lines if k=='f')}")
print(f"X {min(xs):.1f}..{max(xs):.1f}  Y {min(ys):.1f}..{max(ys):.1f}  Z {min(zs):.1f}..{max(zs):.1f}")
print("groups:", sum(1 for k,_ in O.lines if k=='o'))
print("wrote Growbot_TARS.obj + Growbot_TARS.mtl")

# ---- mass estimate (parametric, from accumulated solid volumes) --------------
# densities g/mm^3: ABS 1.04, CF-ABS 1.10, TPU 1.20 (walls = solid perimeters).
ABS, CFP, TPU = 1.04e-3, 1.10e-3, 1.20e-3
def vmass(rho,*names): return rho*sum(O.vol.get(n,0.0) for n in names)
m_shell = vmass(ABS,"mat_body","mat_panel_line","mat_panel","mat_accent","mat_accent_rib")
m_drive = vmass(CFP,"mat_print_cf")
m_tpu   = vmass(TPU,"mat_tpu","mat_tpu_tread")
m_servo, m_batt, m_elec = 4*55.0, 194.0, 106.0          # bought parts (catalog)
m_total = m_shell+m_drive+m_tpu+m_servo+m_batt+m_elec
print("--- mass estimate (g) -------------------------")
print(f"  printed shell + panels (ABS) : {m_shell:6.0f}")
print(f"  printed drive parts (CF)     : {m_drive:6.0f}")
print(f"  TPU pads/soles               : {m_tpu:6.0f}")
print(f"  4x MG996R servo (55 ea)      : {m_servo:6.0f}")
print(f"  LiPo 2S 3300                 : {m_batt:6.0f}")
print(f"  electronics (Pi/PCA/bucks)   : {m_elec:6.0f}")
print(f"  TOTAL                        : {m_total:6.0f}   ({m_total/1000:.2f} kg)")
