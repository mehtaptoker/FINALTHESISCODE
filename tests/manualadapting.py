#####This code is made with help of G00GLE GEMINI############
import json
import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from geometry_env.simulator import GearTrainSimulator
from common.data_models import Gear, Point 
from visualization.renderer import Renderer
from pathfinding.finder import Pathfinder


import os, json, math, numpy as np
from common.data_models import Point
from preprocessing.processor import Processor
from gear_generator.factory import GearFactory
from visualization.renderer import Renderer

def create_tangent_flow_layout(example_name="Example3",
                               gear_diameters=(20, 12, 10, 8, 25),
                               clearance=0.0,        # extra gap added to tangency
                               margin=0.8,           # min edge clearance vs boundary
                               max_shrink_steps=6,   # small safety shrink attempts
                               shrink_factor=0.92):  # per step
    """
    Place gears so each new gear is tangent to the previous one, flowing along the path.
    - gear_diameters: sequence of diameters [input, g1, g2, ..., output]
    - clearance: extra spacing between pitch circles (0 = exact tangency)
    - margin: min distance from gear circle to polygon edges
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDIR  = os.path.join(BASE_DIR, "data")
    INTDIR = os.path.join(BASE_DIR, "data", "intermediate")
    OUTDIR = os.path.join(BASE_DIR, "output", example_name)
    os.makedirs(INTDIR, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)

    img   = os.path.join(INDIR, f"{example_name}.png")
    cons  = os.path.join(INDIR, f"{example_name}_constraints.json")
    proc  = os.path.join(INTDIR, f"{example_name}_processed.json")
    pathj = os.path.join(OUTDIR, "path.json")

    # Preprocess to get shafts/boundary
    try:
        Processor.process_input(img, cons, proc)
    except Exception as e:
        print(f"(preprocess) {e}")

    with open(proc, "r") as f:
        js = json.load(f)["normalized_space"]
    shaft_in  = tuple(js["input_shaft"].values())
    shaft_out = tuple(js["output_shaft"].values())
    polygon   = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in js["boundaries"]]

    # Load or form a simple path from input->output
    if os.path.exists(pathj):
        with open(pathj) as f:
            raw = json.load(f)
        path_pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in raw]
    else:
        # 5-point straight path
        path_pts = [(
            shaft_in[0] + t*(shaft_out[0]-shaft_in[0]),
            shaft_in[1] + t*(shaft_out[1]-shaft_in[1])
        ) for t in np.linspace(0, 1, 5)]
        with open(pathj, "w") as f:
            json.dump([{"x":x,"y":y} for (x,y) in path_pts], f, indent=2)

    # --- helpers ---
    def unit(v):
        n = np.linalg.norm(v)
        return v/n if n>1e-12 else np.array([1.0, 0.0])

    def point_in_poly(pt, poly):
        x,y = pt; inside=False
        for i in range(len(poly)):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
            if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1):
                inside = not inside
        return inside

    def seg_dist(a,b,p):
        a,b,p = np.array(a),np.array(b),np.array(p)
        ab=b-a; t=np.clip(np.dot(p-a,ab)/(np.dot(ab,ab)+1e-12),0,1)
        proj=a+t*ab
        return float(np.linalg.norm(p-proj))

    def circle_ok(c, r):
        if not point_in_poly(c, polygon): return False
        clr = min(seg_dist(polygon[i], polygon[(i+1)%len(polygon)], c) for i in range(len(polygon)))
        return clr >= (r + margin)

    def guide_dir(center, k):
        # direction toward next path waypoint (or to output if at end)
        tgt = path_pts[min(k, len(path_pts)-1)]
        return unit(np.array(tgt) - np.array(center))

    # --- build tangent chain ---
    gf = GearFactory(module=1.0)
    gears = []

    # place first gear exactly on input shaft (treat diam[0] as input gear)
    c0 = np.array(shaft_in, dtype=float)
    r0 = gear_diameters[0]/2.0
    # if the first is invalid, shrink slightly to fit
    rr = r0
    placed_c0 = tuple(c0)
    if not circle_ok(placed_c0, rr):
        for _ in range(max_shrink_steps):
            rr *= shrink_factor
            if circle_ok(placed_c0, rr):
                break
    g = gf.create_gear_from_diameter("gear_input", center=tuple(placed_c0), desired_diameter=2*rr)
    gears.append(g)
    prev_c, prev_r = np.array([g.center.x, g.center.y]), g.driving_radius  # use driving radius for distance

    # place the intermediate gears tangent along path direction
    for idx, d in enumerate(gear_diameters[1:-1], start=1):
        r = d/2.0
        u = guide_dir(prev_c, idx)
        cand = prev_c + u * (prev_r + r + clearance)

        # try nudges and small shrink if boundary fails
        ok = False; c = cand.copy(); rr = r
        if circle_ok(tuple(c), rr): ok=True
        if not ok:
            # flip side, lateral nudges, then shrink a bit
            for side in [1, -1]:
                if ok: break
                base = prev_c + side*u*(prev_r + rr + clearance)
                for lat in [0.0, 0.5, -0.5, 1.0, -1.0]:
                    if ok: break
                    nperp = np.array([-u[1], u[0]])
                    c = base + lat*nperp
                    if circle_ok(tuple(c), rr): ok=True; break
                if not ok:
                    rr2 = rr
                    for _ in range(max_shrink_steps):
                        rr2 *= shrink_factor
                        c2 = base
                        if circle_ok(tuple(c2), rr2):
                            c, rr, ok = c2, rr2, True
                            break
        if not ok:
            print(f"⚠ could not place gear_{idx} safely; skipping")
            continue

        gi = gf.create_gear_from_diameter(f"gear_{idx}", center=tuple(c), desired_diameter=2*rr)
        gears.append(gi)
        prev_c, prev_r = np.array([gi.center.x, gi.center.y]), gi.driving_radius

    # finally place output gear tangent toward output shaft
    r_out = gear_diameters[-1]/2.0
    u = unit(np.array(shaft_out) - prev_c)
    c_out = prev_c + u*(prev_r + r_out + clearance)
    rr = r_out
    # validate output gear
    if not circle_ok(tuple(c_out), rr):
        # try slight shrink
        for _ in range(max_shrink_steps):
            rr *= shrink_factor
            if circle_ok(tuple(c_out), rr):
                break
    gout = gf.create_gear_from_diameter("gear_output", center=tuple(c_out), desired_diameter=2*rr)
    gears.append(gout)

    # --- save + render ---
    out_json = os.path.join(OUTDIR, "tangent_flow_layout.json")
    with open(out_json, "w") as f:
        json.dump([g.to_json() for g in gears], f, indent=2)
    print(f"Saved tangent layout: {out_json}")

    try:
        Renderer.render_processed_data(
            processed_data_path=proc,
            output_path=os.path.join(OUTDIR, "tangent_flow_layout.png"),
            path=[Point(x=p[0], y=p[1]) for p in path_pts],
            gears=gears
        )
        print(f"Visualization: {os.path.join(OUTDIR, 'tangent_flow_layout.png')}")
    except Exception as e:
        print(f"Render warning: {e}")

    return gears

