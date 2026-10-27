# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mujoco==3.11.0", "numpy", "scipy", "usd-core==26.8"]
# [tool.uv]
# exclude-newer = "2026-10-27T00:00:00Z"  # reproducible resolution; bump deliberately
# ///
"""Convert the lab course's collision prims (mad_assets `scenes/default.usd`) into `course.xml`.

    uv run scripts/mjswan/bake_course.py           # rerun when the course changes in mad_assets
    uv run scripts/mjswan/bake_course.py --check   # exit 1 if course.xml no longer matches mad_assets

Stairs, ramp, pallets and the brick road become static MuJoCo geoms (also used as visuals), so
the demo and CI need neither mad_assets (356 MB of LFS) nor usd-core.
"""

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

HERE = Path(__file__).parent
MAD_ASSETS = Path(os.environ.get("MAD_ASSETS_PATH", HERE.parents[2] / "mad_assets"))


def _add_heightfield(spec: mujoco.MjSpec, name: str, pts: np.ndarray, res: float = 0.05) -> None:
    """Rasterize a dense surface mesh (1.3 M points for the brick road) into an hfield.

    At 2 cm, MuJoCo-Warp overflows its 50 contacts per hfield pair under a G1 foot and drops
    contacts (robots trip at the road's edge); 5 cm keeps it under the limit.
    """
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    ncol, nrow = np.ceil((hi - lo) / res).astype(int) + 1
    ix = np.clip(((pts[:, 0] - lo[0]) / res).astype(int), 0, ncol - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / res).astype(int), 0, nrow - 1)
    z = np.full((nrow, ncol), -np.inf)
    np.maximum.at(z, (iy, ix), pts[:, 2])
    while np.isinf(z).any():  # fill the few empty cells from their neighbours
        pad = np.pad(z, 1, constant_values=-np.inf)
        nb = np.max([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]], axis=0)
        z = np.where(np.isinf(z), nb, z)
    zmax = z.max()
    half = (hi - lo) / 2
    # MuJoCo hfield: row 0 is -y, column 0 is -x; data normalized to [0, 1] * size[2].
    spec.add_hfield(name=name, nrow=nrow, ncol=ncol, size=[half[0], half[1], zmax, 0.05], userdata=(z / zmax).ravel())
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=[*(lo + half), 0], material="stone"
    )


def course_spec() -> mujoco.MjSpec:
    stage = Usd.Stage.Open(str(MAD_ASSETS / "scenes" / "default.usd"))
    spec = mujoco.MjSpec()
    spec.modelname = "course"
    root = spec.worldbody
    spec.add_material(name="wood", rgba=(0.62, 0.48, 0.33, 1))
    spec.add_material(name="steel", rgba=(0.55, 0.57, 0.6, 1))
    spec.add_material(name="stone", rgba=(0.45, 0.42, 0.4, 1))

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not path.startswith(("/env", "/equipment")):
            continue
        if "/floor/" in path:  # the terrain plane
            continue
        name = path.strip("/").replace("/", "__")
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if prim.IsA(UsdGeom.Cube):
            t = Gf.Transform(xf)
            q = t.GetRotation().GetQuat()
            half = np.array(t.GetScale()) * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            pos = np.array(t.GetTranslation())
            if "/step_" in path:
                # The USD steps are 2-5 cm slabs floating over open risers; the toe catches
                # under the next slab and the robot trips. The visual stairs are solid to the
                # floor, so fill each step down to z=0.
                top = pos[2] + half[2]
                half[2], pos[2] = top / 2, top / 2
            root.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half,
                pos=pos,
                quat=[q.GetReal(), *q.GetImaginary()],
                material="wood" if "pallet" in path else "steel",
            )
            continue
        # Meshes: bake the world transform into the points (row-vector convention).
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        m = np.array(xf)
        pts = pts @ m[:3, :3] + m[3, :3]
        if "brick_road" in path:
            _add_heightfield(spec, name, pts)
        else:  # the ramp: a wedge, its convex hull is exact enough (and all MuJoCo collides with)
            pts = pts[ConvexHull(pts).vertices]
            spec.add_mesh(name=name, uservert=pts.astype(np.float32).ravel())
            root.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, material="wood")
    return spec


if __name__ == "__main__":
    sha = subprocess.run(["git", "-C", MAD_ASSETS, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    spec = course_spec()
    spec.compile()
    out = HERE / "course.xml"
    if "--check" in sys.argv:  # the geometry, not the header: mad_assets commits that leave it alone pass
        stale = out.read_text().split("\n", 1)[1] != spec.to_xml()
        print(f"{out} {'is stale, rerun without --check' if stale else 'matches'} mad_assets {sha}")
        raise SystemExit(stale)
    out.write_text(f"<!-- Generated by bake_course.py from mad_assets {sha} scenes/default.usd. -->\n{spec.to_xml()}")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
