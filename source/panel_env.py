"""Controlled tabletop text-state environment with PyBullet Panda contacts.

The simulator is the data generator/evaluator, never a predictor input. The
robot executes a fixed press primitive; displayed values change upon measured
button depression, rather than on receipt of a requested command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "panel-assets"
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]


def font(size, index=0):
    found = [f for f in FONTS if Path(f).exists()]
    if not found:
        raise RuntimeError("No readable system font; install fonts in private assets")
    return ImageFont.truetype(found[index % len(found)], size)


def assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    mesh = ASSETS / "screen.obj"
    if not mesh.exists():
        mesh.write_text("v -0.105 -0.105 0\nv 0.105 -0.105 0\nv 0.105 0.105 0\nv -0.105 0.105 0\n"
                        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
                        "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n", encoding="utf-8")
    button = ASSETS / "button.urdf"
    if not button.exists():
        button.write_text('''<robot name="button">
<link name="base"><inertial><mass value="0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
<link name="cap"><inertial><mass value="0.02"/><inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/></inertial>
<visual><geometry><cylinder radius="0.022" length="0.018"/></geometry><material name="red"><color rgba="0.8 0.15 0.1 1"/></material></visual>
<collision><geometry><cylinder radius="0.022" length="0.018"/></geometry></collision></link>
<joint name="press" type="prismatic"><parent link="base"/><child link="cap"/><axis xyz="0 0 1"/>
<limit lower="-0.018" upper="0" effort="20" velocity="0.2"/><dynamics damping="0.2" friction="0.01"/></joint>
</robot>''', encoding="utf-8")
    return mesh, button


class PanelEnv:
    """Three independently addressable panels and a position-controlled Panda."""
    def __init__(self, seed=0, family="counter", width=768, height=640, style=0):
        self.rng = np.random.default_rng(seed)
        self.family, self.style = family, style
        self.width, self.height = width, height
        self.client = p.connect(p.DIRECT)
        self._texture_prefix = f"{os.getpid()}-{seed}-{self.client}"
        self._render_serial = 0
        self.mesh, self.button_urdf = assets()
        self.tick = 0
        self.ids = ["".join(self.rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"), 5)) for _ in range(3)]
        self.values = self.rng.integers(15, 70, size=3).astype(float)
        self.gains = self.rng.choice([1, 2, 3], size=3)
        self.modes = np.zeros(3, dtype=int)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1/240, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.client)
        self._box([0.7, 0.65, 0.025], [0, 0, 0.575], [0.40, 0.43, 0.46, 1])
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, -0.43, 0.60],
            p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True,
            physicsClientId=self.client)
        self.rest = [0, -0.6, 0, -2.1, 0, 1.5, 0.8]
        for j, q in enumerate(self.rest):
            p.resetJointState(self.robot, j, q, physicsClientId=self.client)
        for j in [9, 10]:
            p.resetJointState(self.robot, j, 0.0, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, 0, force=20, physicsClientId=self.client)
        self.xs = [-0.26, 0, 0.26]
        self.buttons, self.screens, self.textures = [], [], []
        for i, x in enumerate(self.xs):
            self._box([0.12, 0.17, 0.035], [x, 0.17, 0.635], [0.1, 0.12, 0.16, 1])
            vis = p.createVisualShape(p.GEOM_MESH, fileName=str(self.mesh),
                    rgbaColor=[1, 1, 1, 1], physicsClientId=self.client)
            body = p.createMultiBody(0, -1, vis, [x, 0.13, 0.672], physicsClientId=self.client)
            self.screens.append(body)
            path = ASSETS / f"texture-{self._texture_prefix}-{i}-initial.png"
            Image.new("RGB", (512, 512), "white").save(path)
            tex = p.loadTexture(str(path), physicsClientId=self.client)
            self.textures.append(tex)
            p.changeVisualShape(body, -1, textureUniqueId=tex, physicsClientId=self.client)
            button = p.loadURDF(str(self.button_urdf), [x, 0.30, 0.686], useFixedBase=True,
                                physicsClientId=self.client)
            self.buttons.append(button)
            p.setJointMotorControl2(button, 0, p.POSITION_CONTROL, 0, force=5,
                                   positionGain=0.2, physicsClientId=self.client)
        self.occ = self._box([0.14, 0.14, 0.025], [0, 0.13, -3], [.13, .15, .18, 1], collision=False)
        self.occluded = -1
        self.view = p.computeViewMatrix([0.70, -0.65, 1.45], [0, 0.15, 0.65], [0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(48, self.width/self.height, 0.05, 4)
        self._refresh()

    def _box(self, half, xyz, color, collision=True):
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client) if collision else -1
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(0, c, v, xyz, physicsClientId=self.client)

    def _refresh(self):
        self._render_serial += 1
        for i, texture in enumerate(self.textures):
            bg = [(244, 247, 240), (222, 239, 248), (250, 241, 220)][self.style % 3]
            image = Image.new("RGB", (512, 512), bg)
            d = ImageDraw.Draw(image)
            d.rounded_rectangle([8, 8, 504, 504], radius=12, outline=(20, 35, 45), width=6)
            d.text((256, 95), self.ids[i], font=font(70, self.style), anchor="mm", fill=(15, 25, 30))
            d.text((256, 263), f"{int(round(self.values[i])):03d}", font=font(142, self.style),
                   anchor="mm", fill=(10, 18, 12))
            d.text((256, 422), self.family.upper(), font=font(34, self.style), anchor="mm", fill=(60, 65, 60))
            # TinyRenderer does not refresh via changeTexture on every build.
            # A new texture path ensures the CPU renderer receives the pixels.
            path = ASSETS / f"texture-{self._texture_prefix}-{i}-render-{self._render_serial}.png"
            image.save(path)
            tex = p.loadTexture(str(path), physicsClientId=self.client)
            p.changeVisualShape(self.screens[i], -1, textureUniqueId=tex, physicsClientId=self.client)

    def _move(self, xyz, steps=90, monitor_target=None):
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        q = p.calculateInverseKinematics(self.robot, 11, xyz, orn,
                maxNumIterations=150, residualThreshold=1e-5, physicsClientId=self.client)
        minimum, contacts = 0.0, 0
        self.last_press_event = False
        for _ in range(steps):
            p.setJointMotorControlArray(self.robot, list(range(7)), p.POSITION_CONTROL,
                targetPositions=q[:7], forces=[100]*7, positionGains=[0.25]*7,
                physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
            buttons = self.buttons if monitor_target is None else [self.buttons[monitor_target]]
            for b in buttons:
                depression = p.getJointState(b, 0, physicsClientId=self.client)[0]
                count = len(p.getContactPoints(self.robot, b, physicsClientId=self.client))
                minimum = min(minimum, depression)
                contacts += count
                if monitor_target is not None and depression < -0.003 and count:
                    self.last_press_event = True
        actual = p.getLinkState(self.robot, 11, physicsClientId=self.client)[4]
        self.last_ee = list(actual)
        return minimum, contacts

    def step(self, target=-1, dt=1):
        """Press target or wait; update displays using actual contact observations."""
        before = self.state()
        depressed, contacts = 0.0, 0
        if target >= 0:
            x = self.xs[target]
            self._move([x, .30, .86], 90)
            depressed, contacts = self._move([x, .30, .668], 120, monitor_target=target)
            pressed = self.last_press_event
            self._move([x, .30, .90], 90)
        else:
            pressed = False
        if pressed:
            if self.family == "counter":
                self.values[target] += self.gains[target]
            elif self.family == "toggle":
                self.modes[target] = 1-self.modes[target]
                self.values[target] += self.gains[target] * (1 if self.modes[target] else -1)
            elif self.family == "flow":
                self.modes[target] = 1-self.modes[target]
            else:
                raise ValueError(self.family)
        if self.family == "flow":
            self.values += self.modes * self.gains * dt
        self.values = np.clip(self.values, 0, 999)
        self.tick += dt
        self._refresh()
        return {"command": {"target": target, "dt": dt}, "pressed": bool(pressed),
                "min_depression_m": float(depressed), "contact_count": contacts,
                "ee_after_primitive": getattr(self, "last_ee", None),
                "before": before, "after": self.state()}

    def hide(self, index=-1):
        self.occluded = index
        xyz = [self.xs[index], .13, .78] if index >= 0 else [0, .13, -3]
        p.resetBasePositionAndOrientation(self.occ, xyz, [0, 0, 0, 1], physicsClientId=self.client)

    def state(self):
        return {"ids": list(self.ids), "values": np.rint(self.values).astype(int).tolist(),
                "modes": self.modes.tolist(), "tick": self.tick}

    def image(self):
        out = p.getCameraImage(self.width, self.height, self.view, self.proj,
                              renderer=p.ER_TINY_RENDERER, shadow=1, physicsClientId=self.client)
        return Image.fromarray(np.asarray(out[2], dtype=np.uint8)[..., :3])

    def close(self):
        p.disconnect(self.client)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        out = ROOT / "data" / "panel-audit"
        out.mkdir(parents=True, exist_ok=True)
        env = PanelEnv(seed=802, family="counter")
        try:
            env.image().save(out / "before.png")
            records = [env.step(i) for i in range(3)]
            env.image().save(out / "after.png")
            env.hide(0)
            env.image().save(out / "occluded.png")
            (out / "events.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(json.dumps({"events": records, "output": str(out)}))
        finally:
            env.close()


if __name__ == "__main__":
    main()
