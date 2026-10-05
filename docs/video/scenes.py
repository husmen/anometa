"""Manim scenes for the Anometa explainer video.

One scene, ``Explainer``, plays the eight narration blocks of ``narration.md`` in
order. Each segment lasts exactly as long as ``out/audio/<scene>.wav`` (see
``out/audio/durations.json``, written by ``build.py``) and starts that WAV as its
soundtrack. Visual beats are cued to a phrase of the narration, timed by its
word position within the block. Render with ``build.py`` (it passes
``--disable_caching`` so that the scene clock counts real frames and audio
stays in sync).

Numbers: ``reports/lock/results.md`` (lock split) and the Track C prototype in
``docs/plans/PLAN_1_RUN_LOG.md`` (dev split). Only ``Text`` is used: no LaTeX.
"""

import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    AnimationGroup,
    Arrow,
    Axes,
    Brace,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowFromEdge,
    ImageMobject,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    Write,
    config,
)
from PIL import Image

HERE = Path(__file__).resolve().parent
AD2 = HERE.parents[1] / "data" / "ad2"
AUDIO = HERE / "out" / "audio"
LEAD_S, TAIL_S = 0.35, 0.85  # must match build.py

BG, FG, MUTED, CARD = "#0e1116", "#e8eaed", "#9aa3ad", "#1b212b"
AMBER, AMBER_2, BLUE, GREY, TEAL = "#f5a524", "#ffcf70", "#4c8bf5", "#8e959c", "#2ec4b6"
RED, CELL = "#ef5350", "#3a4452"
FONT = "Avenir Next"
SCENARIOS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def txt(s: str, size: float = 30, color: str = FG, bold: bool = False) -> VGroup:
    """Centred text in the video's font; one ``Text`` per line.

    Rendered at 4x and scaled down: Pango's kerning breaks at small font sizes.
    """
    weight = "BOLD" if bold else "NORMAL"
    lines = [
        Text(line, font=FONT, font_size=4 * size, color=color, weight=weight).scale(0.25)
        for line in s.split("\n")
    ]
    return VGroup(*lines).arrange(DOWN, buff=0.004 * size)


def photo(path: Path, height: float, square: bool = False, crop_top: int = 0) -> ImageMobject:
    """Load an image downscaled for speed; optionally square-cropped or cut at the top."""
    img = Image.open(path).convert("RGB")
    img = img.crop((0, crop_top, img.width, img.height))
    if square:
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img.thumbnail((900, 900))
    return ImageMobject(np.asarray(img)).scale_to_fit_height(height)


def vdots(color: str = MUTED) -> VGroup:
    """A vertical ellipsis drawn with dots (the font's glyph has a misleading bounding box)."""
    return VGroup(*(Dot(radius=0.035, color=color) for _ in range(3))).arrange(DOWN, buff=0.07)


def box(
    label: str, color: str, width: float = 3.0, height: float = 1.1, size: float = 26
) -> VGroup:
    """A rounded diagram box with a centred label."""
    rect = RoundedRectangle(
        corner_radius=0.15, width=width, height=height, stroke_color=color, stroke_width=3
    )
    rect.set_fill(CARD, 1)
    return VGroup(rect, txt(label, size).move_to(rect))


def cells(n: int, color: str, size: float = 0.26, shades: bool = False, seed: int = 0) -> VGroup:
    """A row of ``n`` square cells, optionally with random opacity like feature values."""
    rng = np.random.default_rng(seed)
    row = VGroup(*(Square(size, stroke_width=1, stroke_color=BG) for _ in range(n))).arrange(
        RIGHT, buff=0.03
    )
    for c in row:
        c.set_fill(color, rng.uniform(0.35, 1.0) if shades else 1.0)
    return row


def blocks() -> dict[str, str]:
    """Narration blocks of ``narration.md`` as ``{scene: text}`` (same parser as build.py)."""
    parts = re.split(r"^## (\w+)\n", (HERE / "narration.md").read_text(), flags=re.M)[1:]
    return {
        name: " ".join(body.split()) for name, body in zip(parts[::2], parts[1::2], strict=True)
    }


class Explainer(Scene):
    """The full explainer: eight segments, each as long as its narration clip."""

    def construct(self) -> None:
        """Play every segment in narration order."""
        self.camera.background_color = BG
        self.durations: dict[str, float] = json.loads((AUDIO / "durations.json").read_text())[
            "durations"
        ]
        self.texts = blocks()
        self.start = 0.0
        for name in self.texts:
            self.seg_name, self.seg_start = name, self.start
            self.end = self.start + self.durations[name]
            self.add_sound(str(AUDIO / f"{name}.wav"))
            getattr(self, name)()
            self.finish()
            self.start = self.end

    # Timing helpers ------------------------------------------------------------

    def now(self) -> float:
        """Scene clock in seconds; counts written frames when caching is disabled."""
        return self.renderer.time

    def cue(self, phrase: str) -> float:
        """Estimated time the current block's narration reaches ``phrase``."""
        text = self.texts[self.seg_name]
        idx = text.index(phrase)
        words = len(text.split())
        speech = self.durations[self.seg_name] - LEAD_S - TAIL_S
        return self.seg_start + LEAD_S + speech * len(text[:idx].split()) / words

    def until(self, t: float) -> None:
        """Hold the frame until scene time ``t`` (no-op if already past it)."""
        if (gap := t - self.now()) >= 1 / config.frame_rate:
            self.wait(gap)

    def at(self, phrase: str, *anims: object, run_time: float = 1.0, lead: float = 0.2) -> None:
        """Play ``anims`` starting just before the narration reaches ``phrase``."""
        self.until(self.cue(phrase) - lead)
        self.play(*anims, run_time=run_time)

    def title(self, s: str) -> VGroup:
        """Segment title at the top left."""
        t = txt(s, 36, bold=True).to_edge(UL, buff=0.5)
        self.play(FadeIn(t, shift=0.2 * DOWN), run_time=0.6)
        return t

    def finish(self) -> None:
        """Fade everything out and hold until the segment's audio ends."""
        fade = 0.5
        self.until(self.end - fade - 1 / config.frame_rate)
        if self.mobjects:
            self.play(
                *(FadeOut(m) for m in self.mobjects),
                run_time=min(fade, max(self.end - self.now(), 1 / config.frame_rate)),
            )
        self.until(self.end)

    # Segments --------------------------------------------------------------------

    def question(self) -> None:
        """Title and the few-labels question."""
        name = txt("Anometa", 96, bold=True)
        sub = txt("Few-shot industrial anomaly detection with TabPFN-3.5", 32, MUTED)
        VGroup(name, sub).arrange(DOWN, buff=0.4)
        self.play(Write(name), FadeIn(sub, shift=0.2 * UP), run_time=1.5)
        self.until(self.cue("Often") - 0.6)
        self.play(FadeOut(sub), name.animate.scale(0.45).to_edge(UL, buff=0.5), run_time=0.8)

        goods = VGroup(*(Square(0.2, stroke_width=0).set_fill(CELL, 1) for _ in range(300)))
        goods.arrange_in_grid(rows=12, cols=25, buff=0.06).move_to(2.3 * LEFT + 0.4 * UP)
        bad = Square(0.2, stroke_width=0).set_fill(RED, 1).next_to(goods, RIGHT, buff=1.6)
        g_lab = txt("≈300 good images", 28, MUTED).next_to(goods, DOWN, buff=0.3)
        b_lab = txt("1 to 5 labelled\ndefects", 28, RED).next_to(bad, DOWN, buff=0.3)
        self.play(
            LaggedStart(*(FadeIn(s) for s in goods), lag_ratio=0.004), FadeIn(g_lab), run_time=1.6
        )
        self.play(FadeIn(bad, scale=3), FadeIn(b_lab), run_time=0.8)

        q = txt("Can a model learn to spot new defects from that little?", 34).to_edge(
            DOWN, buff=1.3
        )
        self.at("Can a model", Write(q), run_time=1.5)
        tab = box(
            "TabPFN-3.5: a foundation model for tables", AMBER, width=8.6, height=0.9, size=30
        )
        tab.next_to(q, DOWN, buff=0.35)
        self.at("Anometa puts", FadeIn(tab, shift=0.2 * UP), run_time=0.8)

    def data(self) -> None:
        """The eight AD2 scenarios and one scene under three lightings."""
        self.title("Data: MVTec AD 2, 8 industrial scenarios")
        tiles = Group()
        for s in SCENARIOS:
            img = photo(AD2 / s / "test_public" / "good" / "000_regular.png", 2.2, square=True)
            tiles.add(Group(img, txt(s.replace("_", " "), 24, MUTED).next_to(img, DOWN, buff=0.12)))
        tiles.arrange_in_grid(rows=2, cols=4, buff=(0.45, 0.3)).shift(0.4 * DOWN)
        self.play(
            LaggedStart(*(FadeIn(t, shift=0.2 * UP) for t in tiles), lag_ratio=0.12), run_time=2.0
        )

        self.until(self.cue("The training images") - 0.4)
        lights = [("regular", "train + test"), ("shift_1", "test only"), ("shift_2", "test only")]
        row = Group()
        for light, use in lights:
            img = photo(
                AD2 / "wallplugs" / "test_public" / "good" / f"000_{light}.png", 3.4, square=True
            )
            name = {"shift_1": "lighting shift 1", "shift_2": "lighting shift 2"}.get(light, light)
            lab = VGroup(
                txt(name, 28), txt(use, 24, AMBER if use == "test only" else MUTED)
            ).arrange(DOWN, buff=0.1)
            row.add(Group(img, lab.next_to(img, DOWN, buff=0.2)))
        row.arrange(RIGHT, buff=0.5).shift(0.5 * DOWN)
        self.play(FadeOut(tiles), FadeIn(row[0]), run_time=0.8)
        note = txt("Wallplugs: the same scene under three lightings", 26, MUTED).next_to(
            row, UP, buff=0.3
        )
        self.play(FadeIn(note), run_time=0.5)
        self.at(
            "The test images",
            LaggedStart(FadeIn(row[1]), FadeIn(row[2]), lag_ratio=0.5),
            run_time=1.2,
        )
        defect = txt("Test images also contain defects, never seen in training", 26, RED)
        self.at("add defects", FadeIn(defect.to_edge(DOWN, buff=0.35)), run_time=0.6)

    def protocol(self) -> None:
        """Split by scene into dev and lock halves, freeze, evaluate the lock half once."""
        self.title("Protocol: split by scene, evaluate once")
        rng = np.random.default_rng(3)

        def scene_stack() -> VGroup:
            return VGroup(
                *(Square(0.34, stroke_width=1, stroke_color=BG).set_fill(CELL, 1) for _ in range(3))
            ).arrange(DOWN, buff=0.04)

        stacks = (
            VGroup(*(scene_stack() for _ in range(10)))
            .arrange(RIGHT, buff=0.35)
            .shift(1.8 * UP)
            .set_z_index(2)
        )
        lab = txt(
            "labelled test images of one scenario, grouped by scene"
            " (one column = one scene, all lightings)",
            22,
            MUTED,
        )
        lab.next_to(stacks, UP, buff=0.3)
        self.play(
            LaggedStart(*(FadeIn(s, shift=0.2 * DOWN) for s in stacks), lag_ratio=0.08),
            FadeIn(lab),
            run_time=1.4,
        )

        dev_box = box("", BLUE, width=5.4, height=2.0)
        lock_box = box("", AMBER, width=5.4, height=2.0)
        VGroup(dev_box, lock_box).arrange(RIGHT, buff=1.8).shift(0.5 * DOWN)
        dev_t = txt("dev half", 30, BLUE, bold=True).next_to(dev_box, UP, buff=0.12)
        lock_t = txt("lock half", 30, AMBER, bold=True).next_to(lock_box, UP, buff=0.12)
        order = rng.permutation(10)
        dev_s, lock_s = [stacks[i] for i in order[:5]], [stacks[i] for i in order[5:]]
        targets: list[object] = []
        for group, b in ((dev_s, dev_box), (lock_s, lock_box)):
            slots = VGroup(*(scene_stack() for _ in group)).arrange(RIGHT, buff=0.3).move_to(b)
            targets += [s.animate.move_to(slot) for s, slot in zip(group, slots, strict=True)]
        self.at(
            "split by scene",
            FadeOut(lab),
            Create(dev_box[0]),
            Create(lock_box[0]),
            FadeIn(dev_t),
            FadeIn(lock_t),
            *targets,
            run_time=1.6,
        )
        dev_use = txt("every experiment and model choice", 24, FG).next_to(dev_box, DOWN, buff=0.25)
        self.at("All experiments", FadeIn(dev_use, shift=0.1 * UP), run_time=0.7)

        frozen = box("frozen config", FG, width=2.6, height=0.8, size=24).move_to(
            (dev_box.get_right() + lock_box.get_left()) / 2 + 2.2 * DOWN
        )
        a1 = Arrow(dev_use.get_bottom(), frozen.get_left(), buff=0.1, color=MUTED, stroke_width=3)
        self.at("then frozen", FadeIn(frozen), Create(a1), run_time=0.8)
        once = txt("evaluated exactly once", 26, AMBER, bold=True).next_to(
            lock_box, DOWN, buff=0.25
        )
        a2 = Arrow(
            frozen.get_right(),
            once.get_bottom() + 0.1 * DOWN,
            buff=0.1,
            color=MUTED,
            stroke_width=3,
        )
        self.at("evaluated exactly once", Create(a2), FadeIn(once), run_time=0.8)
        self.play(Indicate(once, color=AMBER_2), run_time=0.8)

    def pipeline(self) -> None:
        """Image to patches to frozen DINOv3 to one row of 18 numbers."""
        self.title("Pipeline: one image, one row of 18 numbers")
        img = photo(AD2 / "walnuts" / "test_public" / "good" / "000_regular.png", 2.8, square=True)
        img.move_to(5.2 * LEFT + 0.9 * UP)
        n = 14
        grid = VGroup(
            *(
                Line(
                    img.get_corner(UL) + RIGHT * img.width * i / n,
                    img.get_corner(UL) + RIGHT * img.width * i / n + DOWN * img.height,
                )
                for i in range(1, n)
            ),
            *(
                Line(
                    img.get_corner(UL) + DOWN * img.height * i / n,
                    img.get_corner(UL) + DOWN * img.height * i / n + RIGHT * img.width,
                )
                for i in range(1, n)
            ),
        ).set_stroke(FG, 1, opacity=0.6)
        patch_lab = txt("16 \N{MULTIPLICATION SIGN} 16 px patches", 24, MUTED).next_to(
            img, DOWN, buff=0.15
        )
        self.play(FadeIn(img), run_time=0.6)
        self.play(Create(grid), FadeIn(patch_lab), run_time=1.0)

        vit = box(
            "frozen DINOv3-L\nvision transformer", FG, width=3.3, height=1.4, size=26
        ).move_to(1.4 * LEFT + 0.9 * UP)
        a0 = Arrow(img.get_right(), vit.get_left(), buff=0.15, color=MUTED, stroke_width=3)
        self.at("frozen DINOv3", Create(a0), FadeIn(vit), run_time=0.8)

        summ = box(
            "global (CLS) token\n+ mean patch", BLUE, width=3.2, height=1.1, size=22
        ).move_to(2.2 * RIGHT + 2.0 * UP)
        pca = box("PCA → 16", BLUE, width=1.9, height=0.8, size=24).next_to(summ, RIGHT, buff=0.6)
        a1 = Arrow(vit.get_right(), summ.get_left(), buff=0.12, color=MUTED, stroke_width=3)
        a2 = Arrow(summ.get_right(), pca.get_left(), buff=0.08, color=MUTED, stroke_width=3)
        self.at("global token", Create(a1), FadeIn(summ), run_time=0.8)
        self.at("compressed", Create(a2), FadeIn(pca), run_time=0.7)

        nov = box(
            "novelty: distance of the most unusual\npatches to their nearest normal patch → 2",
            TEAL,
            width=5.6,
            height=1.1,
            size=21,
        ).move_to(3.4 * RIGHT + 0.1 * DOWN)
        a3 = Arrow(vit.get_right(), nov.get_left(), buff=0.12, color=MUTED, stroke_width=3)
        self.at("Two more numbers", Create(a3), FadeIn(nov), run_time=0.8)

        row = VGroup(cells(16, BLUE, 0.42, shades=True, seed=1), cells(2, TEAL, 0.42)).arrange(
            RIGHT, buff=0.12
        )
        row.move_to(2.2 * DOWN)
        brace = Brace(row, DOWN, color=MUTED)
        row_lab = txt("one row of 18 numbers per image", 28, bold=True).next_to(
            brace, DOWN, buff=0.1
        )
        self.at(
            "That gives",
            ReplacementTransform(pca.copy(), row[0]),
            ReplacementTransform(nov.copy(), row[1]),
            run_time=1.0,
        )
        self.play(FadeIn(brace), FadeIn(row_lab), run_time=0.6)

    def tabpfn(self) -> None:
        """A table of normal, defect and test rows, read by TabPFN in one forward pass."""
        self.title("TabPFN: reads a table, no training")
        rows = VGroup()
        spec = [("0", CELL)] * 4 + [("⋮", None)] + [("1", RED)] * 2 + [("?", FG)] * 3
        for i, (lab, color) in enumerate(spec):
            if color is None:
                r = VGroup(vdots())
            else:
                r = VGroup(
                    cells(18, BLUE if lab != "1" else RED, 0.21, shades=True, seed=10 + i),
                    txt(lab, 24, RED if lab == "1" else FG),
                )
                r.arrange(RIGHT, buff=0.35)
            rows.add(r)
        rows.arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to(2.7 * LEFT + 0.4 * DOWN)
        rows[4].shift((rows[0][0].get_center()[0] - rows[4].get_center()[0]) * RIGHT)
        rows[7:].shift(0.25 * DOWN)
        head = VGroup(
            txt("18 features", 22, MUTED).move_to(rows[0][0]),
            txt("label", 22, MUTED).move_to(rows[0][1]),
        )
        head.next_to(rows[0], UP, buff=0.2)
        head[1].align_to(rows[0][1], LEFT)
        normal_b = Brace(VGroup(*rows[:5]), LEFT, color=MUTED)
        defect_b = Brace(VGroup(*rows[5:7]), LEFT, color=RED)
        test_b = Brace(VGroup(*rows[7:]), LEFT, color=FG)
        nb = txt("≈300\nnormal", 22, MUTED).next_to(normal_b, LEFT, buff=0.1)
        db = txt("few\ndefects", 22, RED).next_to(defect_b, LEFT, buff=0.1)
        tb = txt("new\nimages", 22).next_to(test_b, LEFT, buff=0.1)
        self.play(
            FadeIn(head),
            LaggedStart(*(FadeIn(r) for r in rows[:5]), lag_ratio=0.15),
            FadeIn(normal_b),
            FadeIn(nb),
            run_time=1.2,
        )
        self.at(
            "few defect rows",
            LaggedStart(*(FadeIn(r) for r in rows[5:7]), lag_ratio=0.3),
            FadeIn(defect_b),
            FadeIn(db),
            run_time=0.8,
        )

        model = box("TabPFN-3.5\none forward pass", AMBER, width=3.0, height=1.4, size=26).move_to(
            2.5 * RIGHT + 0.4 * DOWN
        )
        ctx_arrow = Arrow(
            rows.get_right() + 0.05 * RIGHT, model.get_left(), buff=0.1, color=MUTED, stroke_width=3
        )
        self.at(
            "In one forward pass",
            LaggedStart(*(FadeIn(r) for r in rows[7:]), lag_ratio=0.2),
            FadeIn(test_b),
            FadeIn(tb),
            Create(ctx_arrow),
            FadeIn(model),
            run_time=1.2,
        )
        probs = VGroup(
            *(
                txt(f"p = {p:.2f}", 26, AMBER).move_to(
                    5.6 * RIGHT + rows[7 + i].get_center()[1] * UP
                )
                for i, p in enumerate((0.04, 0.91, 0.12))
            )
        )
        probs.align_to(model.get_right() + 1.3 * RIGHT, LEFT)
        out = Arrow(model.get_right(), probs.get_left(), buff=0.2, color=AMBER, stroke_width=3)
        ill = VGroup(
            txt("p(defect)", 22, AMBER, bold=True).next_to(probs, UP, buff=0.2),
            txt("example values", 18, MUTED).next_to(probs, DOWN, buff=0.15),
        )
        self.at("predicts the probability", Create(out), FadeIn(probs), FadeIn(ill), run_time=1.0)
        no = txt("no gradient updates  ·  no tuning", 30, AMBER).to_edge(DOWN, buff=0.45)
        self.at("There are no", FadeIn(no, shift=0.2 * UP), run_time=0.7)

    def results(self) -> None:
        """Lock AUROC versus k with label-free references, and balanced log loss at k = 1."""
        self.title("Lock results: image-level AUROC")
        ks = [1, 2, 5]
        ax = Axes(
            x_range=[0.5, 3.5, 1],
            y_range=[0.5, 0.85, 0.05],
            x_length=5.6,
            y_length=4.6,
            axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True},
            tips=False,
        ).move_to(2.95 * LEFT + 0.5 * DOWN)
        xt = VGroup(
            *(
                txt(str(k), 24, MUTED).next_to(ax.c2p(i + 1, 0.5), DOWN, buff=0.15)
                for i, k in enumerate(ks)
            )
        )
        yt = VGroup(
            *(
                txt(f"{v:.2f}", 20, MUTED).next_to(ax.c2p(0.5, v), LEFT, buff=0.12)
                for v in np.arange(0.5, 0.851, 0.05)
            )
        )
        xl = txt("labelled defects k", 24, MUTED).next_to(xt, DOWN, buff=0.2)
        yl = txt("AUROC, lock split", 24, MUTED).rotate(np.pi / 2).next_to(yt, LEFT, buff=0.2)
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), run_time=1.0)

        def series(
            vals: list[float], color: str, name: str, label_pos: object
        ) -> tuple[VGroup, VGroup]:
            pts = [ax.c2p(i + 1, v) for i, v in enumerate(vals)]
            line = VGroup(*(Line(a, b, color=color, stroke_width=5) for a, b in pairwise(pts)))
            dots = VGroup(*(Dot(p, radius=0.08, color=color) for p in pts))
            nums = VGroup(
                *(
                    txt(f"{v:.3f}", 20, color).next_to(p, label_pos, buff=0.12)
                    for p, v in zip(pts, vals, strict=True)
                )
            )
            lab = txt(name, 24, color, bold=True).next_to(pts[-1], RIGHT, buff=0.5)
            return VGroup(line, dots), VGroup(nums, lab)

        tab_l, tab_n = series([0.761, 0.778, 0.789], AMBER, "TabPFN-3.5", UP)
        log_l, log_n = series([0.683, 0.740, 0.745], BLUE, "logistic regression", DOWN)
        knn_l, knn_n = series([0.567, 0.608, 0.662], GREY, "kNN", DOWN)
        self.at(
            "TabPFN reaches", Create(tab_l), FadeIn(tab_n[0][0]), FadeIn(tab_n[1]), run_time=1.2
        )
        self.at(
            "zero point six eight",
            Create(log_l),
            FadeIn(log_n[0][0]),
            FadeIn(log_n[1]),
            run_time=1.0,
        )
        self.play(Create(knn_l), FadeIn(knn_n), run_time=0.8)
        self.at(
            "With five defects",
            FadeIn(tab_n[0][1:]),
            FadeIn(log_n[0][1:]),
            Indicate(tab_n[0][2], color=AMBER_2),
            run_time=1.0,
        )

        bars_title = VGroup(
            txt("balanced log loss, k = 1", 26, bold=True), txt("lower is better", 22, MUTED)
        )
        bars_title.arrange(DOWN, buff=0.08).move_to(4.75 * RIGHT + 2.45 * UP)
        base = Line(3.0 * RIGHT + 0.6 * DOWN, 6.5 * RIGHT + 0.6 * DOWN, color=MUTED, stroke_width=2)
        unit = 1.7  # scene units per unit of log loss
        bars = VGroup()
        for i, (name, v, color) in enumerate(
            (("TabPFN-3.5", 0.65, AMBER), ("logistic\nregression", 1.25, BLUE))
        ):
            bar = Rectangle(width=1.1, height=v * unit, stroke_width=0).set_fill(color, 1)
            bar.move_to(base.get_left() + (0.9 + 1.7 * i) * RIGHT, aligned_edge=DOWN)
            bars.add(
                VGroup(
                    bar,
                    txt(f"{v:.2f}", 26, color, bold=True).next_to(bar, UP, buff=0.1),
                    txt(name, 20, MUTED).next_to(bar, DOWN, buff=0.12),
                )
            )
        self.at(
            "far better calibrated",
            FadeIn(bars_title),
            Create(base),
            *(GrowFromEdge(b[0], DOWN) for b in bars),
            run_time=1.0,
        )
        self.play(*(FadeIn(b[1:]) for b in bars), run_time=0.5)

        band = Rectangle(
            width=ax.x_length, height=ax.c2p(0, 0.780)[1] - ax.c2p(0, 0.766)[1], stroke_width=0
        )
        band.set_fill(TEAL, 0.35).move_to(ax.c2p(2, 0.773))
        edges = VGroup(
            *(
                DashedLine(ax.c2p(0.5, v), ax.c2p(3.5, v), color=TEAL, stroke_width=2)
                for v in (0.766, 0.780)
            )
        )
        refs = VGroup(
            txt("teal band: label-free scores", 22, TEAL, bold=True),
            txt("(no defect labels used)", 20, TEAL),
            txt("Mahalanobis 0.766", 20),
            txt("TabPFN outlier 0.778", 20),
            txt("DINOv3 distance 0.780", 20),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        refs.move_to(4.75 * RIGHT + 2.45 * DOWN)
        self.at("A good label-free", FadeIn(band), Create(edges), FadeIn(refs), run_time=1.0)

    def maps(self) -> None:
        """Patch rows to defect maps, the Wall Plugs example and mean AU-PRO bars."""
        self.title("Pixel level: one row per patch, one defect map")
        rng = np.random.default_rng(7)
        n = 8
        patches = VGroup(
            *(Square(0.28, stroke_width=1, stroke_color=BG).set_fill(CELL, 1) for _ in range(n * n))
        )
        patches.arrange_in_grid(n, n, buff=0.02).move_to(4.8 * LEFT + 0.9 * UP)
        p_lab = txt("16 \N{MULTIPLICATION SIGN} 16 px patches", 22, MUTED).next_to(
            patches, DOWN, buff=0.15
        )
        prows = VGroup(
            *(cells(10, BLUE, 0.16, shades=True, seed=20 + i) for i in range(5))
        ).arrange(DOWN, buff=0.06)
        dots = vdots()
        prow_g = VGroup(prows, dots).arrange(DOWN, buff=0.06).move_to(1.9 * LEFT + 0.9 * UP)
        r_lab = txt("one row per patch", 22, MUTED).next_to(prow_g, DOWN, buff=0.15)
        model = box("TabPFN", AMBER, width=1.9, height=0.9, size=26).move_to(0.9 * RIGHT + 0.9 * UP)
        yy, xx = np.mgrid[0:n, 0:n]
        heat = np.exp(-((yy - 5) ** 2 + (xx - 2) ** 2) / 2.0) + 0.08 * rng.random((n, n))
        hmap = patches.copy().move_to(4.4 * RIGHT + 0.9 * UP)
        for c, h in zip(hmap, heat.ravel(), strict=True):
            c.set_fill(AMBER, float(np.clip(h, 0.06, 1)))
        h_lab = txt("defect map (illustration)", 22, MUTED).next_to(hmap, DOWN, buff=0.15)
        arrows = VGroup(
            Arrow(patches.get_right(), prow_g.get_left(), buff=0.15, color=MUTED, stroke_width=3),
            Arrow(prow_g.get_right(), model.get_left(), buff=0.15, color=MUTED, stroke_width=3),
            Arrow(model.get_right(), hmap.get_left(), buff=0.15, color=MUTED, stroke_width=3),
        )
        concept = VGroup(patches, p_lab, prow_g, r_lab, model, hmap, h_lab, arrows)
        self.play(FadeIn(patches), FadeIn(p_lab), run_time=0.6)
        self.at("becomes a row", Create(arrows[0]), FadeIn(prow_g), FadeIn(r_lab), run_time=0.8)
        self.at("TabPFN scores", Create(arrows[1]), FadeIn(model), run_time=0.6)
        self.play(
            Create(arrows[2]),
            LaggedStart(*(FadeIn(c) for c in hmap), lag_ratio=0.01),
            FadeIn(h_lab),
            run_time=1.0,
        )

        strip = photo(HERE / "assets" / "trackc_example.webp", 2.2, crop_top=36).move_to(1.25 * UP)
        strip.scale_to_fit_width(min(strip.width, 13.2))
        panels = [
            "image, defect outlined",
            "DINOv3 distance",
            "PatchCore",
            "fused, no labels",
            "TabPFN-Fast",
        ]
        s_lab = VGroup(
            *(
                txt(p, 18, AMBER if p == "TabPFN-Fast" else MUTED).move_to(
                    strip.get_corner(DOWN + LEFT) + strip.width * (i + 0.5) / 5 * RIGHT + 0.2 * DOWN
                )
                for i, p in enumerate(panels)
            )
        )
        self.at("On the dev split", FadeOut(concept), FadeIn(strip), FadeIn(s_lab), run_time=0.8)

        data = [
            ("TabPFN-3.5-Fast", 0.468, AMBER_2),
            ("TabPFN-3.5", 0.460, AMBER),
            ("logistic regression", 0.402, BLUE),
            ("DINOv3 distance", 0.397, TEAL),
            ("fused, no labels", 0.353, TEAL),
            ("PatchCore", 0.189, TEAL),
        ]
        unit = 9.0  # scene units per unit of AU-PRO
        x0 = 2.4 * LEFT
        bars = VGroup()
        for i, (name, v, color) in enumerate(data):
            y = -0.7 - 0.43 * i
            bar = Rectangle(width=v * unit, height=0.32, stroke_width=0).set_fill(
                color, 1 if color != TEAL else 0.75
            )
            bar.move_to(x0 + y * UP, aligned_edge=LEFT)
            name_t = txt(name, 22, color).next_to(bar, LEFT, buff=0.2)
            name_t.align_to(x0 + 0.2 * LEFT, RIGHT)
            bars.add(
                VGroup(
                    bar,
                    name_t,
                    txt(f"{v:.3f}", 22, color, bold=True).next_to(bar, RIGHT, buff=0.15),
                )
            )
        cap = txt(
            "Wall Plugs example above · bars: mean AU-PRO@0.05 (higher is better), k = 5,"
            " dev split, 8 scenarios \N{MULTIPLICATION SIGN} 3 seeds",
            18,
            MUTED,
        )
        cap.to_edge(DOWN, buff=0.25)
        self.play(
            LaggedStart(
                *(
                    AnimationGroup(GrowFromEdge(b[0], LEFT), FadeIn(b[1]), FadeIn(b[2]))
                    for b in bars
                ),
                lag_ratio=0.15,
            ),
            FadeIn(cap),
            run_time=1.8,
        )
        self.at(
            "beat both",
            Indicate(bars[0], color=AMBER_2, scale_factor=1.04),
            Indicate(bars[1], color=AMBER, scale_factor=1.04),
            run_time=1.0,
        )

    def takeaways(self) -> None:
        """Speed, cross-hardware reproduction and the open-source line."""
        self.title("Takeaways")
        specs = [
            ("fitting takes", "4 ms", "TabPFN fit on an RTX 3090,\nno training"),
            ("one scenario runs", "14 s", "Vial end to end, RTX 3090\nEfficientAD-S: 44 min"),
            ("Results match", "≤ 0.005", "AUROC gap to RTX 3090\non M4 Pro and on CPU"),
        ]
        cards = VGroup()
        for _, big, small in specs:
            rect = RoundedRectangle(
                corner_radius=0.2, width=4.2, height=3.0, stroke_color=CELL, stroke_width=2
            )
            rect.set_fill(CARD, 1)
            content = (
                VGroup(txt(big, 60, AMBER, bold=True), txt(small, 22, FG))
                .arrange(DOWN, buff=0.35)
                .move_to(rect)
            )
            cards.add(VGroup(rect, content))
        cards.arrange(RIGHT, buff=0.35).shift(0.3 * UP)
        for (phrase, _, _), card in zip(specs, cards, strict=True):
            self.at(phrase, FadeIn(card, shift=0.2 * UP), run_time=0.7)
        foot = (
            VGroup(
                txt("Anometa: open source, Apache-2.0 · every run recorded with its manifest", 26),
                txt("TabPFN-3.5, DINOv3 and MVTec AD 2 keep their own licences", 20, MUTED),
            )
            .arrange(DOWN, buff=0.15)
            .to_edge(DOWN, buff=0.6)
        )
        self.at("Anometa is open source", FadeIn(foot, shift=0.2 * UP), run_time=0.8)
