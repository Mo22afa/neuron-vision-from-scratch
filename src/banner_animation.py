"""Generate a GitHub-compatible animated neon banner from the original SVG.

Run: python -m src.banner_animation
"""
from io import BytesIO
from math import pi, sin
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFilter


ORIGINAL_WIDTH = 1400
ORIGINAL_HEIGHT = 520
SCALE = 0.72
FRAMES = 20


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (
        u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
        u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1],
    )


def xy(point):
    return (round(point[0]*SCALE), round(point[1]*SCALE))


def draw_light_trail(d, points, center, color, max_width=8):
    for k in range(12):
        t = (center - 0.043*(11-k)) % 1.0
        if t > center:
            continue
        px, py = xy(bezier(*points, t))
        radius = max(1, round(max_width * (k+1)/12 * SCALE))
        d.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)


def render():
    source = Path("assets/neuron-vision-banner.svg")
    if not source.is_file():
        raise FileNotFoundError(f"Missing banner source: {source}")

    size = (round(ORIGINAL_WIDTH*SCALE), round(ORIGINAL_HEIGHT*SCALE))
    png = cairosvg.svg2png(
        url=str(source), output_width=size[0], output_height=size[1]
    )
    base = Image.open(BytesIO(png)).convert("RGBA")

    inputs = [(145,126), (247,264), (383,310), (213,218), (349,402), (281,356)]
    outputs = [(126,161), (264,269), (310,323), (402,377), (218,215)]
    paths = [
        ((887,a),(944,a),(943,b),(1007,b)) for a,b in inputs
    ] + [
        ((1007,a),(1072,a),(1080,b),(1163,b)) for a,b in outputs
    ]
    colors = [(64,238,255,235)]*len(inputs) + [(203,130,255,235)]*len(outputs)

    frames = []
    for frame_i in range(FRAMES):
        phase = frame_i / FRAMES
        neon = Image.new("RGBA", size, (0,0,0,0))
        draw = ImageDraw.Draw(neon)

        # Signals travel along the same input/hidden/output connections
        # represented in the original SVG.
        for j, (path, color) in enumerate(zip(paths, colors)):
            progress = (phase*(1.0+0.08*(j%3)) + 0.121*j) % 1
            draw_light_trail(draw, path, progress, color)

        # Traveling title underline glow.
        title_x = 72 + 552*phase
        title_y = round(260*SCALE)
        for j in range(14):
            x = round((title_x-j*11)*SCALE)
            if round(72*SCALE) <= x <= round(624*SCALE):
                draw.ellipse(
                    (x-2,title_y-2,x+2,title_y+2),
                    fill=(83,245,238,round(240*(1-j/14)))
                )

        # Subtle pulses around selected neurons.
        for j,(px,py) in enumerate(
            [(1007,310),(1163,269),(1007,172),(887,247)]
        ):
            wave = (sin(2*pi*(phase-0.23*j))+1)/2
            radius = round((10+7*wave)*SCALE)
            x,y = xy((px,py))
            if j % 2 == 0:
                color = (60,255,220,round(130*wave))
            else:
                color = (199,120,255,round(130*wave))
            draw.ellipse(
                (x-radius,y-radius,x+radius,y+radius),
                outline=color, width=2
            )

        # Electric highlight races around the network panel's perimeter.
        left,top,right,bottom = (
            round(v*SCALE) for v in (764,62,1305,461)
        )
        perimeter = 2*((right-left)+(bottom-top))
        cursor = phase*perimeter
        previous = None
        for k in range(150):
            p = (cursor-k*2.2) % perimeter
            if p < right-left:
                point = (left+p,top)
            elif p < (right-left)+(bottom-top):
                point = (right,top+p-(right-left))
            elif p < 2*(right-left)+(bottom-top):
                point = (
                    right-(p-(right-left)-(bottom-top)),bottom
                )
            else:
                point = (
                    left,bottom-(p-2*(right-left)-(bottom-top))
                )
            if previous is not None:
                if abs(point[0]-previous[0])+abs(point[1]-previous[1]) < 8:
                    draw.line(
                        (previous,point),
                        fill=(79,234,255,round(170*(1-k/150))),
                        width=2
                    )
            previous = point

        glow = neon.filter(ImageFilter.GaussianBlur(5))
        frame = Image.alpha_composite(base,glow)
        frame = Image.alpha_composite(frame,neon)
        frames.append(
            frame.convert("RGB").quantize(
                colors=96, method=Image.Quantize.MEDIANCUT
            )
        )

    target = Path("assets/neuron-vision-animated.gif")
    target.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(
        target, save_all=True, append_images=frames[1:],
        duration=110, loop=0, optimize=True, disposal=2
    )
    with Image.open(target) as gif:
        assert getattr(gif,"n_frames",0) == FRAMES
    print(
        f"Generated {target} "
        f"({FRAMES} frames; {target.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    render()
