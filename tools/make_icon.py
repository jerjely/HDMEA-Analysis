"""Draw the HDMEA-Analysis app icon: rounded navy tile, faint electrode grid,
one bright extracellular spike waveform. Writes PNG 512/256, .ico and .icns."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2048                       # supersampled canvas
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "hdmea", "resources")
os.makedirs(OUT, exist_ok=True)

def lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

# --- tile with vertical gradient #101d3d -> #060b1a
grad = Image.new("RGB", (S, S))
top, bot = (0x10, 0x1d, 0x3d), (0x06, 0x0b, 0x1a)
arr = np.zeros((S, S, 3), np.uint8)
for y in range(S):
    arr[y, :] = lerp(top, bot, y / (S - 1))
grad = Image.fromarray(arr)
mask = Image.new("L", (S, S), 0)
m = int(S * 0.04); r = int(S * 0.20)
ImageDraw.Draw(mask).rounded_rectangle([m, m, S - m, S - m], radius=r, fill=255)
tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
tile.paste(grad, (0, 0), mask)

# --- faint 8x8 electrode grid
g = ImageDraw.Draw(tile)
n = 8; x0, x1 = S * 0.20, S * 0.80
step = (x1 - x0) / (n - 1)
for i in range(n):
    for j in range(n):
        cx, cy = x0 + i * step, x0 + j * step
        rr = S * 0.012
        g.rounded_rectangle([cx - rr, cy - rr, cx + rr, cy + rr], radius=rr * 0.3, fill=(70, 110, 170, 150))

# --- bright spike waveform (sharp negative trough, slower positive rebound)
t = np.linspace(-1, 1, 900)
w = -1.0 * np.exp(-((t + 0.02) / 0.075) ** 2) + 0.38 * np.exp(-((t - 0.22) / 0.17) ** 2) \
    - 0.06 * np.exp(-((t + 0.25) / 0.10) ** 2)
xs = S * 0.13 + (t + 1) / 2 * S * 0.74
ys = S * 0.47 - w * S * 0.30
# resample by arc length so stamped discs give a clean, even stroke
seg = np.hypot(np.diff(xs), np.diff(ys)); L = np.r_[0, np.cumsum(seg)]
u = np.linspace(0, L[-1], int(L[-1] / 2))
xs, ys = np.interp(u, L, xs), np.interp(u, L, ys)
pts = list(zip(xs, ys))


def stroke(draw, width, fill):
    rr = width / 2
    for x, y in pts:
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
stroke(gd, S * 0.075, (60, 230, 255, 255))
glow = glow.filter(ImageFilter.GaussianBlur(S * 0.03))
glow.putalpha(glow.getchannel("A").point(lambda a: int(a * 0.55)))
tile = Image.alpha_composite(tile, glow)
d = ImageDraw.Draw(tile)
stroke(d, S * 0.042, (120, 240, 255, 255))
stroke(d, S * 0.016, (235, 252, 255, 255))
# keep everything inside the rounded tile
tile.putalpha(Image.fromarray(np.minimum(np.array(tile.getchannel("A")), np.array(mask))))

big = tile.resize((1024, 1024), Image.LANCZOS)
big.resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, "icon_512.png"))
big.resize((256, 256), Image.LANCZOS).save(os.path.join(OUT, "icon_256.png"))
big.save(os.path.join(OUT, "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
big.save(os.path.join(OUT, "icon.icns"))
print("icons written to", os.path.abspath(OUT))
