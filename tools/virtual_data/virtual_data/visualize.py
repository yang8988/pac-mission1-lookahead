"""2D top-view SVG of a 5-1/5-2 decision (mission: 2D/3D visualisation).

Shows the pallet heightmap (placed boxes shaded by top height), every 5-1
candidate footprint coloured by its 5-2 verdict, the EMS used for the chosen
candidate, and a legend with "N generated -> M masked -> K valid" and the
mask reasons. Pure SVG, no plotting dependency.
"""

from collections import Counter
import html

from pac_candidates.geometry import rotated_dims

VALID = "#2e9d5b"
MASKED = "#d64545"
CHOSEN = "#1f5fbf"
EMS = "#8a5cd6"
PALLET = "#f7f3ea"


def _shade(top, max_h):
    t = 0.0 if max_h <= 0 else min(1.0, top / max_h)
    v = int(225 - 150 * t)
    return f"rgb({v},{v},{min(255, v + 20)})"


def decision_svg(state, box, candidate_set, chosen_id=None, title=None, width=900):
    p = state.pallet.size
    margin = 30
    plot = width - 2 * margin - 280  # right side: legend
    scale = plot / max(p.x, p.y)
    height = int(margin * 2 + p.y * scale + 40)

    def X(x):
        return margin + x * scale

    def Y(y):  # SVG y grows downward; pallet y grows upward
        return margin + 40 + (p.y - y) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin}" y="{margin + 12}" font-size="16" fill="#222">'
        f"{html.escape(title or f'5-1/5-2 decision: box {box.box_id}')}</text>",
        f'<rect x="{X(0)}" y="{Y(p.y)}" width="{p.x * scale}" height="{p.y * scale}" '
        f'fill="{PALLET}" stroke="#555" stroke-width="2"/>',
    ]
    max_top = max((b.pose.z + rotated_dims(b.size, b.pose.yaw)[2] for b in state.pallet.boxes), default=0.0)
    for b in sorted(state.pallet.boxes, key=lambda q: q.pose.z):
        dx, dy, dz = rotated_dims(b.size, b.pose.yaw)
        top = b.pose.z + dz
        parts.append(
            f'<rect x="{X(b.pose.x):.1f}" y="{Y(b.pose.y + dy):.1f}" width="{dx * scale:.1f}" '
            f'height="{dy * scale:.1f}" fill="{_shade(top, p.z)}" stroke="#333" stroke-width="1">'
            f"<title>{html.escape(b.box_id)} top={top:.3f} m, {b.weight_kg:.1f} kg</title></rect>"
        )
        parts.append(
            f'<text x="{X(b.pose.x) + 3:.1f}" y="{Y(b.pose.y + dy) + 12:.1f}" font-size="9" '
            f'fill="#222">{top:.2f}</text>'
        )
    generation = candidate_set.generation
    rejected = candidate_set.rejected
    reasons = Counter()
    chosen = None
    for cand in generation.candidates:
        pose = cand.target_pose
        dx, dy, _ = rotated_dims(box.size, pose.yaw)
        verdict = rejected.get(cand.candidate_id)
        ok = verdict is None
        if not ok:
            reasons.update({r.split(":", 1)[0] for r in verdict.details.get("reasons", ())})
        if cand.candidate_id == chosen_id:
            chosen = (cand, dx, dy)
            continue
        color = VALID if ok else MASKED
        tip = cand.candidate_id + (
            " valid" if ok else " masked: " + ", ".join(verdict.details.get("reasons", ()))
        )
        parts.append(
            f'<rect x="{X(pose.x):.1f}" y="{Y(pose.y + dy):.1f}" width="{dx * scale:.1f}" '
            f'height="{dy * scale:.1f}" fill="{color}" fill-opacity="0.07" stroke="{color}" '
            f'stroke-opacity="0.8" stroke-width="1" stroke-dasharray="{"" if ok else "3,2"}">'
            f"<title>{html.escape(tip)} z={pose.z:.3f}</title></rect>"
        )
        cx, cy = X(pose.x + dx / 2), Y(pose.y + dy / 2)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{color}"/>')
    if chosen is not None:
        cand, dx, dy = chosen
        pose = cand.target_pose
        info = generation.infos.get(cand.candidate_id)
        if info is not None and info.ems_lower_m and info.ems_upper_m:
            lx, ly, _ = info.ems_lower_m
            ux, uy, _ = info.ems_upper_m
            parts.append(
                f'<rect x="{X(lx):.1f}" y="{Y(uy):.1f}" width="{(ux - lx) * scale:.1f}" '
                f'height="{(uy - ly) * scale:.1f}" fill="none" stroke="{EMS}" stroke-width="2" '
                f'stroke-dasharray="6,3"><title>EMS of the chosen candidate</title></rect>'
            )
        parts.append(
            f'<rect x="{X(pose.x):.1f}" y="{Y(pose.y + dy):.1f}" width="{dx * scale:.1f}" '
            f'height="{dy * scale:.1f}" fill="{CHOSEN}" fill-opacity="0.35" stroke="{CHOSEN}" '
            f'stroke-width="2.5"><title>chosen {html.escape(cand.candidate_id)} '
            f"z={pose.z:.3f}</title></rect>"
        )
    lx = width - 270
    ly = margin + 50
    lines = [
        (f"box {box.box_id}  {box.size.x:.2f}x{box.size.y:.2f}x{box.size.z:.2f} m  {box.weight_kg:.1f} kg", "#222"),
        (f"{candidate_set.summary()}", "#222"),
        (f"pallet {len(state.pallet.boxes)} boxes, max top {max_top:.2f} / {p.z:.2f} m", "#222"),
        ("", "#222"),
        ("placed box: darker = higher, label = top (m)", "#555"),
        ("valid candidate", VALID),
        ("masked candidate (dashed)", MASKED),
        ("chosen candidate", CHOSEN),
        ("EMS of chosen", EMS),
        ("", "#222"),
        ("mask reasons (candidates, all failing checks):", "#222"),
    ] + [(f"  {k}: {v}", MASKED) for k, v in reasons.most_common(8)]
    for i, (text, color) in enumerate(lines):
        parts.append(
            f'<text x="{lx}" y="{ly + 18 * i}" font-size="12" fill="{color}">{html.escape(text)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
