# viz_misclassifications.py
# -*- coding: utf-8 -*-
"""
Visualizações 2D/3D das amostras mal-classificadas (misclassifications)
a partir dos landmarks .npy (sem OpenCV).

Entrada:
  - report_figs_v3/misclassifications.csv  (file,true,pred,confidence)
  - libras_data/<classe>/*.npy             (landmarks originais)
  - results_v3/norm_stats.json             (para fixar T usado na comparação)

Saída (em --outdir):
  - miscls_2d/<true>_as_<pred>_N.png           (amostra 2D)
  - miscls_3d/<true>_as_<pred>_N.png           (amostra 3D)
  - overlay_2d/<true>_as_<pred>_N.png          (amostra vs centroides)
  - centroids/<classe>_centroid_2d.png/.npy    (prototipos por classe)
"""

import os, csv, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- conexões MediaPipe (21 pts) ----------
MP_EDGES_ONE = [
    (0,1),(1,2),(2,3),(3,4),          # polegar
    (0,5),(5,6),(6,7),(7,8),          # indicador
    (0,9),(9,10),(10,11),(11,12),     # médio
    (0,13),(13,14),(14,15),(15,16),   # anelar
    (0,17),(17,18),(18,19),(19,20)    # mínimo
]

def edges_for_F(F):
    if F == 63:
        return MP_EDGES_ONE
    if F == 126:
        return MP_EDGES_ONE + [(a+21, b+21) for (a,b) in MP_EDGES_ONE]
    return []

# ---------- utilidades ----------
def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    T = int(d["T"])
    return T

def pad_or_crop_to_T(x, T):
    t = x.shape[0]
    if t == T: return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    pad = np.zeros((T - t, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def frame_center(T): return T // 2

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- leitura do CSV ----------
def read_miscls(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({
                "file": r["file"],
                "true": r["true"],
                "pred": r["pred"],
                "confidence": float(r["confidence"])
            })
    return rows

# ---------- desenho 2D/3D ----------
def draw_2d(points_xyz, edges, title, outpng):
    xs, ys = points_xyz[:,0], points_xyz[:,1]
    plt.figure(figsize=(5,5))
    if edges:
        for a,b in edges:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], linewidth=2, alpha=0.8)
    plt.scatter(xs, ys, s=20)
    plt.gca().invert_yaxis()
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.savefig(outpng); plt.close()

def draw_3d(points_xyz, edges, title, outpng):
    xs, ys, zs = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    if edges:
        for a,b in edges:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], linewidth=1.5, alpha=0.9)
    ax.scatter(xs, ys, zs, s=20)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout(); plt.savefig(outpng); plt.close()

def overlay_2d(sample_xyz, true_cent_xyz, pred_cent_xyz, edges, title, outpng):
    plt.figure(figsize=(6,5))
    # conexões da amostra
    if edges:
        for a,b in edges:
            plt.plot([sample_xyz[a,0], sample_xyz[b,0]],
                     [sample_xyz[a,1], sample_xyz[b,1]], alpha=0.6)
    # pontos
    plt.scatter(sample_xyz[:,0], sample_xyz[:,1], s=24, label="Amostra", alpha=0.9)
    # centroides (linhas finas)
    if true_cent_xyz is not None:
        plt.scatter(true_cent_xyz[:,0], true_cent_xyz[:,1], s=28, marker="x", label="Centro (true)")
    if pred_cent_xyz is not None:
        plt.scatter(pred_cent_xyz[:,0], pred_cent_xyz[:,1], s=28, marker="^", label="Centro (pred)")
    plt.gca().invert_yaxis()
    plt.title(title); plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.tight_layout(); plt.savefig(outpng); plt.close()

# ---------- centroides por classe ----------
def compute_centroids(data_dir, classes, T, limit_per_class=None):
    """
    Retorna dict: class_name -> (F, centroid_xyz (P,3), usado_F)
      - F pode ser 63 ou 126 (landmarks absolutos salvos)
      - centroid_xyz é o frame central médio (sobre amostras), em 3D
    """
    centroids = {}
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir): continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".npy")]
        if not files: continue
        if limit_per_class:
            files = files[:limit_per_class]

        acc = []
        F_ref = None
        for fp in files:
            arr = np.load(fp).astype(np.float32)
            if arr.ndim != 2: continue
            arr = pad_or_crop_to_T(arr, T)   # (T,F_raw ABS)
            if F_ref is None: F_ref = arr.shape[1]
            if arr.shape[1] != F_ref: continue  # pula se não bater
            idx = frame_center(T)
            pts = arr[idx].reshape(-1, 3)    # (P,3)
            acc.append(pts)
        if acc:
            centroids[cls] = (F_ref, np.mean(np.stack(acc, axis=0), axis=0))
    return centroids

# ---------- principal ----------
def main():
    ap = argparse.ArgumentParser(description="Plot 2D/3D das misclassifications usando landmarks salvos.")
    ap.add_argument("--csv",     default=os.path.join("report_figs_v3","misclassifications.csv"))
    ap.add_argument("--data_dir",default="libras_data")
    ap.add_argument("--norm",    default=os.path.join("results_v3","norm_stats.json"))
    ap.add_argument("--outdir",  default=os.path.join("report_figs_v3","miscls_viz"))
    ap.add_argument("--limit",   type=int, default=None, help="Limitar nº de exemplos plotados (global)")
    ap.add_argument("--centroid_max", type=int, default=None, help="No. máximo de arquivos para calcular cada centróide")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    out2d   = os.path.join(args.outdir, "miscls_2d");   ensure_dir(out2d)
    out3d   = os.path.join(args.outdir, "miscls_3d");   ensure_dir(out3d)
    outov   = os.path.join(args.outdir, "overlay_2d");  ensure_dir(outov)
    outcent = os.path.join(args.outdir, "centroids");   ensure_dir(outcent)

    T = load_norm_stats(args.norm)
    rows = read_miscls(args.csv)
    if args.limit: rows = rows[:args.limit]

    # classes presentes no CSV (para centroides)
    classes = sorted(set([r["true"] for r in rows] + [r["pred"] for r in rows]))
    centroids = compute_centroids(args.data_dir, classes, T, limit_per_class=args.centroid_max)

    # Salva imagens dos centroides (2D)
    for cls, (F_c, Cxyz) in centroids.items():
        edges = edges_for_F(F_c)
        draw_2d(Cxyz, edges, f"Centroid 2D - {cls}", os.path.join(outcent, f"{cls}_centroid_2d.png"))
        # também salva em .npy se quiser usar depois
        np.save(os.path.join(outcent, f"{cls}_centroid.npy"), Cxyz)

    # varredura das misclassifications
    for k, r in enumerate(rows):
        fp   = r["file"]
        tcls = r["true"]
        pcls = r["pred"]
        conf = r["confidence"]

        if not os.path.isfile(fp):
            print(f"[AVISO] Não encontrei {fp}, pulando."); continue

        arr = np.load(fp).astype(np.float32)       # (T0,F_raw ABS)
        if arr.ndim != 2: 
            print(f"[AVISO] {fp} com shape inesperado: {arr.shape}"); continue
        arr = pad_or_crop_to_T(arr, T)
        F = arr.shape[1]
        P = F // 3
        edges = edges_for_F(F)

        idx = frame_center(T)
        pts = arr[idx].reshape(P, 3)               # (P,3)

        tag = f"{tcls}_as_{pcls}_{k:04d}_conf{conf:.3f}"
        # 2D da amostra
        draw_2d(pts, edges, f"{tcls} → {pcls} (conf={conf:.2f})", os.path.join(out2d, f"{tag}.png"))
        # 3D da amostra
        draw_3d(pts, edges, f"{tcls} → {pcls} (conf={conf:.2f})", os.path.join(out3d, f"{tag}.png"))

        # overlay 2D com centroides
        true_cent = centroids.get(tcls, (None, None))[1]
        pred_cent = centroids.get(pcls, (None, None))[1]
        overlay_2d(pts, true_cent, pred_cent, edges,
                   f"{tcls} (amostra) vs centros [true/pred]",
                   os.path.join(outov, f"{tag}.png"))

    print(f"\n✅ Imagens salvas em: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
