# hpa_star_lunar.py
import numpy as np
import rasterio
from scipy.ndimage import sobel
import heapq
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm   # NEW: progress bar

# =========================
# Utility: slope & mask
# =========================
def compute_slope_deg(elevation: np.ndarray, pixel_size: float) -> np.ndarray:
    dzdx = sobel(elevation, axis=1) / pixel_size
    dzdy = sobel(elevation, axis=0) / pixel_size
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    return np.degrees(slope_rad)

def traversable_from_slope(slope_deg: np.ndarray, slope_tolerance_deg: float) -> np.ndarray:
    return slope_deg <= slope_tolerance_deg

def crop(arr, center_xy, size):
    cy, cx = center_xy
    half = size // 2
    y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
    return arr[y0:y1, x0:x1], (y0, x0)

# =========================
# Low-level A*
# =========================
def astar_grid(trav: np.ndarray, start, goal, cost_weight_map=None):
    H, W = trav.shape
    def inb(y, x): return 0 <= y < H and 0 <= x < W
    def h(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])

    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    openq = []
    g = {start: 0.0}
    parent = {}
    f0 = h(start, goal)
    heapq.heappush(openq, (f0, 0.0, start))

    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            path = deque([cur])
            while cur in parent:
                cur = parent[cur]
                path.appendleft(cur)
            return list(path)
        cy, cx = cur
        for dy, dx in neigh:
            ny, nx = cy+dy, cx+dx
            if not inb(ny, nx) or not trav[ny, nx]:
                continue
            step = np.hypot(dy, dx)
            extra = 0.0 if cost_weight_map is None else cost_weight_map[ny, nx]
            cand = gc + step + extra
            if (ny, nx) not in g or cand < g[(ny, nx)]:
                g[(ny, nx)] = cand
                parent[(ny, nx)] = cur
                f = cand + h((ny, nx), goal)
                heapq.heappush(openq, (f, cand, (ny, nx)))
    return []

# =========================
# HPA* Implementation
# =========================
class HPAStar:
    def __init__(self, traversable: np.ndarray, cluster_size_px: int, cell_cost=None):
        self.trav = traversable
        self.H, self.W = traversable.shape
        self.C = cluster_size_px
        self.cell_cost = cell_cost
        self.clusters = {}
        self.entrances = defaultdict(list)
        self.portal_nodes = []
        self.portal_global = []
        self.portal_index = {}
        self.intra_cost = {}
        self.adj_edges = defaultdict(list)
        self._partition()
        self._find_entrances()
        self._precompute_intra_edges()
        self._build_inter_edges()

    def _partition(self):
        C = self.C
        for y0 in range(0, self.H, C):
            for x0 in range(0, self.W, C):
                y1 = min(self.H, y0 + C)
                x1 = min(self.W, x0 + C)
                key = (y0 // C, x0 // C)
                self.clusters[key] = (y0, y1, x0, x1)

    def _find_entrances(self):
        for key, (y0, y1, x0, x1) in self.clusters.items():
            sub = self.trav[y0:y1, x0:x1]
            Hc, Wc = sub.shape
            locs = []
            if Hc == 0 or Wc == 0: continue
            for x in range(Wc):
                if sub[0, x]: locs.append((0, x))
                if sub[Hc-1, x]: locs.append((Hc-1, x))
            for y in range(Hc):
                if sub[y, 0]: locs.append((y, 0))
                if sub[y, Wc-1]: locs.append((y, Wc-1))
            locs = sorted(set(locs))
            self.entrances[key] = locs

        idx = 0
        for key, locs in self.entrances.items():
            y0, y1, x0, x1 = self.clusters[key]
            for ly, lx in locs:
                gy, gx = y0 + ly, x0 + lx
                node = (key, (ly, lx))
                self.portal_nodes.append(node)
                self.portal_global.append((gy, gx))
                self.portal_index[node] = idx
                idx += 1

    def _precompute_intra_edges(self, verbose=True):
        cluster_iter = self.entrances.items()
        if verbose:
            cluster_iter = tqdm(self.entrances.items(), desc="[HPA*] Precomputing clusters")

        for key, locs in cluster_iter:
            y0, y1, x0, x1 = self.clusters[key]
            sub_trav = self.trav[y0:y1, x0:x1]
            sub_cost = None if self.cell_cost is None else self.cell_cost[y0:y1, x0:x1]
            for i in range(len(locs)):
                for j in range(i+1, len(locs)):
                    a = locs[i]; b = locs[j]
                    path = astar_grid(sub_trav, a, b, cost_weight_map=sub_cost)
                    if not path: continue
                    cost = sum(
                        np.hypot(v[0]-u[0], v[1]-u[1]) +
                        (0.0 if sub_cost is None else sub_cost[v[0], v[1]])
                        for u, v in zip(path, path[1:])
                    )
                    ni = self.portal_index[(key, a)]
                    nj = self.portal_index[(key, b)]
                    self.intra_cost[(ni, nj)] = (cost, path)
                    self.intra_cost[(nj, ni)] = (cost, list(reversed(path)))

    def _build_inter_edges(self):
        for (i, j), (c, p) in self.intra_cost.items():
            self.adj_edges[i].append((j, c, p))
        g_to_idx = {g: i for i, g in enumerate(self.portal_global)}
        H, W = self.H, self.W
        def inb(y, x): return 0 <= y < H and 0 <= x < W
        for i, (key_i, (ly, lx)) in enumerate(self.portal_nodes):
            y0, y1, x0, x1 = self.clusters[key_i]
            gy, gx = y0 + ly, x0 + lx
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = gy+dy, gx+dx
                if not inb(ny, nx) or not self.trav[ny, nx]: continue
                j = g_to_idx.get((ny, nx), None)
                if j is not None:
                    step_cost = np.hypot(dy, dx)
                    extra = 0.0 if self.cell_cost is None else self.cell_cost[ny, nx]
                    self.adj_edges[i].append((j, step_cost + extra, [(ly, lx), (ly+dy, lx+dx)]))

    def _abstract_astar(self, start_nodes, goal_nodes):
        def h(i, goal_set):
            yi, xi = self.portal_global[i]
            return min(np.hypot(yi - yg, xi - xg) for (yg, xg) in goal_set)
        goal_globs = [self.portal_global[i] for i in goal_nodes]
        goal_set = set(goal_nodes)
        openq = []
        g = {}
        parent = {}
        for i, gc0 in start_nodes:
            g[i] = gc0
            heapq.heappush(openq, (gc0 + h(i, goal_globs), gc0, i))
            parent[i] = None
        reached = None
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur in goal_set:
                reached = cur; break
            for (nbr, w, _) in self.adj_edges[cur]:
                cand = gc + w
                if nbr not in g or cand < g[nbr]:
                    g[nbr] = cand
                    parent[nbr] = cur
                    heapq.heappush(openq, (cand + h(nbr, goal_globs), cand, nbr))
        if reached is None: return []
        seq = deque([reached])
        p = parent[reached]
        while p is not None:
            seq.appendleft(p); p = parent[p]
        return list(seq)

    def hpa_search(self, start_px, goal_px):
        """
        Returns full-resolution pixel path from start_px to goal_px using HPA* refinement.
        """
        # Helper: convert pixel → cluster + local
        def to_cluster(y, x):
            cy, cx = y // self.C, x // self.C
            y0, y1, x0, x1 = self.clusters[(cy, cx)]
            return (cy, cx), (y - y0, x - x0)

        if not self.trav[start_px] or not self.trav[goal_px]:
            return []

        # --- Cluster and entrances for start and goal
        skey, sloc = to_cluster(*start_px)
        gkey, gloc = to_cluster(*goal_px)
        s_portals = self.entrances[skey]
        g_portals = self.entrances[gkey]

        # Fallback: if cluster has no entrances, just do plain A*
        if len(s_portals) == 0 or len(g_portals) == 0:
            return astar_grid(self.trav, start_px, goal_px, cost_weight_map=self.cell_cost)

        # --- Connect start to its portals (local A*)
        y0s, y1s, x0s, x1s = self.clusters[skey]
        subS = self.trav[y0s:y1s, x0s:x1s]
        subSc = None if self.cell_cost is None else self.cell_cost[y0s:y1s, x0s:x1s]

        start_nodes = []
        s_local_paths = {}
        for sp in s_portals:
            path = astar_grid(subS, sloc, sp, cost_weight_map=subSc)
            if not path:
                continue
            cost = sum(
                np.hypot(v[0]-u[0], v[1]-u[1]) +
                (0.0 if subSc is None else subSc[v[0], v[1]])
                for u, v in zip(path, path[1:])
            )
            i = self.portal_index[(skey, sp)]
            start_nodes.append((i, cost))
            s_local_paths[i] = path

        if not start_nodes:
            return astar_grid(self.trav, start_px, goal_px, cost_weight_map=self.cell_cost)

        # --- Connect goal to its portals (local A*)
        y0g, y1g, x0g, x1g = self.clusters[gkey]
        subG = self.trav[y0g:y1g, x0g:x1g]
        subGc = None if self.cell_cost is None else self.cell_cost[y0g:y1g, x0g:x1g]

        goal_nodes = []
        g_local_paths_rev = {}
        for gp in g_portals:
            path = astar_grid(subG, gp, gloc, cost_weight_map=subGc)
            if not path:
                continue
            cost = sum(
                np.hypot(v[0]-u[0], v[1]-u[1]) +
                (0.0 if subGc is None else subGc[v[0], v[1]])
                for u, v in zip(path, path[1:])
            )
            j = self.portal_index[(gkey, gp)]
            goal_nodes.append(j)
            g_local_paths_rev[j] = path

        if not goal_nodes:
            return astar_grid(self.trav, start_px, goal_px, cost_weight_map=self.cell_cost)

        # --- High-level search over abstract graph
        node_path = self._abstract_astar(start_nodes, goal_nodes)
        if not node_path:
            return astar_grid(self.trav, start_px, goal_px, cost_weight_map=self.cell_cost)

        # --- Stitch full path ---
        full = []

        # 1) Start → first portal
        first_portal = node_path[0]
        if first_portal not in s_local_paths:
            sp_key, sp_local = self.portal_nodes[first_portal]
            pathS = astar_grid(subS, sloc, sp_local, cost_weight_map=subSc)
        else:
            pathS = s_local_paths[first_portal]
        for (ly, lx) in pathS:
            full.append((y0s + ly, x0s + lx))

        # 2) Between portals along node_path
        for a, b in zip(node_path, node_path[1:]):
            (keyA, la), (keyB, lb) = self.portal_nodes[a], self.portal_nodes[b]
            if keyA == keyB:
                if (a, b) in self.intra_cost:
                    _, p = self.intra_cost[(a, b)]
                    y0, y1, x0, x1 = self.clusters[keyA]
                    for (ly, lx) in p[1:]:
                        full.append((y0 + ly, x0 + lx))
                else:
                    y0, y1, x0, x1 = self.clusters[keyA]
                    sub = self.trav[y0:y1, x0:x1]
                    subc = None if self.cell_cost is None else self.cell_cost[y0:y1, x0:x1]
                    p = astar_grid(sub, la, lb, cost_weight_map=subc)
                    if not p:
                        return []
                    for (ly, lx) in p[1:]:
                        full.append((y0 + ly, x0 + lx))
            else:
                yb, xb = self.portal_global[b]
                if not full or full[-1] != (yb, xb):
                    full.append((yb, xb))

        # 3) Last portal → goal
        last_portal = node_path[-1]
        pathG = g_local_paths_rev[last_portal]
        for (ly, lx) in pathG[1:]:
            full.append((y0g + ly, x0g + lx))

        return full


# =========================
# Plotting
# =========================
def plot2d(elev, trav, path=None, start=None, goal=None, title="HPA* Path"):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(elev, cmap='gray', origin='upper')
    mask = np.ma.masked_where(trav, np.zeros_like(trav))
    ax.imshow(mask, cmap='Reds', alpha=0.15)
    if start: ax.plot(start[1], start[0], 'go', label='Start')
    if goal: ax.plot(goal[1], goal[0], 'ro', label='Goal')
    if path:
        ys, xs = zip(*path)
        ax.plot(xs, ys, 'c-', lw=2, label='HPA* Path')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout(); plt.show()

def plot3d(elev, path=None, start=None, goal=None, title="3D Terrain with Path"):
    Y, X = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(111, projection='3d')

    # Terrain surface
    ax.plot_surface(X, Y, elev, cmap='terrain',
                    linewidth=0, antialiased=False,
                    alpha=0.9, zorder=0)

    # Path (slightly above terrain so it doesn’t sink in)
    if path and len(path) > 1:
        ys, xs = zip(*path)
        zs = elev[ys, xs] + 50  # hover above surface
        ax.plot(xs, ys, zs, 'r-', lw=4, label="Path")

    # Force markers ABOVE terrain
    z_offset = np.nanmax(elev) + 1000  # way above surface

    if start:
        sy, sx = start
        ax.scatter3D([sx], [sy], [z_offset],
                     c='lime', s=5000, marker='o',
                     edgecolors='k', linewidths=2,
                     label="Start", depthshade=False)

    if goal:
        gy, gx = goal
        ax.scatter3D([gx], [gy], [z_offset],
                     c='blue', s=5000, marker='^',
                     edgecolors='k', linewidths=2,
                     label="Goal", depthshade=False)

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_zlabel('Elevation')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()








# =========================
# Runner
# =========================
def load_dem(path):
    with rasterio.open(path) as src:
        return src.read(1)

def prepare_grid(elev, pixel_size, slope_tol_deg, crop_center=None, crop_size=None, slope_cost_alpha=0.0):
    slope = compute_slope_deg(elev, pixel_size)
    trav = traversable_from_slope(slope, slope_tol_deg)
    if slope_cost_alpha > 0:
        cost = slope_cost_alpha * (slope / max(1e-6, slope_tol_deg))
        cost[~trav] = np.inf
    else:
        cost = None
    origin = (0, 0)
    if crop_center is not None and crop_size is not None:
        elev, origin = crop(elev, crop_center, crop_size)
        trav, _ = crop(trav, crop_center, crop_size)
        if cost is not None:
            cost, _ = crop(cost, crop_center, crop_size)
    return elev, trav, origin, cost

def run_hpa(elev, trav, start_global, goal_global, cluster_size_px=200, cell_cost=None):
    hpa = HPAStar(trav, cluster_size_px=cluster_size_px, cell_cost=cell_cost)
    return hpa.hpa_search(start_global, goal_global)

# =========================
# DEMO
# =========================
if __name__ == "__main__":
    SOURCES = [("DEM_5m", "superResFIL1.tif", 5.0)]
    SLOPE_TOL = 30
    CROP_CENTER = (250, 250)
    CROP_SIZE = 100
    CLUSTER_SIZE_PX = 100
    SLOPE_COST_ALPHA = 0.15
    start_local = (50, 50)
    goal_local  = (90, 90)   # keep inside CROP_SIZE to avoid out-of-bounds

    for label, path, pxsize in SOURCES:
        print(f"\n=== {label} | pixel_size={pxsize} m/px ===")
        elev_full = load_dem(path)
        elev, trav, origin, cost = prepare_grid(
            elev_full, pxsize, SLOPE_TOL,
            crop_center=CROP_CENTER, crop_size=CROP_SIZE,
            slope_cost_alpha=SLOPE_COST_ALPHA
        )
        start, goal = start_local, goal_local
        # 🔎 DEBUG HERE
        print("Cropped DEM shape:", elev.shape)
        print("Start:", start, "Goal:", goal)
        sy, sx = start
        gy, gx = goal
        print("Start elevation:", elev[sy, sx])
        print("Goal elevation:", elev[gy, gx])

        path = run_hpa(elev, trav, start, goal,
                       cluster_size_px=CLUSTER_SIZE_PX,
                       cell_cost=cost)

        if not path:
            print("⚠️ No HPA* path found. Still plotting terrain without path.")
            path = None  # ensure plotting works

        # Always plot
        plot2d(elev, trav, path=path, start=start, goal=goal, title=f"{label} (HPA*)")
        plot3d(elev, path=path, title=f"{label} (HPA*) 3D")

