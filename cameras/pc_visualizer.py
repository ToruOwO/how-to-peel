#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import socket
import time
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from werkzeug.serving import make_server
import plotly.graph_objs as go
import plotly.io as pio
import open3d as o3d  # optional; used by preview_in_open3d


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return port


class Visualizer:
    """
    Interactive Plotly pointcloud visualizer that can be started/stopped cleanly.
    """

    def __init__(self, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port if port is not None else find_free_port()

        self.app = Flask(__name__)
        self.selected_points = []
        self._serialized_points = dict(x=[], y=[], z=[], colors=[])

        # Routes
        @self.app.route("/", methods=["GET"])
        def index():
            return render_template_string("""
            <style>
            #selected-points {
                max-height: 200px;
                overflow-y: auto;
                font-size: 12px;
                padding-left: 10px;
            }
            </style>

            <div style="display: flex;">
                <div style="flex: 3;">
                    <div id="plotly-div"></div>
                </div>
                <div style="flex: 1; padding-left:20px;">
                    <h4>Selected Points</h4>
                    <ul id="selected-points"></ul>
                    <button onclick="submitPoints()">Submit</button>
                </div>
            </div>

            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
            let selectedXYZ = [];

            function render(points){
                const trace = {
                    x: points.x, y: points.y, z: points.z,
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {size: 4, color: points.colors},
                    name: 'Original Points'
                };
                const layout = {
                    width: window.innerWidth * 0.90,
                    height: window.innerHeight * 0.95,
                    margin: {l:0, r:0, b:0, t:0},
                    scene: {aspectmode: 'data'}
                };
                Plotly.newPlot('plotly-div', [trace], layout);

                document.getElementById('plotly-div').on('plotly_click', function(event){
                    const pt = event.points[0];
                    const xval = pt.x.toFixed(3);
                    const yval = pt.y.toFixed(3);
                    const zval = pt.z.toFixed(3);
                    selectedXYZ.push([xval, yval, zval]);
                    navigator.clipboard.writeText(xval + ',' + yval + ',' + zval).catch(()=>{});
                    updateList();
                });
            }

            function updateList(){
                const ul = document.getElementById('selected-points');
                ul.innerHTML = '';
                selectedXYZ.forEach(function(pt, idx){
                    const li = document.createElement('li');
                    li.textContent = `(${pt[0]}, ${pt[1]}, ${pt[2]})`;
                    li.style.cursor = 'pointer';
                    li.onclick = function(){
                        selectedXYZ.splice(idx,1);
                        updateList();
                    };
                    ul.appendChild(li);
                });
            }

            function submitPoints(){
                fetch("/submit", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({waypoints: selectedXYZ})
                }).then(res => res.text())
                .then(msg => {
                    selectedXYZ = [];
                    updateList();
                })
                .catch(err => alert("Error:"+err));
            }

            fetch("/data").then(r => r.json()).then(render);
            </script>
            """)

        @self.app.route("/data", methods=["GET"])
        def data():
            return jsonify(self._serialized_points)

        @self.app.route("/submit", methods=["POST"])
        def submit():
            data = request.get_json()
            self.selected_points = data.get("waypoints", [])
            print("Selected Points:", self.selected_points)
            return "OK"

        # Stoppable server
        self._server = make_server(self.host, self.port, self.app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -------- Public API --------
    def visualize_pointcloud(self, pointcloud: np.ndarray):
        """Start the HTTP server (non-blocking). Call shutdown() to stop it."""
        self._prepare_pointcloud(pointcloud)
        self._thread.start()
        print(f"[Visualizer] Serving at http://{self.host}:{self.port}")

    def shutdown(self, timeout: float = 2.0):
        """Stop the server and join the serving thread."""
        if self._server:
            try:
                self._server.shutdown()
            except Exception as e:
                print("shutdown() error:", e)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def visualize_pointcloud_and_save(self, pointcloud: np.ndarray, save_path: str):
        """Save a static image (PNG, etc.) of the pointcloud."""
        trace = self._generate_trace(pointcloud, size=4, opacity=0.9)
        fig = go.Figure(
            data=[trace],
            layout=go.Layout(
                margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode="data")
            ),
        )
        fig.write_image(save_path, width=1024, height=768)

    def save_visualization_to_file(self, pointcloud: np.ndarray, file_path: str):
        """Save an interactive HTML file."""
        trace = self._generate_trace(pointcloud)
        fig_html = pio.to_html(go.Figure(data=[trace]), full_html=True)
        with open(file_path, "w") as f:
            f.write(fig_html)

    @staticmethod
    def preview_in_open3d(point_clouds: list[np.ndarray]):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)
        for pcd in point_clouds:
            geometry.points = o3d.utility.Vector3dVector(pcd[:, :3])
            colors = (
                np.zeros_like(pcd[:, :3]) if pcd.shape[1] < 6 else pcd[:, 3:6] / 255.0
            )
            geometry.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

    # -------- Helpers --------
    def _prepare_pointcloud(self, pcd: np.ndarray):
        x = pcd[:, 0]
        y = pcd[:, 1]
        z = pcd[:, 2]

        if pcd.shape[1] == 3:
            xyz = pcd[:, :3]
            min_coords = xyz.min(axis=0)
            max_coords = xyz.max(axis=0)
            normalized = (xyz - min_coords) / (max_coords - min_coords + 1e-8)
            colors = [
                "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
                for r, g, b in normalized
            ]
        elif pcd.shape[1] >= 6:
            rgb = pcd[:, 3:6]
            colors = ["rgb({},{},{})".format(int(r), int(g), int(b)) for r, g, b in rgb]
        else:
            raise ValueError(f"Pointcloud shape {pcd.shape} is not supported")

        self._serialized_points = dict(
            x=x.tolist(), y=y.tolist(), z=z.tolist(), colors=colors
        )

    def _generate_trace(self, pointcloud: np.ndarray, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            xyz = pointcloud[:, :3]
            min_coords = xyz.min(axis=0)
            max_coords = xyz.max(axis=0)
            normalized = (xyz - min_coords) / (max_coords - min_coords + 1e-8)
            colors = [
                "rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255))
                for r, g, b in normalized
            ]
        elif pointcloud.shape[1] >= 6:
            rgb = pointcloud[:, 3:6]
            colors = ["rgb({},{},{})".format(int(r), int(g), int(b)) for r, g, b in rgb]
        else:
            raise ValueError(f"Pointcloud shape {pointcloud.shape} is not supported")

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            marker=dict(size=size, opacity=opacity, color=colors),
            name="Pointcloud",
        )


# ---------------- Example loop usage ----------------
if __name__ == "__main__":
    for i in range(3):
        print(f"Iteration {i}")
        pts = np.random.randn(5000, 3).astype(np.float32)

        vis = Visualizer()
        vis.visualize_pointcloud(pts)

        time.sleep(3)  # simulate work / time for interaction

        vis.shutdown()
        print("Shutdown clean.\n")
