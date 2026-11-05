"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

from gymnasium import error
import numpy as np
import six

from rware.warehouse import Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)

_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 30
        self.icon_size = 20

        # Original RWARE map area.
        self.map_width = 1 + self.cols * (self.grid_size + 1)
        self.map_height = 2 + self.rows * (self.grid_size + 1)

        # Extra right-side debug panel for per-agent local observations.
        # This does not affect the environment state or observations.
        self.side_panel_width = 260
        self.side_panel_padding = 10
        self.mini_grid_size = 13

        self.width = self.map_width + self.side_panel_width
        self.height = self.map_height
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display, resizable=True, visible=False
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def _ensure_window_size_for_side_panel(self, env):
        """Resize horizontally so agent local views wrap into extra columns.

        The main warehouse map keeps its original height. If the right panel
        cannot fit all local views vertically, the panel becomes wider and the
        remaining local views are drawn in additional columns.
        """
        if not hasattr(env, "agents") or not hasattr(env, "sensor_range"):
            required_width = int(self.map_width + self.side_panel_width)
            required_height = int(self.map_height)
        else:
            n_agents = len(env.agents)
            r = int(env.sensor_range)
            metrics = self._side_panel_layout_metrics(r)

            available_h = max(
                1,
                self.map_height
                - 2 * self.side_panel_padding
                - metrics["title_h"],
            )
            rows_per_col = max(1, int(available_h // metrics["agent_block_h"]))
            n_panel_cols = max(1, int(math.ceil(n_agents / rows_per_col)))
            required_side_panel_width = (
                2 * self.side_panel_padding
                + n_panel_cols * metrics["panel_col_width"]
            )

            # Keep the map height fixed; expand only horizontally.
            required_width = int(self.map_width + required_side_panel_width)
            required_height = int(self.map_height)

        if self.window.width != required_width or self.window.height != required_height:
            self.window.set_size(required_width, required_height)
            self.width = required_width
            self.height = required_height

    def render(self, env, return_rgb_array=False):
        if not return_rgb_array and not self.window.visible:
            self.window.set_visible(True)
        self._ensure_window_size_for_side_panel(env)

        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_agent_views_on_map(env)
        self._draw_goals(env)
        self._draw_shelfs(env)
        self._draw_training_comm_edges(env)
        self._draw_agents(env)
        self._draw_side_panel(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen
    
    _TEAM_COLORS = [
        (220, 80, 80),    # team 0: red
        (80, 120, 220),   # team 1: blue
        (80, 180, 100),   # team 2: green
        (220, 170, 60),   # team 3: yellow
        (170, 90, 200),   # team 4: purple
    ]


    def _team_color(self, team_id):
        return self._TEAM_COLORS[int(team_id) % len(self._TEAM_COLORS)]


    def _light_color(self, color, alpha=0.45):
        return tuple(
            int(c + (255 - c) * alpha)
            for c in color
        )
    def _dark_color(self, color, alpha=0.35):
        return tuple(
            int(c * (1.0 - alpha))
            for c in color
        )

    def _side_panel_layout_metrics(self, r):
        view_size = 2 * int(r) + 1
        title_h = 24
        agent_label_h = 14
        view_label_h = 12
        view_gap = 8
        row_gap = 28
        # Fit both local views even when a single agent block would otherwise
        # exceed the fixed map height (e.g. tiny warehouse, sensor range 5).
        available_grid_h = (
            self.map_height - 2 * self.side_panel_padding - title_h
            - agent_label_h - 2 * view_label_h - view_gap - row_gap
        )
        cell = max(1, min(self.mini_grid_size, available_grid_h // (2 * view_size)))
        block_w = view_size * cell
        block_h = view_size * cell
        agent_block_h = (
            agent_label_h
            + view_label_h
            + block_h
            + view_gap
            + view_label_h
            + block_h
            + row_gap
        )

        return {
            "view_size": view_size,
            "cell": cell,
            "block_w": block_w,
            "block_h": block_h,
            "title_h": title_h,
            "agent_label_h": agent_label_h,
            "view_label_h": view_label_h,
            "view_gap": view_gap,
            "row_gap": row_gap,
            "agent_block_h": agent_block_h,
            "panel_col_width": max(125, block_w + 45),
        }

    def _rotated_local_indices(self, dx, dy, r, direction):
        """Map local offsets to display row/col using the CNN observation rotation."""
        row = dy + r
        col = dx + r
        view_size = 2 * r + 1

        if direction == Direction.DOWN:
            return view_size - 1 - row, view_size - 1 - col
        if direction == Direction.LEFT:
            return col, view_size - 1 - row
        if direction == Direction.RIGHT:
            return view_size - 1 - col, row
        return row, col

    def _cell_center(self, x, y):
        """Return screen-space center of a warehouse cell."""
        render_y = self.rows - int(y) - 1
        cx = (self.grid_size + 1) * int(x) + self.grid_size // 2 + 1
        cy = (self.grid_size + 1) * render_y + self.grid_size // 2 + 1
        return cx, cy

    def _add_filled_rect(self, batch, x0, y0, x1, y1, color):
        batch.add(
            4,
            gl.GL_QUADS,
            None,
            ("v2f", (x0, y0, x1, y0, x1, y1, x0, y1)),
            ("c3B", 4 * tuple(color)),
        )

    def _add_rect_outline(self, batch, x0, y0, x1, y1, color):
        color = tuple(color)
        batch.add(2, gl.GL_LINES, None, ("v2f", (x0, y0, x1, y0)), ("c3B", (*color, *color)))
        batch.add(2, gl.GL_LINES, None, ("v2f", (x1, y0, x1, y1)), ("c3B", (*color, *color)))
        batch.add(2, gl.GL_LINES, None, ("v2f", (x1, y1, x0, y1)), ("c3B", (*color, *color)))
        batch.add(2, gl.GL_LINES, None, ("v2f", (x0, y1, x0, y0)), ("c3B", (*color, *color)))

    def _draw_agent_views_on_map(self, env):
        """Draw each agent's current sensor-range box on the main map."""
        if not hasattr(env, "sensor_range"):
            return

        batch = pyglet.graphics.Batch()
        r = int(env.sensor_range)

        for agent in env.agents:
            if hasattr(env, "agent_team_ids"):
                color = self._light_color(self._team_color(env.agent_team_ids[agent.id - 1]), 0.25)
            else:
                color = (160, 160, 160)

            x0 = max(0, int(agent.x) - r)
            x1 = min(self.cols - 1, int(agent.x) + r)
            y0 = max(0, int(agent.y) - r)
            y1 = min(self.rows - 1, int(agent.y) + r)

            # Convert environment y-range to pyglet y-range.
            render_y_bottom = self.rows - y1 - 1
            render_y_top = self.rows - y0 - 1

            left = (self.grid_size + 1) * x0 + 1
            right = (self.grid_size + 1) * (x1 + 1)
            bottom = (self.grid_size + 1) * render_y_bottom + 1
            top = (self.grid_size + 1) * (render_y_top + 1)

            self._add_rect_outline(batch, left, bottom, right, top, color)

        batch.draw()

    def _draw_training_comm_edges(self, env):
        """Draw dynamic communication/network edges between agents on the main map."""
        edges = getattr(env, "training_comm_edges", [])
        eligible = set(edges)
        if not edges:
            return

        batch = pyglet.graphics.Batch()
        for i, j in edges:
            if i < 0 or j < 0 or i >= len(env.agents) or j >= len(env.agents):
                continue
            ai = env.agents[int(i)]
            aj = env.agents[int(j)]
            x1, y1 = self._cell_center(ai.x, ai.y)
            x2, y2 = self._cell_center(aj.x, aj.y)

            color = (0, 110, 90) if (i, j) in eligible else (165, 165, 165)
            if getattr(env, "training_comm_directed", False):
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length <= 0:
                    continue
                ux, uy = dx / length, dy / length
                shift = 3.0
                x1, y1 = x1 + ux*11 - uy*shift, y1 + uy*11 + ux*shift
                x2, y2 = x2 - ux*12 - uy*shift, y2 - uy*12 + ux*shift
                batch.add(3, gl.GL_TRIANGLES, None,
                          ("v2f", (x2, y2, x2-ux*8-uy*4, y2-uy*8+ux*4,
                                   x2-ux*8+uy*4, y2-uy*8-ux*4)),
                          ("c3B", (*color, *color, *color)))
            batch.add(
                2,
                gl.GL_LINES,
                None,
                ("v2f", (x1, y1, x2, y2)),
                ("c3B", (*color, *color)),
            )
        gl.glLineWidth(2.0)
        batch.draw()
        gl.glLineWidth(1.0)

    def _draw_side_panel(self, env):
        """Draw a right-side panel containing each agent's local field of view.

        Local views are arranged column-wise. When the panel height is not
        enough, remaining agents are drawn in a new column to the right instead
        of increasing the window height.
        """
        panel_origin_x = self.map_width + 1
        panel_x0 = self.map_width + self.side_panel_padding
        top = self.map_height - self.side_panel_padding

        batch = pyglet.graphics.Batch()
        # Panel background and border.
        self._add_filled_rect(batch, panel_origin_x, 0, self.width, self.map_height, (245, 245, 245))
        self._add_rect_outline(batch, panel_origin_x, 1, self.width - 1, self.map_height - 1, (80, 80, 80))
        batch.draw()

        title = pyglet.text.Label(
            ("COMM OFF | independent" if not getattr(env, "communication_enabled", True)
             else f"PGCT accepted links: {len(getattr(env, 'training_comm_edges', []))} | {getattr(env, 'communication_topology', 'physical')}"),
            font_name="Calibri",
            font_size=12,
            bold=True,
            x=panel_x0,
            y=top,
            anchor_x="left",
            anchor_y="top",
            color=(*_BLACK, 255),
        )
        title.draw()

        if not hasattr(env, "sensor_range"):
            return

        r = int(env.sensor_range)
        metrics = self._side_panel_layout_metrics(r)
        cell = metrics["cell"]
        block_h = metrics["block_h"]

        available_h = max(
            1,
            self.map_height
            - 2 * self.side_panel_padding
            - metrics["title_h"],
        )
        rows_per_col = max(1, int(available_h // metrics["agent_block_h"]))

        for idx, agent in enumerate(env.agents):
            col_idx = idx // rows_per_col
            row_idx = idx % rows_per_col

            col_x0 = panel_x0 + col_idx * metrics["panel_col_width"]
            y_cursor = (
                top
                - metrics["title_h"]
                - row_idx * metrics["agent_block_h"]
            )

            if hasattr(env, "agent_team_ids"):
                team_id = int(env.agent_team_ids[agent.id - 1])
                agent_color = self._team_color(team_id)
                title_text = f"A{agent.id} / true T{team_id}"
            else:
                agent_color = _AGENT_COLOR
                title_text = f"A{agent.id}"

            if hasattr(env, "inferred_team_ids"):
                pred = int(env.inferred_team_ids[agent.id - 1])
                conf = float(env.inferred_team_confidence[agent.id - 1]) if hasattr(env, "inferred_team_confidence") else 0.0
                if pred >= 0:
                    title_text += f" / pred T{pred} ({conf:.2f})"

            label = pyglet.text.Label(
                title_text,
                font_name="Calibri",
                font_size=10,
                bold=True,
                x=col_x0,
                y=y_cursor,
                anchor_x="left",
                anchor_y="top",
                color=(*agent_color, 255),
            )
            label.draw()

            world_label_y = y_cursor - metrics["agent_label_h"]
            world_label = pyglet.text.Label(
                "world",
                font_name="Calibri",
                font_size=8,
                x=col_x0,
                y=world_label_y,
                anchor_x="left",
                anchor_y="top",
                color=(*_BLACK, 255),
            )
            world_label.draw()

            grid_x0 = col_x0
            grid_y0 = world_label_y - metrics["view_label_h"] - block_h
            self._draw_agent_local_view(env, agent, grid_x0, grid_y0, cell, r)

            rotated_label_y = grid_y0 - metrics["view_gap"]
            rotated_label = pyglet.text.Label(
                "agent/CNN",
                font_name="Calibri",
                font_size=8,
                x=col_x0,
                y=rotated_label_y,
                anchor_x="left",
                anchor_y="top",
                color=(*_BLACK, 255),
            )
            rotated_label.draw()

            rotated_grid_y0 = (
                rotated_label_y
                - metrics["view_label_h"]
                - block_h
            )
            self._draw_agent_local_view(
                env,
                agent,
                grid_x0,
                rotated_grid_y0,
                cell,
                r,
                rotate_to_agent=True,
            )

    def _draw_agent_local_view(
        self,
        env,
        center_agent,
        x0,
        y0,
        cell,
        r,
        rotate_to_agent=False,
    ):
        """Draw one agent-centered local observation panel."""
        view_size = 2 * r + 1
        batch = pyglet.graphics.Batch()

        shelf_by_pos = {(int(shelf.x), int(shelf.y)): shelf for shelf in env.shelfs}
        agent_by_pos = {(int(agent.x), int(agent.y)): agent for agent in env.agents}
        goals_by_pos = {tuple(map(int, goal)) for goal in getattr(env, "goals", [])}
        request_ids = {shelf.id for shelf in getattr(env, "request_queue", [])}

        # Background cells.
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                wx = int(center_agent.x) + dx
                wy = int(center_agent.y) + dy
                if rotate_to_agent:
                    display_row, display_col = self._rotated_local_indices(
                        dx, dy, r, center_agent.dir
                    )
                else:
                    display_row = dy + r
                    display_col = dx + r
                # World view is north-up; rotated view puts the agent's front at the top.
                px0 = x0 + display_col * cell
                py0 = y0 + (view_size - 1 - display_row) * cell
                px1 = px0 + cell
                py1 = py0 + cell

                if wx < 0 or wx >= self.cols or wy < 0 or wy >= self.rows:
                    fill = (210, 210, 210)
                else:
                    fill = (255, 255, 255)
                    if (wx, wy) in goals_by_pos:
                        if hasattr(env, "goal_team_ids") and (wx, wy) in env.goal_team_ids:
                            fill = self._light_color(self._team_color(env.goal_team_ids[(wx, wy)]), 0.35)
                        else:
                            fill = (220, 220, 220)

                self._add_filled_rect(batch, px0, py0, px1, py1, fill)
                self._add_rect_outline(batch, px0, py0, px1, py1, (150, 150, 150))

                shelf = shelf_by_pos.get((wx, wy))
                if shelf is not None:
                    if hasattr(env, "shelf_team_ids") and shelf.id in env.shelf_team_ids:
                        color = self._team_color(env.shelf_team_ids[shelf.id])
                        if shelf.id in request_ids:
                            color = self._light_color(color, 0.35)
                    else:
                        color = _SHELF_REQ_COLOR if shelf.id in request_ids else _SHELF_COLOR
                    pad = 2
                    self._add_filled_rect(batch, px0 + pad, py0 + pad, px1 - pad, py1 - pad, color)

        batch.draw()

        # Draw agents after shelves so they stay visible.
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                wx = int(center_agent.x) + dx
                wy = int(center_agent.y) + dy
                agent = agent_by_pos.get((wx, wy))
                if agent is None:
                    continue

                if rotate_to_agent:
                    display_row, display_col = self._rotated_local_indices(
                        dx, dy, r, center_agent.dir
                    )
                else:
                    display_row = dy + r
                    display_col = dx + r
                cx = x0 + display_col * cell + cell / 2
                cy = y0 + (view_size - 1 - display_row) * cell + cell / 2
                radius = max(3, cell / 3)

                if hasattr(env, "agent_team_ids"):
                    color = self._team_color(env.agent_team_ids[agent.id - 1])
                else:
                    color = _AGENT_COLOR
                if agent.id == center_agent.id:
                    color = self._dark_color(color, 0.15)

                verts = []
                resolution = 10
                for k in range(resolution):
                    angle = 2 * math.pi * k / resolution
                    verts += [radius * math.cos(angle) + cx, radius * math.sin(angle) + cy]
                circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
                glColor3ub(*color)
                circle.draw(GL_POLYGON)

        # Center mark.
        center_label = pyglet.text.Label(
            str(center_agent.id),
            font_name="Calibri",
            font_size=8,
            bold=True,
            x=x0 + r * cell + cell / 2,
            y=y0 + r * cell + cell / 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_WHITE, 255),
        )
        center_label.draw()


    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # HORIZONTAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # VERTICAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            
            if hasattr(env, "shelf_team_ids") and shelf.id in env.shelf_team_ids:
                team_id = env.shelf_team_ids[shelf.id]
                base_color = self._team_color(team_id)

                # requested shelf는 더 밝게 표시
                shelf_color = self._light_color(base_color) if shelf in env.request_queue else base_color
            else:
                shelf_color = ( _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR)
            
            # ##################################
            """shelf_color = (
                _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR
            )"""
            # ##################################
            
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # TL - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # TR - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # BR - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BR - Y
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BL - Y
                    ),
                ),
                ("c3B", 4 * shelf_color),
            )
        batch.draw()

    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        # draw goal boxes
        for goal in env.goals:
            raw_goal = tuple(map(int, goal))
            if hasattr(env, "goal_team_ids") and raw_goal in env.goal_team_ids:
                goal_color = self._team_color(env.goal_team_ids[raw_goal])
            else:
                goal_color = _GOAL_COLOR

            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * goal_color),
            )
        batch.draw()

        # draw goal labels
        for goal in env.goals:
            raw_goal = tuple(map(int, goal))
            if hasattr(env, "goal_team_ids") and raw_goal in env.goal_team_ids:
                label_text = f"G{env.goal_team_ids[raw_goal]}"
            else:
                label_text = "G"

            x, y = goal
            y = self.rows - y - 1
            label_x = x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1)
            label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1)
            label = pyglet.text.Label(
                label_text,
                font_name="Calibri",
                font_size=18,
                bold=False,
                x=label_x,
                y=label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_WHITE, 255),
            )
            label.draw()

    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        radius = self.grid_size / 3

        resolution = 6

        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            # make a circle
            verts = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = (
                    radius * math.cos(angle)
                    + (self.grid_size + 1) * col
                    + self.grid_size // 2
                    + 1
                )
                y = (
                    radius * math.sin(angle)
                    + (self.grid_size + 1) * row
                    + self.grid_size // 2
                    + 1
                )
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))

            # ########################################
            # draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
            # ########################################
            if hasattr(env, "agent_team_ids"):
                team_id = env.agent_team_ids[agent.id - 1]
                base_color = self._team_color(team_id)

                # shelf를 들고 있으면 더 밝은 팀 색으로 표시
                draw_color = self._light_color(base_color) if agent.carrying_shelf else base_color
            else:
                draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
            
            glColor3ub(*draw_color)
            circle.draw(GL_POLYGON)

        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1,  # CENTER X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1,  # CENTER Y
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.RIGHT.value else 0
                        )  # DIR X
                        + (
                            -radius if agent.dir.value == Direction.LEFT.value else 0
                        ),  # DIR X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.UP.value else 0
                        )  # DIR Y
                        + (
                            -radius if agent.dir.value == Direction.DOWN.value else 0
                        ),  # DIR Y
                    ),
                ),
                ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
            )
        batch.draw()

    def _draw_badge(self, row, col, index):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(index),
            font_name="Times New Roman",
            font_size=9,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
