"""Live plot server for active model trace visualization during streaming."""

import logging
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def start_live_plot_server(csv_path: Path, port: int = 5000) -> None:
    """
    Start a Flask web server that serves live plots of active model trace.

    Args:
        csv_path: Path to the active_model_trace.csv file
        port: Port to run the Flask server on (default: 5000)
    """
    try:
        from flask import Flask
    except ImportError:
        logger.warning(
            "Flask not installed. Install it with: pip install flask matplotlib. "
            "Live plots will not be available."
        )
        return

    app = Flask(__name__)
    csv_path = Path(csv_path)

    def generate_plot_html() -> str:
        """Generate HTML with embedded plot image."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Active Model Trace - Live Plot</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .plot-container {{ margin-top: 20px; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
                .info {{ color: #666; font-size: 14px; margin-top: 10px; }}
                .error {{ color: red; padding: 10px; background-color: #ffe6e6; border-radius: 4px; margin: 10px 0; }}
                .success {{ color: green; padding: 10px; background-color: #e6ffe6; border-radius: 4px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Active Model Trace - Live Visualization</h1>
                <p class="info">Live plot refreshes every 5 seconds. Shows the evolution of selected model index over time.</p>
                <div class="plot-container">
                    {plot_content}
                </div>
                <div class="info">
                    <strong>Last Updated:</strong> {timestamp}
                </div>
            </div>
        </body>
        </html>
        """

        try:
            if not csv_path.exists():
                error_html = '<div class="error">Waiting for trace file to be created...</div>'
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return html_template.format(plot_content=error_html, timestamp=timestamp)

            df = pd.read_csv(csv_path)

            if df.empty:
                error_html = '<div class="error">No data in trace file yet...</div>'
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return html_template.format(plot_content=error_html, timestamp=timestamp)

            # Ensure numeric columns are properly typed
            df["event_index"] = pd.to_numeric(df["event_index"], errors="coerce")
            df["active_model_index"] = pd.to_numeric(df["active_model_index"], errors="coerce")
            df["strategy"] = df["strategy"].astype(str)

            # Drop rows with NaN values that resulted from conversion errors
            df = df.dropna(subset=["event_index", "active_model_index"])

            if df.empty:
                error_html = '<div class="error">No valid data in trace file yet...</div>'
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return html_template.format(plot_content=error_html, timestamp=timestamp)

            # Create figure with subplots for each strategy
            strategies = df["strategy"].unique()
            num_strategies = len(strategies)

            # Calculate grid dimensions
            cols = min(2, num_strategies)
            rows = (num_strategies + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if num_strategies == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, strategy in enumerate(strategies):
                ax = axes[idx]
                strategy_data = df[df["strategy"] == strategy]

                # Sort by event_index to ensure proper ordering
                strategy_data = strategy_data.sort_values("event_index")

                ax.plot(
                    strategy_data["event_index"].values,
                    strategy_data["active_model_index"].values,
                    marker="o",
                    linestyle="-",
                    linewidth=1,
                    markersize=3,
                    alpha=0.7,
                )
                ax.set_xlabel("Event Index")
                ax.set_ylabel("Active Model Index")
                ax.set_title(f"Strategy: {strategy}")
                ax.grid(True, alpha=0.3)

                # Set y-axis to show integer values only
                max_model_idx = int(strategy_data["active_model_index"].max())
                ax.set_ylim(-0.5, max_model_idx + 0.5)
                ax.set_yticks(range(0, max_model_idx + 1))

            # Hide unused subplots
            for idx in range(num_strategies, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()

            # Convert plot to base64 image
            import io
            import base64

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            plot_html = f'<img src="data:image/png;base64,{img_base64}" alt="Active Model Trace Plot">'

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return html_template.format(plot_content=plot_html, timestamp=timestamp)

        except Exception as e:
            logger.exception("Error generating plot: %s", e)
            error_html = f'<div class="error">Error generating plot: {str(e)}</div>'
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return html_template.format(plot_content=error_html, timestamp=timestamp)

    @app.route("/")
    def index():
        """Serve the live plot page."""
        return generate_plot_html()

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok"}, 200

    def run_server():
        """Run the Flask server in a background thread."""
        logger.info("Starting live plot server on http://localhost:%d", port)
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logger.info("Live plot server started. Access it at http://localhost:%d", port)
