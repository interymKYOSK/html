from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import json
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_FILE = os.path.join(BASE_DIR, "Meteostat_and_openweathermap_plots_only.html")

logger.info(f"Flask app initialized")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"App root path: {app.root_path}")
logger.info(f"Static folder: {app.static_folder}")

# Ensure Python 3.6 compatibility
if sys.version_info < (3, 7):
    print("Running on Python 3.6 - some features may be limited")


@app.route("/test-static")
def test_static():
    import os

    static_dir = os.path.join(BASE_DIR, "static")
    files = os.listdir(static_dir) if os.path.exists(static_dir) else []
    return jsonify(
        {"static_dir": static_dir, "exists": os.path.exists(static_dir), "files": files}
    )


@app.route("/")
@app.route("/predictions/")
def index():
    logger.info("GET / - Serving index.html")
    try:
        # Try multiple paths for index.html
        index_paths = [
            "index.html",
            os.path.join(os.path.dirname(__file__), "index.html"),
            "/var/www/virtual/zef/html/predictions/index.html",
        ]

        for path in index_paths:
            if os.path.exists(path):
                content = open(path, "r", encoding="utf-8").read()
                logger.info(
                    f"Successfully loaded index.html from {path} ({len(content)} bytes)"
                )
                return content

        logger.error(f"index.html not found in any of: {index_paths}")
        return jsonify({"error": "index.html not found"}), 404
    except Exception as e:
        logger.error(f"Error loading index.html: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/initial-plots")
@app.route("/predictions/initial-plots")
def initial_plots():
    print("Serving:", PLOT_FILE)

    if not os.path.exists(PLOT_FILE):
        return f"File not found: {PLOT_FILE}", 500

    return send_file(PLOT_FILE)


@app.route("/predict", methods=["POST"])
@app.route("/predictions/predict", methods=["POST"])
def predict():
    logger.info(f"POST /predict - Received request with data: {request.json}")
    try:
        # Get form data
        data = request.json
        location = data.get("location", "KYOLO")
        latitude = data.get("latitude", "47.9161926")
        longitude = data.get("longitude", "7.70911552")
        start_date = data.get("start_date", "")
        turbine_power_kW = data.get("turbine_power_kW", "1000")
        pv_power_kWp = data.get("pv_power_kWp", "1000")

        logger.info(
            f"Parameters: loc={location} lat={latitude}, lon={longitude}, date={start_date}, turbine_power_kW={turbine_power_kW}, pv_power_kWp={pv_power_kWp}"
        )

        # Path to your weather prediction script
        script_path = os.path.join(
            os.path.dirname(__file__), "energy_weather_node_past_future.py"
        )
        work_dir = os.path.dirname(__file__)

        logger.info(f"Script path: {script_path}")
        logger.info(f"Work directory: {work_dir}")
        logger.info(f"Script exists: {os.path.exists(script_path)}")

        # Build command with arguments
        cmd = [sys.executable, script_path]

        if location:
            cmd.extend(["--location", location])
        if start_date:
            cmd.extend(["--first_date", start_date])
        if latitude and longitude:
            cmd.extend(["--latitude", str(latitude), "--longitude", str(longitude)])
        if turbine_power_kW:
            cmd.extend(["--turbine_power_kW", str(turbine_power_kW)])
        if pv_power_kWp:
            cmd.extend(["--PV_power_kWp", str(pv_power_kWp)])

        logger.info(f"Executing command: {' '.join(cmd)}")

        # Run the script in the correct directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=work_dir,
        )

        logger.info(f"Script returned code: {result.returncode}")
        if result.stdout:
            logger.info(f"Script stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.error(f"Script stderr: {result.stderr[:500]}")

        if result.returncode != 0:
            logger.error(f"Script execution failed")
            return (
                jsonify(
                    {
                        "error": "Script execution failed",
                        "details": result.stderr,
                        "output": result.stdout,
                        "command": " ".join(cmd),
                    }
                ),
                500,
            )

        # Look for the plots-only HTML file in the working directory
        plots_filename = os.path.join(
            work_dir, "Meteostat_and_openweathermap_plots_only.html"
        )

        logger.info(f"Looking for output file: {plots_filename}")
        logger.info(f"File exists: {os.path.exists(plots_filename)}")

        if os.path.exists(plots_filename):
            with open(plots_filename, "r", encoding="utf-8") as f:
                html_content = f.read()

            logger.info(
                f"Successfully loaded {len(html_content)} bytes from output file"
            )
            # Return the plots directly (already just divs, no body tags)
            return jsonify(
                {"success": True, "html": html_content, "filename": plots_filename}
            )
        else:
            # List files to help debug
            files = [f for f in os.listdir(work_dir) if f.endswith(".html")]
            logger.error(f"Output file not found. Available HTML files: {files}")
            return (
                jsonify(
                    {
                        "error": "Output file not found",
                        "expected": plots_filename,
                        "available_files": files,
                        "work_dir": work_dir,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        logger.error("Script execution timeout")
        return jsonify({"error": "Script execution timeout"}), 500
    except Exception as e:
        import traceback

        logger.error(f"Exception in predict: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
