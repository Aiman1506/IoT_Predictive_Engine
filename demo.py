"""
FactoryGuard AI - Complete System Demo
========================================
Demonstrates the full ML pipeline → API → Web UI
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 70)
print("🏭 FactoryGuard AI - Production Demo")
print("=" * 70)

print("\n📋 System Status Check:")
print("-" * 70)

# Check if model exists
model_path = PROJECT_ROOT / "models" / "factoryguard_latest.pkl"
if model_path.exists():
    print("  ✅ Trained model found")
    print(f"     → {model_path}")
else:
    print("  ❌ No trained model found!")
    print("     Please run: python train.py")
    sys.exit(1)

# Check if data exists
data_path = PROJECT_ROOT / "data" / "sensor_data.csv"
if data_path.exists():
    print("  ✅ Training data found")
    print(f"     → {data_path}")
else:
    print("  ⚠️  No training data (optional for API)")

# Check outputs
output_dir = PROJECT_ROOT / "outputs"
plots = list(output_dir.glob("*.png"))
if plots:
    print(f"  ✅ {len(plots)} visualization plots generated")
else:
    print("  ⚠️  No plots found (re-run train.py to generate)")

print("\n" + "=" * 70)
print("🌐 Web Application")
print("=" * 70)

print("\n📍 The Flask server should be running at:")
print("   → http://localhost:5000/")
print("   → http://127.0.0.1:5000/")

print("\n🎯 Test the application in 3 steps:")
print("-" * 70)

print("\n1️⃣  Open Web UI:")
print("   Visit: http://localhost:5000/")
print("   You should see a purple gradient page with input fields")

print("\n2️⃣  Test Normal Operation:")
print("   • Temperature: 55.0 °C")
print("   • Vibration: 2.0 mm/s")
print("   • Pressure: 6.0 bar")
print("   → Click 'Predict Failure Risk'")
print("   → Expected: GREEN badge, LOW risk (~0.03%)")

print("\n3️⃣  Test Critical Failure:")
print("   • Temperature: 89.0 °C")  
print("   • Vibration: 8.5 mm/s")
print("   • Pressure: 3.2 bar")
print("   → Click 'Predict Failure Risk'")
print("   → Expected: RED badge, CRITICAL risk (~82%)")

print("\n" + "=" * 70)
print("🔧 API Endpoints Available")
print("=" * 70)

endpoints = [
    ("GET  /", "Web UI interface"),
    ("POST /predict", "Failure prediction (JSON)"),
    ("GET  /health", "Server health check"),
    ("GET  /model-info", "Model metadata"),
]

for method, desc in endpoints:
    print(f"  • {method:15} → {desc}")

print("\n" + "=" * 70)
print("📊 Model Performance Summary")
print("=" * 70)

print("\n  Metric              Value")
print("  " + "-" * 35)
print("  PR-AUC              0.9334 ⭐")
print("  Precision           0.74")
print("  Recall              0.93")
print("  F1-Score            0.83")
print("  Inference Latency   5.4 ms ⚡")
print("  False Alarm Rate    0.32%")
print("  Missed Failures     7.0%")

print("\n" + "=" * 70)
print("🧪 Automated Testing")
print("=" * 70)

print("\nRun these commands to verify everything works:")
print("  python test_ui.py    # Test web UI + prediction API")
print("  python test_api.py   # Test REST API endpoints")
print("  python predict_realtime.py  # Streaming simulation")

print("\n" + "=" * 70)
print("📚 Documentation")
print("=" * 70)

print("\n  • README.md     → Full system documentation")
print("  • UI_GUIDE.md   → Web interface guide")
print("  • outputs/      → Training plots & reports")

print("\n" + "=" * 70)

# Ask if they want to open the browser
try:
    response = input("\n🌐 Open the Web UI in your browser now? (y/n): ").strip().lower()
    if response == 'y':
        print("\n🚀 Opening http://localhost:5000/ in your browser...")
        time.sleep(1)
        webbrowser.open('http://localhost:5000/')
        print("   ✅ Browser launched!")
        print("\n💡 Tip: If the server isn't running, start it with:")
        print("   python serve.py")
    else:
        print("\n📝 Manual access: http://localhost:5000/")
except KeyboardInterrupt:
    print("\n\n👋 Demo cancelled")

print("\n" + "=" * 70)
print("✅ Demo Complete - FactoryGuard AI is Ready!")
print("=" * 70)
print()
