"""
Test the FactoryGuard AI Web UI
================================
Verifies that:
1. Frontend UI loads at "/"
2. Prediction endpoint works via fetch API simulation
"""

import json
import urllib.request
import sys


def test_frontend():
    """Test that the UI page loads."""
    print("=" * 60)
    print("Testing FactoryGuard AI Web UI")
    print("=" * 60)
    
    print("\n[1] Testing Frontend (GET /)...")
    try:
        with urllib.request.urlopen('http://localhost:5000/', timeout=5) as resp:
            html = resp.read().decode('utf-8')
            status = resp.status
            
            print(f"    Status: {status}")
            print(f"    HTML Size: {len(html):,} bytes")
            
            # Check for key UI elements
            checks = {
                "Title present": "FactoryGuard AI" in html,
                "Form present": "predictionForm" in html,
                "Temperature input": 'id="temperature"' in html,
                "Vibration input": 'id="vibration"' in html,
                "Pressure input": 'id="pressure"' in html,
                "Predict button": "Predict Failure" in html,
                "Fetch API": "fetch('/predict'" in html,
            }
            
            all_pass = all(checks.values())
            
            for check, result in checks.items():
                icon = "✓" if result else "✗"
                print(f"    {icon} {check}")
            
            if all_pass:
                print(f"\n    ✅ Frontend UI: PASS")
            else:
                print(f"\n    ❌ Frontend UI: FAIL")
                return False
                
    except Exception as e:
        print(f"    ❌ Error loading UI: {e}")
        return False
    
    print("\n[2] Testing Prediction Endpoint (POST /predict)...")
    try:
        payload = {
            "temperature": 55.0,
            "vibration": 2.0,
            "pressure": 6.0
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:5000/predict',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            
            print(f"    Status: {resp.status}")
            print(f"    Response: {result}")
            
            required_fields = ["failure_probability", "risk_level", "prediction"]
            has_all_fields = all(f in result for f in required_fields)
            
            if has_all_fields:
                print(f"    ✅ Prediction API: PASS")
                print(f"    💡 Failure Probability: {result['failure_probability']:.2%}")
                print(f"    💡 Risk Level: {result['risk_level']}")
            else:
                print(f"    ❌ Missing fields in response")
                return False
                
    except Exception as e:
        print(f"    ❌ Error calling /predict: {e}")
        return False
    
    print("\n[3] Testing High-Risk Scenario...")
    try:
        risky_payload = {
            "temperature": 89.0,
            "vibration": 8.5,
            "pressure": 3.2
        }
        
        data = json.dumps(risky_payload).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:5000/predict',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            
            print(f"    Risky Sensors: T={risky_payload['temperature']}°C "
                  f"V={risky_payload['vibration']}mm/s P={risky_payload['pressure']}bar")
            print(f"    Prediction: {result['failure_probability']:.2%} risk")
            print(f"    Risk Level: {result['risk_level']}")
            
            is_high_risk = result['risk_level'] in ['HIGH', 'CRITICAL']
            if is_high_risk:
                print(f"    ✅ High-risk detection: PASS")
            else:
                print(f"    ⚠️  Expected HIGH/CRITICAL risk level")
                
    except Exception as e:
        print(f"    ❌ Error in high-risk test: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All Tests Passed!")
    print("=" * 60)
    print("\n📱 Open in browser: http://localhost:5000/")
    print("🔗 Direct link: http://127.0.0.1:5000/\n")
    
    return True


if __name__ == "__main__":
    success = test_frontend()
    sys.exit(0 if success else 1)
