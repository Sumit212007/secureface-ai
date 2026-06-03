from flask import Flask, request, jsonify
from pipeline.orchestrator import AuthPipeline
from database.user_store import UserStore
import cv2
import numpy as np
import uuid

app = Flask(__name__)

print("Loading SecureEdge AI...")
pipeline = AuthPipeline(
    require_blink=False,
    cosine_threshold=0.55,
    min_gap=0.05,
)
print("SecureEdge AI Loaded")

print("Initializing User Database...")
user_store = UserStore()
print("User Database Ready")


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "SecureEdge AI"
    })


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    """Register a new user account."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
        
        full_name = data.get("full_name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        
        # Validation
        if not full_name:
            return jsonify({
                "success": False,
                "message": "Full name is required"
            }), 400
        
        if not email or "@" not in email:
            return jsonify({
                "success": False,
                "message": "Valid email is required"
            }), 400
        
        if not password or len(password) < 8:
            return jsonify({
                "success": False,
                "message": "Password must be at least 8 characters"
            }), 400
        
        # Register user
        user = user_store.register_user(full_name, email, password)
        
        return jsonify({
            "success": True,
            "message": "Registration successful",
            "user": user
        }), 201
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400
    except Exception as e:
        print(f"[API] Registration error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Registration failed"
        }), 500


@app.route("/login", methods=["POST"])
def login():
    """Authenticate user and return user data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
        
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        
        if not email or not password:
            return jsonify({
                "success": False,
                "message": "Email and password required"
            }), 400
        
        # Authenticate
        user = user_store.authenticate_user(email, password)
        
        if user is None:
            return jsonify({
                "success": False,
                "message": "Invalid email or password"
            }), 401
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": user
        }), 200
        
    except Exception as e:
        print(f"[API] Login error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Login failed"
        }), 500


@app.route("/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    """Get verification history for a user."""
    try:
        history = user_store.get_verification_history(user_id)
        
        return jsonify({
            "success": True,
            "history": history
        }), 200
        
    except Exception as e:
        print(f"[API] History retrieval error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Failed to retrieve history"
        }), 500



@app.route("/verify", methods=["POST"])
def verify():
    print("\n--- [Flask API] Incoming /verify POST request ---")
    print(f"[Flask API] Content-Type: {request.content_type}")
    print(f"[Flask API] Files available: {list(request.files.keys())}")

    if "image" not in request.files:
        print("[Flask API] Error: 'image' part missing from request.files")
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        }), 400

    try:
        file = request.files["image"]
        print(f"[Flask API] File info: filename='{file.filename}', content_type='{file.content_type}'")

        image_bytes = file.read()
        print(f"[Flask API] Successfully read {len(image_bytes)} bytes from file upload")

        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            print("[Flask API] Error: cv2.imdecode returned None, image data might be corrupt")
            raise ValueError("Decoded frame is None")
        print(f"[Flask API] Image decoded successfully. Resolution: {frame.shape}")

        print("[Flask API] Executing authentication pipeline...")
        result = pipeline.authenticate(frame)
        print(f"[Flask API] Pipeline Auth Result - Decision: {result.decision.value}, Identity: {result.identity}, Similarity: {result.similarity:.4f}, Liveness Score: {result.liveness_score:.4f}")

        response_data = {
            "success": True,
            "decision": result.decision.value,
            "identity": result.identity,
            "similarity": float(result.similarity),
            "liveness_score": float(result.liveness_score),
            "liveness_decision": result.liveness_decision,
            "face_detected": result.face_detected,
            "processing_time_ms": float(result.processing_time_ms),
            "message": result.error_message
        }
        
        # Save to verification history if user_id is provided and verification was successful
        user_id = request.form.get("user_id")
        if user_id and result.decision.value == "ALLOW":
            try:
                verification_id = str(uuid.uuid4())[:8].upper()
                user_store.add_verification_history(
                    user_id=int(user_id),
                    verification_id=verification_id,
                    decision=result.decision.value,
                    similarity=float(result.similarity),
                    liveness_score=float(result.liveness_score)
                )
                response_data["verification_id"] = verification_id
                print(f"[Flask API] Saved verification history for user {user_id}")
            except Exception as e:
                print(f"[Flask API] Warning: Failed to save verification history: {str(e)}")
                # Don't fail the verification response if history save fails
        
        return jsonify(response_data)

    except Exception as e:
        print(f"[Flask API] Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/enroll", methods=["POST"])
def enroll():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        }), 400

    label = request.form.get("label")

    if not label:
        return jsonify({
            "success": False,
            "message": "label required"
        }), 400

    try:

        file = request.files["image"]

        image_bytes = file.read()

        npimg = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        success = pipeline.enroll(frame, label)

        return jsonify({
            "success": success,
            "label": label
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )