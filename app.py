from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from bson.objectid import ObjectId
from utils.extractor import extract_form_structure
import tempfile
import subprocess
import os
import speech_recognition as sr
from pydub import AudioSegment
os.environ["DEEPFACE_BACKEND"] = "torch" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
from deepface import DeepFace
import config
import random
import base64
import io
import cv2
import numpy as np
from pymongo import ReturnDocument
from gtts import gTTS
import uuid
from datetime import datetime

def equalize_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final


app = Flask(__name__)
app.secret_key = config.SECRET_KEY

client = MongoClient(config.MONGO_URI)
db = client.get_default_database()  # bank09

users = db["users"]
accounts = db["accounts"]
transactions = db["transactions"]
kyc = db["kyc"]
admins = db["admins"]

#-------------accounts-----------------------------------------------
def create_account_for_user(user_id):
    """
    Create an account for a user using the account number already stored in users collection.
    """
    user = users.find_one({"_id": user_id})
    if not user:
        raise ValueError("User not found")

    acc_number = user["account_number"]

    # Check if account already exists in accounts collection
    if not accounts.find_one({"account_number": acc_number}):
        accounts.insert_one({
            "user_id": user_id,
            "account_number": acc_number,  # ✅ same as in users
            "balance": 0,
            "created_at": datetime.utcnow()
        })

    return acc_number

#-------------------admin data-------------------------------------------

default_admin = admins.find_one({"email": "admin@bank.com"})
if not default_admin:
    admins.insert_one({
        "name": "Super Admin",
        "email": "admin@bank.com",
        "password": generate_password_hash("1234"),  # default password
        "role": "admin",
        "created_at": datetime.utcnow()
    })
   
#---------------admin route---------------------   
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        admin = admins.find_one({"email": email})
        if admin and check_password_hash(admin["password"], password):
            session["admin_id"] = str(admin["_id"])
            flash("Admin logged in successfully. Redirecting to registration form.", "success")
            return redirect(url_for("register"))   # ✅ Always go to registration form
        else:
            flash("Only admins can sign in", "danger")

    return render_template("admin_login.html")
    
# ---------------- Helpers ----------------
def current_user():
    uid = session.get("uid")
    if not uid:
        return None
    return users.find_one({"_id": ObjectId(uid)})

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

#--------------auto fill-------------------------------------------------------
def merge_autofill_profile(u, kyc_record=None, loan_record=None, deposit_record=None):
    """
    Merge a best-effort 'profile' for auto-filling forms.
    Priority:
    1) Latest domain-specific record (loan for loan form, deposit for FD)
    2) KYC record
    3) User base profile
    """
    prof = {}

    # Base user (from registration)
    if u:
        prof.update({
            "full_name": u.get("name", ""),
            "email": u.get("email", ""),
            "phone_number": u.get("phone_number", ""),
            "aadhaar_number": u.get("aadhaar_number", ""),
            "nominee": u.get("nominee", ""),
            "account_number": u.get("account_number", "")
        })

    # KYC fields (override user if present)
    if kyc_record:
        prof.update({
            "full_name": kyc_record.get("full_name", prof.get("full_name", "")),
            "address": kyc_record.get("address", ""),
            "id_number": kyc_record.get("id_number", ""),
            "phone_number": kyc_record.get("phone_number", prof.get("phone_number", "")),
            "aadhaar_number": kyc_record.get("aadhaar_number", prof.get("aadhaar_number", "")),
            "email": kyc_record.get("email", prof.get("email", "")),
            "id_type": kyc_record.get("id_type", ""),
        })

    # Loan fields
    if loan_record:
        prof.update({
            "loan_type": loan_record.get("loan_type", ""),
            "loan_amount": loan_record.get("loan_amount", ""),
            "duration": loan_record.get("duration", ""),
            "income": loan_record.get("income", ""),
            "employment_status": loan_record.get("employment_status", ""),
            "tenure": loan_record.get("duration", ""),  # alias for template
        })

    # Deposit fields
    if deposit_record:
        prof.update({
            "deposit_type": deposit_record.get("deposit_type", ""),
            "amount": deposit_record.get("amount", ""),
            "duration": deposit_record.get("duration", ""),
            "nominee": deposit_record.get("nominee", prof.get("nominee", "")),
        })

    return prof

# Note: Removed duplicate simple /deposit route that caused endpoint name collision.
# The routes below (services/deposit) implement the actual deposit logic for logged-in users.

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- Registration ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    # Admin must be logged in
    if "admin_id" not in session:
        flash("Admin login required.", "danger")
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").lower().strip()   # ✅ Added email
        password = request.form.get("password", "").strip()     # 4-digit PIN (plain)
        role = request.form.get("role", "user")
        account_number = request.form.get("account_number", "").strip()
        phone_number = request.form.get("phone_number", "").strip()
        aadhaar_number = request.form.get("aadhaar_number", "").strip()
        nominee = request.form.get("nominee", "").strip()
        face_images_json = request.form.get("face_images")  # JSON array of base64 images

        # ✅ Validate password: must be exactly 4 digits
        if not (password.isdigit() and len(password) == 4):
            flash("Password must be a 4-digit number.", "danger")
            return redirect(url_for("register"))

        # ✅ Validate required fields
        if not all([name, email, password, account_number, face_images_json]):
            flash("All fields including face enrollment are required.", "danger")
            return redirect(url_for("register"))

        # ✅ Process face images for embeddings
        embeddings_list = []
        try:
            import json
            face_images = json.loads(face_images_json)
            if not isinstance(face_images, list) or len(face_images) == 0:
                flash("Invalid face images data.", "danger")
                return redirect(url_for("register"))

            for img_b64 in face_images:
                img_bytes = base64.b64decode(img_b64.split(",")[1])
                img_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = equalize_image(img)

                # Generate embedding using DeepFace
                embedding = DeepFace.represent(img, model_name="Facenet512")[0]["embedding"]
                embeddings_list.append(list(embedding))

        except Exception as e:
            flash(f"Face processing error: {str(e)}", "danger")
            return redirect(url_for("register"))

        # ✅ Insert user into MongoDB
        try:
            res = users.insert_one({
                "name": name,
                "email": email,  # ✅ Store email
                "password": password,  # ⚠️ Stored as plain 4-digit PIN
                "account_number": account_number,
                "phone_number": phone_number,
                "aadhaar_number": aadhaar_number,
                "nominee": nominee,
                "role": role,
                "face_embeddings": embeddings_list,
                "created_at": datetime.utcnow()
            })

            # ✅ Initialize account with zero balance
            accounts.insert_one({
                "user_id": res.inserted_id,
                "account_number": account_number,
                "balance": 0.0,
                "created_at": datetime.utcnow()
            })

            flash("✅ User registered successfully with face enrollment.", "success")
            return redirect(url_for("register"))

        except Exception as e:
            flash(f"Database error: {str(e)}", "danger")
            return redirect(url_for("register"))

    # GET request → render form
    return render_template("register.html")

#------------login-----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        u = users.find_one({"email": email})
        if not u or u.get("password", "") != password:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        session["uid"] = str(u["_id"])
        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))
    
    return render_template("login.html")

#-----------logout---------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

#---------time and date-----------------
from datetime import datetime
import pytz

@app.route("/dashboard")
@login_required
def dashboard():
    u = current_user()
    acc = accounts.find_one({"user_id": u["_id"]})

    # Fetch last 5 transactions
    txs = list(transactions.find({"user_id": u["_id"]}).sort("ts", -1).limit(5))

    # Timezones
    utc = pytz.utc
    ist = pytz.timezone("Asia/Kolkata")

    for t in txs:
        ts = t.get("ts")
        if ts:
            if isinstance(ts, datetime):
                # Force treat as UTC → then convert to IST
                if ts.tzinfo is None:
                    ts = utc.localize(ts)
                t["ts"] = ts.astimezone(ist).strftime("%d-%m-%Y %I:%M %p")
            else:
                # If stored as string, parse as UTC first
                try:
                    dt = datetime.fromisoformat(str(ts))
                    if dt.tzinfo is None:
                        dt = utc.localize(dt)
                    t["ts"] = dt.astimezone(ist).strftime("%d-%m-%Y %I:%M %p")
                except:
                    t["ts"] = str(ts)

    return render_template("dashboard.html", user=u, account=acc, txs=txs)

#-------------deposit (services/deposit)---------------------------

@app.route("/services/deposit", methods=["GET", "POST"])
@login_required
def deposit():
    u = current_user()
    acc = accounts.find_one({"user_id": u["_id"]})

    # ✅ If account doesn't exist, create one
    if not acc:
        acc_number = create_account_for_user(u["_id"])
        acc = accounts.find_one({"user_id": u["_id"]})

    if request.method == "POST":
        try:
            amount = float(request.form.get("amount", 0))
        except ValueError:
            flash("Invalid amount.", "danger")
            return redirect(url_for("deposit"))

        if amount <= 0:
            flash("Amount must be positive.", "warning")
            return redirect(url_for("deposit"))

        # Update account balance and get the new balance
        updated_acc = accounts.find_one_and_update(
            {"_id": acc["_id"]},
            {"$inc": {"balance": amount}},
            return_document=ReturnDocument.AFTER
        )

        # ✅ Generate unique receipt number (timestamp + random digits)
        receipt_number = f"DEP{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"

        # Insert transaction record
        transactions.insert_one({
            "user_id": u["_id"],
            "type": "deposit",
            "amount": amount,
            "balance_after": updated_acc["balance"],
            "receipt_number": receipt_number,
            "ts": datetime.utcnow()
        })

        # ✅ Prepare deposit info for printing
        deposit_info = {
            "name": u.get("name"),
            "account_number": u.get("account_number"),
            "amount": amount,
            "balance_after": updated_acc["balance"],
            "receipt_number": receipt_number,
            "datetime": datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
        }

        flash(f"Deposit successful! Updated balance: ₹{updated_acc['balance']}", "success")
        return render_template("services/deposit.html", account=updated_acc, deposit_info=deposit_info)

    # GET request: Just render the form without deposit info
    return render_template("services/deposit.html", account=acc)

# ----------- Withdraw -----------
@app.route("/services/withdraw", methods=["GET","POST"])
@login_required
def withdraw():
    u = current_user()
    acc = accounts.find_one({"user_id": u["_id"]})

    # ✅ If account doesn't exist, create one
    if not acc:
        acc_number = create_account_for_user(u["_id"])
        acc = accounts.find_one({"user_id": u["_id"]})

    if request.method == "POST":
        try:
            amount = float(request.form.get("amount", "0"))
        except ValueError:
            flash("Invalid amount.", "danger")
            return redirect(url_for("withdraw"))

        if amount <= 0:
            flash("Amount must be positive.", "warning")
            return redirect(url_for("withdraw"))

        if acc["balance"] < amount:
            flash("Insufficient balance.", "danger")
            return redirect(url_for("withdraw"))

        # Update balance
        updated_acc = accounts.find_one_and_update(
            {"_id": acc["_id"]},
            {"$inc": {"balance": -amount}},
            return_document=ReturnDocument.AFTER
        )

        # ✅ Generate unique receipt number
        receipt_number = f"WDR{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"

        # Insert transaction record
        transactions.insert_one({
            "user_id": u["_id"],
            "type": "withdraw",
            "amount": amount,
            "balance_after": updated_acc["balance"],
            "receipt_number": receipt_number,
            "ts": datetime.utcnow()
        })

        # ✅ Prepare withdraw info for printing
        withdraw_info = {
            "name": u.get("name"),
            "account_number": u.get("account_number"),
            "amount": amount,
            "balance_after": updated_acc["balance"],
            "receipt_number": receipt_number,
            "datetime": datetime.utcnow().strftime("%d-%m-%Y %I:%M %p")
        }

        flash(f"Withdrawal successful! Updated balance: ₹{updated_acc['balance']}", "success")
        return render_template("services/withdraw.html", account=updated_acc, withdraw_info=withdraw_info)

    # GET request → just form
    return render_template("services/withdraw.html", account=acc)

# ----------- kyc -----------
@app.route("/services/kyc", methods=["GET", "POST"])
@login_required
def kyc_page():
    u = current_user()
    record = kyc.find_one({"user_id": u["_id"]})

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        address = request.form.get("address", "").strip()
        id_number = request.form.get("id_number", "").strip()
        phone_number = request.form.get("phone_number", "").strip()
        aadhaar_number = request.form.get("aadhaar_number", "").strip()
        email = request.form.get("email", "").strip()
        id_type = request.form.get("id_type", "").strip()

        # Validation
        if not full_name or not address or not id_number:
            flash("Full Name, Address, and ID Number are required.", "danger")
            return redirect(url_for("kyc_page"))

        if not phone_number or not aadhaar_number or not email or not id_type:
            flash("All fields are required.", "danger")
            return redirect(url_for("kyc_page"))

        if len(phone_number) != 10:
            flash("Invalid phone number. It must be 10 digits.", "danger")
            return redirect(url_for("kyc_page"))

        if len(aadhaar_number) != 12:
            flash("Invalid Aadhaar Number. It must be 12 digits.", "danger")
            return redirect(url_for("kyc_page"))

        # Prepare data
        data = {
            "full_name": full_name,
            "address": address,
            "id_number": id_number,
            "phone_number": phone_number,
            "aadhaar_number": aadhaar_number,
            "email": email,
            "id_type": id_type,
            "updated_at": datetime.utcnow(),
        }

        if record:
            kyc.update_one({"_id": record["_id"]}, {"$set": data})
        else:
            data["user_id"] = u["_id"]
            data["created_at"] = datetime.utcnow()
            kyc.insert_one(data)

        flash("KYC information saved successfully.", "success")
        return redirect(url_for("dashboard"))

    # ✅ Use merge_autofill_profile so it prefills from user/kyc/loan/deposit
    form_defaults = merge_autofill_profile(u, record, None, None)

    return render_template("services/kyc.html", record=form_defaults)


# ----------- Loan Application -----------
@app.route("/services/loan", methods=["GET", "POST"])
@login_required
def loan_form():
    u = current_user()
    kyc_record = kyc.find_one({"user_id": u["_id"]})
    loan_record = db["loans"].find_one({"user_id": u["_id"]}, sort=[("created_at", -1)])  # last loan

    if request.method == "POST":
        loan_type = request.form.get("loan_type", "").strip()
        amount = request.form.get("loan_amount", "").strip()
        tenure = request.form.get("tenure", "").strip()  # template uses 'tenure'
        income = request.form.get("income", "").strip()
        employment_status = request.form.get("employment_status", "").strip()

        if not loan_type or not amount or not tenure or not income:
            flash("All fields are required for loan application.", "danger")
            return redirect(url_for("loan_form"))

        db["loans"].insert_one({
            "user_id": u["_id"],
            "full_name": request.form.get("full_name", u["name"]),
            "loan_type": loan_type,
            "loan_amount": float(amount),
            "duration": int(tenure),  # store as duration (months)
            "income": float(income),
            "employment_status": employment_status,
            "status": "Pending",
            "created_at": datetime.utcnow()
        })
        flash("Loan application submitted successfully.", "success")
        return redirect(url_for("dashboard"))

    # Server-side prefill (works even if JS is disabled)
    form_defaults = merge_autofill_profile(u, kyc_record, loan_record, None)
    return render_template("services/loan_form.html",
                           user=u, record=kyc_record, loan_record=loan_record, form=form_defaults)



# ----------- Fixed Deposit / Recurring Deposit -----------
@app.route("/services/fixed_deposit", methods=["GET", "POST"])
@login_required
def fixed_deposit():
    u = current_user()
    kyc_record = kyc.find_one({"user_id": u["_id"]})
    deposit_record = db["deposits"].find_one({"user_id": u["_id"]}, sort=[("created_at", -1)])  # last FD

    if request.method == "POST":
        deposit_type = request.form.get("deposit_type", "").strip()
        amount = request.form.get("amount", "").strip()
        duration = request.form.get("duration", "").strip()
        nominee = request.form.get("nominee", "").strip()

        if not deposit_type or not amount or not duration or not nominee:
            flash("All fields are required for Fixed Deposit/Recurring Deposit.", "danger")
            return redirect(url_for("fixed_deposit"))

        db["deposits"].insert_one({
            "user_id": u["_id"],
            "full_name": request.form.get("full_name", u["name"]),
            "deposit_type": deposit_type,
            "amount": float(amount),
            "duration": int(duration),
            "nominee": nominee,
            "status": "Active",
            "created_at": datetime.utcnow()
        })
        flash("Deposit created successfully.", "success")
        return redirect(url_for("dashboard"))

    form_defaults = merge_autofill_profile(u, kyc_record, None, deposit_record)
    return render_template("services/fixed_deposit.html",
                           user=u, record=kyc_record, deposit_record=deposit_record, form=form_defaults)

# ----------- Autofill API (JSON) -----------
@app.route("/api/autofill")
@login_required
def api_autofill():
    """
    Use: /api/autofill?type=kyc|loan|deposit
    Returns a merged profile suitable for client-side auto-fill.
    """
    u = current_user()
    kind = (request.args.get("type") or "").lower()

    kyc_record = kyc.find_one({"user_id": u["_id"]})
    loan_record = None
    deposit_record = None

    if kind == "loan":
        loan_record = db["loans"].find_one({"user_id": u["_id"]}, sort=[("created_at", -1)])
    elif kind in ("deposit", "fixed_deposit", "fd", "rd"):
        deposit_record = db["deposits"].find_one({"user_id": u["_id"]}, sort=[("created_at", -1)])

    prof = merge_autofill_profile(u, kyc_record, loan_record, deposit_record)
    return jsonify({"status": "ok", "data": prof})


# ----------- English page -----------
@app.route("/english")
def english():
    return render_template("english.html")

# ---------------- Face login ----------------
@app.route("/face_login", methods=["POST"])
def face_login():
    import json
    password = request.form.get("password", "").strip()
    face_images_json = request.form.get("face_image", "")

    if not password or not face_images_json:
        flash("Missing password or face image.", "danger")
        return redirect(url_for("login"))

    # Find user by password instead of email
    user = users.find_one({"password": password})
    if not user or "face_embeddings" not in user:
        flash("No face data found for this password.", "danger")
        return redirect(url_for("login"))

    try:
        face_images = json.loads(face_images_json)
        if not isinstance(face_images, list) or len(face_images) == 0:
            flash("Invalid face image data.", "danger")
            return redirect(url_for("login"))

        # Take only the first captured frame
        img_b64 = face_images[0]
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = equalize_image(img_rgb)  # 🌟 Normalize lighting

        # Generate embedding for the captured frame
        embedding = DeepFace.represent(img_rgb, model_name="Facenet512")[0]["embedding"]

        print(f"\n🎯 Face login attempt using password: {password}")
        stored_embeddings = user["face_embeddings"]
        match_found = False

        # Compare with stored embeddings using cosine distance
        for idx, stored_emb in enumerate(stored_embeddings, 1):
            distance = cosine(embedding, stored_emb)
            print(f"\n🔹 Cosine distance to stored embedding #{idx}: {distance:.4f}")
            if distance < 0.4:  # matching threshold
                match_found = True
                break

        if match_found:
            session["uid"] = str(user["_id"])
            flash("✅ Face login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Face not recognized. Try again.", "danger")
            return redirect(url_for("login"))

    except Exception as e:
        print("❌ Face recognition error:", str(e))
        flash(f"Face recognition error: {str(e)}", "danger")
        return redirect(url_for("login"))
    
#----------------colorization-------------------------------------
def equalize_image(img_rgb):
    """
    Apply histogram equalization to the luminance channel of an RGB image
    """
    import cv2
    # Convert RGB to YCrCb
    img_y_cr_cb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
    # Equalize the Y channel (brightness)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    
    # Merge back
    img_eq = cv2.merge([y_eq, cr, cb])
    
    # Convert back to RGB
    img_rgb_eq = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2RGB)
    
    return img_rgb_eq

#------------------transaction-----------------------------------------

# AJAX endpoint to get receiver name from last 4 digits
@app.route("/services/get_receiver_name", methods=["POST"])
@login_required
def get_receiver_name():
    data = request.get_json()
    last4 = data.get("receiver_last4", "").strip()

    if not last4 or len(last4) != 4:
        return jsonify({"status": "error", "message": "Invalid account number"})

    receiver = accounts.find_one({"account_number": {"$regex": f"{last4}$"}})
    if not receiver:
        return jsonify({"status": "error", "message": "Account not found"})

    receiver_user = users.find_one({"_id": receiver["user_id"]})
    return jsonify({"status": "success", "name": receiver_user["name"]})


# Main transaction route
@app.route("/services/transaction", methods=["GET", "POST"])
@login_required
def transaction():
    u = current_user()
    sender_account = accounts.find_one({"user_id": u["_id"]})

    if request.method == "POST":
        data = request.get_json() or request.form

        # sender_last4 comes from logged-in user
        sender_last4 = str(sender_account["account_number"])[-4:]
        receiver_last4 = data.get("receiver_last4", "").strip()
        try:
            amount = float(data.get("amount", 0))
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Invalid amount"})

        if amount <= 0:
            return jsonify({"status": "error", "message": "Amount must be positive"})
        sender = sender_account
        receiver = accounts.find_one({"account_number": {"$regex": f"{receiver_last4}$"}})
        if not receiver:
            return jsonify({"status": "error", "message": "Invalid receiver account number"})
        if sender["_id"] == receiver["_id"]:
            return jsonify({"status": "error", "message": "Cannot transfer to the same account"})
        if sender["balance"] < amount:
            return jsonify({"status": "error", "message": "Insufficient balance"})
        # Perform the transaction
        accounts.update_one({"_id": sender["_id"]}, {"$inc": {"balance": -amount}})
        accounts.update_one({"_id": receiver["_id"]}, {"$inc": {"balance": amount}})
        # Record transactions
        transactions.insert_one({
            "user_id": sender["user_id"],
            "type": "transfer_out",
            "from": sender["account_number"],
            "to": receiver["account_number"],
            "amount": amount,
            "ts": datetime.utcnow()
        })
        transactions.insert_one({
            "user_id": receiver["user_id"],
            "type": "transfer_in",
            "from": sender["account_number"],
            "to": receiver["account_number"],
            "amount": amount,
            "ts": datetime.utcnow()
        })
        # Get updated balances
        updated_sender = accounts.find_one({"_id": sender["_id"]})
        updated_receiver = accounts.find_one({"_id": receiver["_id"]})
        return jsonify({
            "status": "success",
            "message": f"Transferred ₹{amount} successfully",
            "sender_name": u["name"],
            "receiver_name": users.find_one({"_id": receiver["user_id"]})["name"],
            "sender_balance": updated_sender["balance"],
            "receiver_balance": updated_receiver["balance"]
        })
    # GET → render template with sender account prefilled
    return render_template("/services/transaction.html", sender_account=sender_account)

#-------------------------------------------------------------------------
@app.route("/digital")
@login_required
def digital():
    return render_template("digital.html")
UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    fields = extract_form_structure(file_path)
    return jsonify(fields)

#------------------tts----------------------------------------------
@app.route("/tts", methods=["POST"])
def tts():
    try:
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "en")
        if not text.strip():
            return jsonify({"audio": None})
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        audio_bytes = mp3_fp.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return jsonify({
            "audio": "data:audio/mp3;base64," + audio_base64
        })
    except Exception as e:
        print("TTS ERROR:", e)   # 🔥 VERY IMPORTANT
        return jsonify({"error": str(e)}), 500

#---------------stt---------------------------------------------
@app.route("/stt", methods=["POST"])
def stt():
    if "audio" not in request.files:
        return jsonify({"text": ""})

    audio_file = request.files["audio"]
    lang = request.form.get("lang", "en-IN")

    # Normalize language
    lang = lang.lower()

    # Language mapping (ROBUST)
    lang_map = {
        "en": "en-IN",
        "en-in": "en-IN",
        "hi": "hi-IN",
        "hi-in": "hi-IN",
        "kn": "kn-IN",
        "kn-in": "kn-IN"
    }

    stt_lang = lang_map.get(lang, "en-IN")

    input_path = "temp_input.webm"
    output_path = "temp_output.wav"

    audio_file.save(input_path)

    # Convert audio to WAV (16kHz mono)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception as e:
        print("FFmpeg error:", e)
        return jsonify({"text": ""})

    recognizer = sr.Recognizer()
    text = ""

    try:
        with sr.AudioFile(output_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data,
            language=stt_lang
        )

        print("STT Language:", stt_lang)
        print("Recognized Text:", text)

    except Exception as e:
        print("STT error:", e)

    # Cleanup
    for f in [input_path, output_path]:
        if os.path.exists(f):
            os.remove(f)

    return jsonify({"text": text})

#-------------------run app-----------------------------------
if __name__ == "__main__":
    app.run(debug=True)