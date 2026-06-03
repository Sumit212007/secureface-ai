# Quick Start Guide

## System Overview

Your SecureVerify application is now a **fully functional authentication and verification system** with:

- ✅ Real SQLite database
- ✅ User registration & login
- ✅ Secure password hashing
- ✅ Persistent sessions (AsyncStorage)
- ✅ Verification history tracking
- ✅ Protected routes

---

## Starting the Application

### 1. Start the Backend (Flask API)

```bash
cd f:\Main-Coading-Learning-Folder\govt-hackathon\secureface-ai

# If venv is not activated:
..\venv\Scripts\activate

# Run the server
python api.py
```

**Expected Output:**
```
Loading SecureEdge AI...
SecureEdge AI Loaded
Initializing User Database...
User Database Ready
 * Running on http://0.0.0.0:5000
```

Database automatically created: `secureface-ai/data/identities.db`

---

### 2. Start the Frontend (Expo App)

In a **new terminal**:

```bash
cd f:\Main-Coading-Learning-Folder\govt-hackathon\secure-verify

npm start
# or
yarn start
```

Then press:
- `a` for Android
- `i` for iOS
- `w` for web

---

## Test Credentials

Use these to test the system:

**Test User 1:**
- Email: `test@example.com`
- Password: `password123`

**Test User 2:**
- Email: `demo@example.com`
- Password: `demo1234`

---

## Complete User Flow

### 1. **Registration**
   - Open app → Click "Create Account"
   - Enter name, email, password, confirm password
   - Click "Create Account"
   - Get redirected to login screen

### 2. **Login**
   - Enter email and password
   - Click "Sign In"
   - Get redirected to Dashboard

### 3. **Dashboard**
   - View your profile (name & email at top)
   - See verification statistics
   - Quick action buttons
   - Recent verifications shown
   - Click logout (top right) to exit

### 4. **Start Verification**
   - Click "Start Verification" button
   - Upload document (step 1)
   - Take selfie (step 2)
   - Get verification result
   - Result auto-saved to backend

### 5. **View History**
   - Click "Verification History"
   - See all your verifications
   - Search by verification ID
   - Filter by status (All/ALLOW/DENY)
   - Pull to refresh

---

## Important Configuration

### Backend URL

If your backend is NOT at `192.168.1.108:5000`, update it:

**File:** `secure-verify/src/services/api.ts`

**Line 3:**
```typescript
const API_BASE_URL = 'http://YOUR_IP:5000';
```

---

## Database Files

### Users Database
- **Location:** `secureface-ai/data/identities.db`
- **Tables:**
  - `users` - Registered users with hashed passwords
  - `verification_history` - All verification records

### View Database Contents

```bash
# From secureface-ai directory
python
>>> import sqlite3
>>> conn = sqlite3.connect('data/identities.db')
>>> cursor = conn.cursor()
>>> cursor.execute("SELECT * FROM users")
>>> for row in cursor.fetchall():
...     print(row)
```

---

## API Endpoints Reference

### Authentication Endpoints

**POST /register**
```bash
curl -X POST http://localhost:5000/register \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "John Doe",
    "email": "john@example.com",
    "password": "securepassword123"
  }'
```

**POST /login**
```bash
curl -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "securepassword123"
  }'
```

**GET /history/<user_id>**
```bash
curl http://localhost:5000/history/1
```

**POST /verify**
```bash
curl -X POST http://localhost:5000/verify \
  -F "image=@selfie.jpg" \
  -F "user_id=1"
```

---

## Common Issues & Solutions

### ❌ "Cannot connect to SecureEdge AI backend"

**Solution:** 
1. Backend not running → Start Flask server
2. Wrong IP address → Update in `services/api.ts`
3. Network issue → Both devices on same WiFi

---

### ❌ "Email already registered"

**Solution:** 
1. Use different email address
2. Or delete user from database:
```python
import sqlite3
conn = sqlite3.connect('data/identities.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM users WHERE email='test@example.com'")
conn.commit()
```

---

### ❌ "User not logged in" / "Cannot access dashboard"

**Solution:**
1. Log in first on login screen
2. Clear AsyncStorage (app data)
3. Check console logs for errors

---

### ❌ "History not showing"

**Solution:**
1. Ensure logged in with correct user
2. Complete a verification first
3. Pull to refresh
4. Check backend logs

---

## Project Structure

```
govt-hackathon/
├── secureface-ai/                    # Backend (Flask + ML)
│   ├── api.py                        # ✅ API endpoints
│   ├── database/
│   │   ├── user_store.py             # ✅ DB operations
│   │   └── schema.py
│   ├── pipeline/
│   │   └── orchestrator.py           # Face verification
│   └── data/
│       └── identities.db             # ✅ SQLite database
│
├── secure-verify/                    # Frontend (Expo React Native)
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts                # ✅ API client
│   │   ├── contexts/
│   │   │   └── AuthContext.tsx       # ✅ Auth state
│   │   ├── app/
│   │   │   ├── _layout.tsx           # ✅ Route protection
│   │   │   ├── login.tsx             # ✅ Real auth
│   │   │   ├── register.tsx          # ✅ Real registration
│   │   │   ├── dashboard.tsx         # ✅ Real dashboard
│   │   │   ├── history.tsx           # ✅ Real history
│   │   │   └── verification/
│   │   │       └── index.tsx         # ✅ Saves to backend
│   │   ├── components/
│   │   ├── theme/
│   │   └── constants/
│   └── package.json
│
└── IMPLEMENTATION_GUIDE.md           # ✅ Full documentation
```

---

## Next Steps

1. ✅ **Test the system** using the workflow above
2. ✅ **Check database** to verify data is persisted
3. ✅ **Review code** in key files
4. ✅ **Deploy to devices** for real testing
5. ✅ **Add enhancements** (see IMPLEMENTATION_GUIDE.md)

---

## Key Features Implemented

### Backend
- [x] SQLite user management
- [x] Password hashing (PBKDF2)
- [x] Email uniqueness validation
- [x] Verification history tracking
- [x] User authentication
- [x] Error handling

### Frontend
- [x] Registration with validation
- [x] Login with backend verification
- [x] Dashboard with real user data
- [x] History screen with backend data
- [x] Verification flow with auto-save
- [x] AsyncStorage persistence
- [x] Route protection
- [x] Loading states
- [x] Error display
- [x] Logout functionality

---

## Security Highlights

✅ Passwords hashed with PBKDF2  
✅ No plain text passwords stored  
✅ Email uniqueness enforced  
✅ Input validation on all fields  
✅ Secure session storage  
✅ Transaction-based database operations  
✅ Foreign key constraints  
✅ Error handling without data exposure  

---

## Support & Debugging

### Enable Console Logs
- Backend: Already logging to console
- Frontend: Use React Native Debugger or console.log

### Check Backend Logs
```
[Flask API] Message...
[API] Registration error: ...
```

### Check Frontend Logs
```
[VerificationScreen] message
Register error: ...
```

---

## Commands Reference

```bash
# Backend
cd secureface-ai
..\venv\Scripts\activate
python api.py

# Frontend
cd secure-verify
npm start          # Start development
npm run android    # Android direct
npm run ios        # iOS direct
npm run web        # Web version

# Database
sqlite3 data/identities.db  # Open database directly
```

---

## Questions?

Refer to:
- `IMPLEMENTATION_GUIDE.md` - Full technical guide
- Backend logs - API errors and messages
- React Native Debugger - Frontend state and logs
- Database schema - User table structure

---

**The system is ready to use! Start the backend and frontend, then test the user flow above.** 🚀
