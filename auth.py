import bcrypt
from database import create_user, get_user, update_last_login

def hash_password(password):
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed.encode('utf-8')
    )

def register_user(username, password):
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if " " in username:
        return False, "Username cannot contain spaces."
    hashed = hash_password(password)
    success = create_user(username, hashed)
    if success:
        return True, "Account created successfully!"
    return False, "Username already exists. Try another."

def login_user(username, password):
    user = get_user(username)
    if not user:
        return False, None, "Username not found."
    if verify_password(password, user[2]):
        update_last_login(user[0])
        return True, {
            "id": user[0],
            "username": user[1],
            "created_at": user[3],
            "last_login": user[4]
        }, "Login successful!"
    return False, None, "Incorrect password."