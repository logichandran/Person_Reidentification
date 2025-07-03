import streamlit as st
import sqlite3
import hashlib
import base64

# ==================== Background image ===
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_local('image3.jpg')


# Function to create a database connection
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to verify login credentials
def verify_user(email, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user:
        # Check if the entered password matches the stored hash
        stored_password = user[4]  # password column in DB
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        if stored_password == password_hash:
            return True
    return False

# Login page
def login_page():
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if verify_user(email, password):
            st.success("Login successful!")
            if st.success("Login successful!"):
                import subprocess
                subprocess.run(['streamlit','run','prediction.py'])
        else:
            st.error("Invalid credentials. Please try again.")

    if st.button("Registration"):
        import subprocess
        subprocess.run(['streamlit','run','register.py'])

# Main function to run the login page
def main():
    login_page()

if __name__ == '__main__':
    main()
