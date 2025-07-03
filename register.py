import streamlit as st
import sqlite3
import base64
import re
import hashlib

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
add_bg_from_local('image3.jpg')

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Person Re-Identification App!"}</h1>', unsafe_allow_html=True)


# Function to create a database connection
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create the users table if it doesn't exist
def create_users_table():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        phone TEXT NOT NULL,
                        password TEXT NOT NULL);''')
    conn.commit()
    conn.close()

# Function to register a new user
def register_user(name, email, phone, password):
    conn = create_connection()
    password_hash = hashlib.sha256(password.encode()).hexdigest()  # Hash the password for security
    conn.execute("INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                 (name, email, phone, password_hash))
    conn.commit()
    conn.close()

# Function to validate email
def validate_email(email):
    pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    return re.match(pattern, email)

# Function to validate phone number
def validate_phone(phone):
    pattern = r'^[6-9]\d{9}$'
    return re.match(pattern, phone)

# Registration page
def registration_page():
    st.title("Registration")

    # User input fields
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Register button
    if st.button("Register"):
        if password == confirm_password:
            if validate_email(email) and validate_phone(phone):
                create_users_table()
                register_user(name, email, phone, password)
                st.success("You have registered successfully!")
                if st.success("You have registered successfully!"):
                    import subprocess
                    subprocess.run(['streamlit','run','login.py'])
            else:
                st.error("Invalid email or phone number!")
        else:
            st.error("Passwords do not match!")
    if st.button("Back to Login"):
        import subprocess
        subprocess.run(['streamlit','run','login.py'])

# Main function to run the registration page
def main():
    registration_page()

if __name__ == '__main__':
    main()
