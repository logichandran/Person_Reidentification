import streamlit as st

import base64

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
add_bg_from_local('image2.jpg')

# Function to create the Welcome page
def welcome_page():
    # Title of the app
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"WELCOME"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"ENHANCING PERSON RE-IDENTIFICATION FOR ROBUST SURVEILLANCE SYSTEMS"}</h1>', unsafe_allow_html=True)
    
    # Introduction Text
    st.markdown(f'<h2 style="color:#000000;text-align: center;font-size:24px;">{"This app allows you to perform person re-identification using Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs)."}</h2>', unsafe_allow_html=True)
    
    # Description of the app
    st.markdown(f'<p style="color:#000000;text-align: justify;font-size:16px;">'
                f'Person re-identification (Re-ID) is the task of matching a person across different camera views in surveillance scenarios. '
                f'Using deep learning techniques, this app provides a solution for identifying individuals from images and videos, allowing '
                f'you to track or verify people across different camera feeds. This solution utilizes both DNN and CNN based models to extract features '
                f'and match persons accurately between different camera images.</p>', unsafe_allow_html=True)
    
    # Next Steps for the user
    st.markdown(f'<h3 style="color:#000000;text-align: center;font-size:18px;">{"You can now:"}</h3>', unsafe_allow_html=True)
    st.markdown(f'<ul style="color:#000000;font-size:16px;">'
                f'<li>Upload videos or images of people for identification</li>'
                f'<li>View results of person re-identification</li>'
                f'</ul>', unsafe_allow_html=True)
    
    # Example of buttons for navigating
    st.markdown(f'<h3 style="color:#000000;text-align: center;font-size:18px;">{"Choose your next step:"}</h3>', unsafe_allow_html=True)
    
    if st.button("New User?  Register"):
        import subprocess
        subprocess.run(['streamlit','run','register.py'])
        
    if st.button("Existing User?  Login"):
        import subprocess
        subprocess.run(['streamlit','run','login.py'])
  

# Main function to show the welcome page
def main():
    welcome_page()

# Run the welcome page function
if __name__ == '__main__':
    main()
