# Python In-built packages
from pathlib import Path
import PIL
import requests

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Function to translate text using RapidAPI
def translate_text_rapidapi(text_to_translate, target_language):
        if target_language == 'en':
            return text_to_translate
            
        url = "https://google-translation-unlimited.p.rapidapi.com/translate"
        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'X-RapidAPI-Key': '4db7d01892msh1a35513855d61b8p176a71jsn3c4545830608',  # Replace with your RapidAPI key
            'X-RapidAPI-Host': 'google-translation-unlimited.p.rapidapi.com'
        }
        data = {
            'texte': text_to_translate,
            'to_lang': target_language
        }

        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            translation_data = response.json().get('translation_data', {})
            if translation_data:
                translated_text = translation_data.get('translation', '')
                return translated_text
            else:
                return "Error: Unexpected response format"
        else:
            return f"Error: {response.status_code}"

# Define class information
class_info = {
    0: {
        'name': 'Alternaria',
        'description': 'This image represents pomegranate affected by Alternaria. Alternaria fruit rot symptoms typically appear as dark  on the surface of pomegranate fruit. These lesions may start as small spots and gradually enlarge, often becoming covered with a velvety black fungal growth. As the disease progresses, the infected areas may become soft and mushy, leading to extensive fruit rot. In severe cases, the entire fruit may be affected.Two preventive sprayings during the blooming period or when the first symptoms appear on the fruits give good control of the disease. Product based on propiconazole, thiophanate methyl or azoxystrobine have been proved highly effective.It is caused dues to alternate fungi. The fungus grows inside pomegranate.Prevent it by fertilising the crop, good field sanitation, remove old fruits.',
    },
    1: {
        'name': 'Anthracnose',
        'description': 'This image represents Anthracnose. Anthracnose in pomegranates, caused by the fungus Glomerella cingulata, shows up as black spots with a yellow halo on leaves, premature leaf dropping, and dark discoloration on fruits. This fungus sticks around in infected plant leftovers and spreads when it rains or the wind blows in spring. To stop it, use healthy plants, ones that can handle the fungus, and keep the area well-ventilated and clean by getting rid of fallen leaves and sick branches. It is also good to spray stuff that kills the fungus early on and to keep your tools clean. Just keep an eye out for any signs of trouble, and if you see any, take action to stop it from spreading.',
    },
    2: {
        'name': 'Bacterial_Blight',
        'description': 'This image represents Bacterial Blight. Bacterial blight in pomegranates, caused by Bacterium Xanthomonas, leads to visible symptoms like yellowish water-soaked circular spots on the plants within 2-3 days of infection, often resulting in leaves falling prematurely. As the disease worsens, these spots become irregular and turn dark brown at their centers, affecting the stems and branches by causing cracking and girdling. This bacterial infection can also lead to tissue death on leaves and twigs and may cause fruits to crack open, dry out, and darken. Anyone can recognize the signs, and preventive steps include using natural remedies like neem leaves and garlic extract, along with proper sanitation and planting practices. It is crucial to remove infected plant parts regularly and avoid waterlogging to protect the crop.',
    },
    3: {
        'name': 'Cercospora',
        'description': 'This image represents Cercospora. Cercospora fruit spot symptoms typically appear as small, circular to irregularly shaped lesions on the fruit. These lesions start as yellow or light brown spots and gradually darken to reddish-brown or purple with a lighter center. As the disease progresses, the spots may increase in size. This can be prevented by spraying at 15days interval of a fungicide. This disease occurs due different soil concentration and rainfall. Infection spreads fast due to humidity and rainfall. Prevent it by  fertilising the crop, good field sanitation..',
    },
    4: {
        'name': 'Healthy',
        'description': 'This image represents a healthy, disease-free pomegranate. Maintain good orchard hygiene, provide adequate nutrition, and monitor for early signs of diseases to keep your pomegranate plants healthy.',
    },
}

# Define languages
languages = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'fa': 'Persian',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'es': 'Spanish',
    'fr': 'French',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ur': 'Urdu',
    'ka': 'Georgian',
    'it': 'Italian',
    'el': 'Greek',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Oriya',
    'pa': 'Punjabi'
}

# Setting page layout
st.set_page_config(
    page_title="Pomegranate Disease Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Select target language
selected_language_value = st.sidebar.selectbox("Select Language:", list(languages.values()))
selected_language = next(key for key, value in languages.items() if value == selected_language_value)

# Main page heading
st.title("Pomegranate Disease Detection")
st.caption('Then click the :blue[Detect Objects] button and check the result.')

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Load Pre-trained ML Model
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# Handle different sources
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = None
        if source_img is not None:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image", use_column_width=True)

    with col2:
        if uploaded_image is not None:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                confidences = boxes.conf
                class_labels = boxes.cls
                # Display the detected image with bounding boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)

                try:
                    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                    with st.expander("Detection Results"):
                        detected_classes = set()
                        for i, box in enumerate(boxes):
                            class_label = int(class_labels[i])  # Convert to int if needed
                            confidence = float(confidences[i])  # Convert to float if needed
                            detected_class_info = class_info.get(class_label, {})

                            # Check if the detected class has already been displayed
                            if detected_class_info['name'] not in detected_classes:
                                detected_classes.add(detected_class_info['name'])

                                # Display detected class name, confidence, and description
                                st.markdown(f"<span style='color: blue;'>{detected_class_info.get('name', 'Unknown')}</span>", unsafe_allow_html=True)
                                st.write(f"Confidence: {confidence}")

                                # Translate the description if selected language is not English
                                if selected_language != 'en':
                                    translated_description = translate_text_rapidapi(detected_class_info.get('description', ''), selected_language)
                                    st.write(f"Translated Description: {translated_description}")
                                else:
                                    st.write(f"Description: {detected_class_info.get('description', '')}")

                except Exception as ex:
                    st.write("No image is uploaded yet!")

# Handle other sources (video, webcam)
elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")

