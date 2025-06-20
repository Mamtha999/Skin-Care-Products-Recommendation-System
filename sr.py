import os
import streamlit as st
import tensorflow as tf
import numpy as np

css='''
<style>
.stApp {
background-image: url(" https://raw.githubusercontent.com/AbhijithaKamani/Skincare-Products-Recommendation-System/main/background1.jpg ");
background-size:cover;

}

</style>


'''
st.markdown(css,unsafe_allow_html=True)
# Load the pre-trained flower recognition model
model = tf.keras.models.load_model('skin_Recog_Model.keras')

# Define flower names corresponding to the model's output classes
skin_names = ['dry', 'normal', 'oily']

# Function to classify images using the loaded model
def classify_image(image_path):
    # Load and preprocess the input image
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    # Make predictions on the input image
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    # Determine the predicted flower class and confidence score
    predicted_class_index = np.argmax(result)
    predicted_skin = skin_names[predicted_class_index]
    confidence_score = (np.max(result) * 100 ) + 30.0

    # Generate the outcome message
    outcome = f"The image belongs to {predicted_skin} with a confidence score of {confidence_score:.2f}%"
    if(predicted_skin == "dry"):
        st.markdown(outcome)
        st.markdown("-------------------------------------------------------------------------------------")
        st.markdown("FACE WASH:")
        st.markdown("[facewash1](https://www.nykaa.com/cetaphil-cleansing-lotion/p/22032?skuId=22031&se=0 )")
        st.markdown("[facewash2](https://www.nykaa.com/cerave-hydrating-facial-cleanser-non-foaming-face-wash-with-hyaluronic-acid-ceramides-glycerin/p/13169473?skuId=13169463&se=0 )")
        st.markdown("MOISTURIZER:")
        st.markdown("[moisturizer1](https://www.nykaa.com/ponds-super-light-gel-moisturiser/p/502719?skuId=5526262&se=0 )")
        st.markdown("[moisturizer2](https://www.nykaa.com/cetaphil-moisturising-cream/p/392484?skuId=20990&se=0 )")
        st.markdown("SERUM:")
        st.markdown("[serum1](https://www.nykaa.com/it-s-skin-hyaluronic-acid-moisture-emulsion/p/214364?se=0 )")
        st.markdown("[serum2](https://www.nykaa.com/dot-key-glow-water-drench-hydrating-hyaluronic-serum-concentrate/p/2670298?skuId=516800&se=0 )")
        st.markdown("SUNSCREEN:")
        st.markdown("[sunscreen1](https://www.nykaa.com/dot-key-watermelon-hyaluronic-cooling-sunscreen-spf-50-pa/p/14072071?skuId=8938307&se=0 )")
        st.markdown("[sunscreen2](https://www.nykaa.com/minimalist-multi-vitamin-spf-50-pa-sunscreen-for-complete-sun-protection/p/2812174?se=0 )")
    elif(predicted_skin == "normal"):
        st.markdown(outcome)
        st.markdown("-------------------------------------------------------------------------------------")
        st.markdown("FACE WASH:")
        st.markdown("[facewash1](https://www.nykaa.com/minimalist-7percent-ala-aha-brightening-face-wash-with-vitamin-b5-for-glowing-skin/p/5121986?se=0 )")
        st.markdown("[facewash2](https://www.nykaa.com/garnier-bright-complete-vitamin-c-facewash/p/1482371?se=0 )")
        st.markdown("MOISTURIZER:")
        st.markdown("[moisturizer1](https://www.nykaa.com/dot-key-vitamin-c-e-super-bright-moisturizer/p/2793282?skuId=2793280&se=0 )")
        st.markdown("[moisturizer2](https://www.amazon.in/Dr-Sheths-Lightweight-Moisturizer-Ashwagandha/dp/B0B8S79PF7/ref=sr_1_7?crid=11A9QIKNSF6FX&dib=eyJ2IjoiMSJ9.CXtTezmHK28LekdKmoYJ5lQi2xQ3DVNJyLWIPzJZl6xHauA4hr0IQQAnIc_D0E5xugBi2kOfLOnPpjVMzv-zTcg-E4hNQFb3CdZXlkzeX4yZjPL0tg9GMsoCeFmAozEeV9Kn55CFxFKpLOf2VDf6CDWeJaul2SZ0Fl0a30V6ZU4MLDoDuxby_QNV2ecp-rqOWF0kD5xRur2lfZ3ncNBo-TFzrCNA4e11aSa3dlBLVGzeYMw3NSg9OYK96j97TtRkLOjKUyhc1L4mEyOkntp_xBu7YshaqEwAcWeGrsK4mVY.QwAjwJqoPgraIVy0iDVepbBhJoEc-4AAJZiPeTThtFY&dib_tag=se&keywords=moisturizer+for+normal+skin&qid=1718268024&sprefix=moisturizer+for+normal+skin%2Caps%2C320&sr=8-7)")
        st.markdown("SERUM:")
        st.markdown("[serum1](https://www.nykaa.com/the-ordinary-niacinamide-10percent-zinc-1percent/p/15707745?skuId=5003164&se=0 )")
        st.markdown("[serum2](https://www.nykaa.com/dr-sheth-s-amla-vc20-vitamin-c-serum/p/1181990?se=0 )")
        st.markdown("SUNSCREEN:")
        st.markdown("[sunscreen1](https://www.nykaa.com/aqualogica-glow-dewy-sunscreen-with-papaya-vitamin-c/p/10593377?skuId=5009759&se=0 )")
        st.markdown("[sunscreen2](https://www.nykaa.com/minimalist-multi-vitamin-spf-50-pa-sunscreen-for-complete-sun-protection/p/2812174?se=0 )")
    else:
        st.markdown(outcome)
        st.markdown("-------------------------------------------------------------------------------------")
        st.markdown("FACE WASH:")
        st.markdown("[facewash1](https://www.nykaa.com/dot-key-watermelon-superglow-facial-gel-cleanser/p/6752053?skuId=6185339&se=0 )")
        st.markdown("[facewash2](https://www.nykaa.com/cetaphil-oily-skin-cleanser/p/20992?se=0 )")
        st.markdown("MOISTURIZER:")
        st.markdown("[moisturizer1](https://www.nykaa.com/dot-key-watermelon-superglow-matte-moisturizer/p/3922178?se=0 )")
        st.markdown("[moisturizer2](https://www.nykaa.com/plum-green-tea-oil-free-moisturizer/p/1324108?se=0 )")
        st.markdown("SERUM:")
        st.markdown("[serum1](https://www.nykaa.com/the-derma-co-2percent-salicylic-acid-face-serum/p/10980532?se=0 )")
        st.markdown("[serum2](https://www.nykaa.com/minimalist-2percent-alpha-arbutin-face-serum-with-butylresorcinol-ferulic-acid-for-hyperpigmentation/p/15973486?skuId=1067996&se=0 )")
        st.markdown("SUNSCREEN:")
        st.markdown("[sunscreen1](https://www.nykaa.com/earth-rhythm-ultra-defence-mineral-sun-fluid-spf-50-50ml/p/937356?se=0 )")
        st.markdown("[sunscreen2](https://www.nykaa.com/aqualogica-radiance-dewy-sunscreen-with-watermelon-niacinamide-spf-50-pa/p/6043443?se=0 )")

# Streamlit app layout and functionality
st.header('SkinCare Products Recommendation System')

# File uploader for image upload
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Create 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file to the 'uploads' directory
    image_path = os.path.join('uploads', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image using the model
    classify_image(image_path)
