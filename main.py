import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background
#
#[theme]
#primaryColor="#F63366"
#backgroundColor="#FFFFFF"
#secondaryBackgroundColor="#F0F2F6"
#textColor="#262730"
#font="sans serif"

#set_background('./bgs/bg5.jpeg')
st.set_page_config(layout='wide')

with st.sidebar:
    selected = option_menu(
    menu_title=None,
    options=["Home","Model","Predict","Types"],
    icons=["house","bookshelf","book","envelope"],
    default_index=0,
    #orientation="horizontal",
)


######################  PREDICT  ###########################################################3

if selected== "Predict":
    #st.set_page_config(layout='wide')



    # set header
    st.header('Please upload a CT scan image')


    st.write(' ')

# upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
    model = load_model('./model/pneumonia_classifier.h5')

# load class names
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

# display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

    # classify image
        class_name, conf_score = classify(image, model, class_names)

    # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))


################### TYPES ############################################################

if selected == "Types":
    #st.set_page_config(layout='wide')


    tab1, tab2, tab3,tab4,tab5 = st.tabs(["Epidural hematoma", "Subdural Hematoma", "Subarachnoid Hemorrhage","Intracerebral hemorrhage","Intraventricular hemorrhage"])


    with tab1:
        st.header("Epidural hematoma")
        #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
        st.write("The temporal area of the head is usually the site of blunt trauma that results in the traditional arterial epidural hematoma. They might also happen following a piercing head injury. Usually, there is a fracture to the skull and bleeding into the possible epidural space due to injury to the middle meningeal artery.")

    with tab2:
        st.header("Subdural Hematoma")
        #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        st.write("When blood enters the subdural space, which is physically the arachnoid space, subdural bleeding takes place. Subdural hemorrhage often happens when a blood artery that connects the brain to the skull is strained, fractured, or ruptured, causing blood to leak into the subdural region.")

    with tab3:
        st.header("Subarachnoid Hemorrhage")
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        st.write("The subarachnoid is leaking blood due to a subarachnoid hemorrhage.   Subarachnoid hemorrhages are classified as either aneurysmal or non-aneurysmal under a second classification method. Aneurysmal subarachnoid hemorrhage happens when a brain aneurysm bursts, causing blood to leak into the subarachnoid space. A subarachnoid hemorrhage that is not associated with an identifiable aneurysm is defined as bleeding into the subarachnoid space.")

    with tab4:
        st.header("Intracerebral hemorrhage")
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        st.write("Internal bleeding within the brain's parenchyma is known as intracerebral hemorrhage (ICH). Numerous factors, including uncontrolled hypertension, burst saccular aneurysms, vascular anomalies, or significant damage, may be to blame for this.The brain's small veins are harmed by high blood pressure, which weakens the arterial wall and increases the risk of rupture.")

    with tab5:
        st.header("Intraventricular hemorrhage")
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
        st.write("Bleeding into the brain's ventricles, which are fluid-filled areas, is known as intraventricular hemorrhage (IVH). Premature newborns are the most prevalent victims of the condition, and the likelihood of IVH increases with the size and prematureness of the child. This is due to the exceedingly weak and immature blood vessels in the brains of preterm newborns. Rarely is IVH present at birth, and when it is, it generally manifests itself in the first few days of life.")



     
    
     
########################### HOME #########################################################
   
if selected == "Home":
    #st.set_page_config(layout='wide')

    # set title
    st.title('DETECTION OF INTRACRANIAL HEMORRHAGE')

    st.write("The term intracranial hemorrhage (ICH) describes bleeding that starts inside the skull and ends up in the brain. Intracerebral hemorrhages come in a variety of forms, each with unique traits and possible causes.Some of its types are:")

    st.write("1.Epidural hematoma")
    st.write("2.Subdural Hematoma")
    st.write("3.Subarachnoid Hemorrhage")
    st.write("4.Intracerebral hemorrhage ")
    st.write("5.Intraventricular hemorrhage")









  
  




   