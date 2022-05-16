from fastai.vision.all import *
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import pathlib
import platform

plt = platform.system()
if plt == 'Linux':pathlib.WindowsPath = pathlib.PosixPath

# LOAD

def load_lottier(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = 'https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json'
lottie_code = load_lottier(lottie_url)

# page config
st.set_page_config(page_title='Shahzodbek\'s app', page_icon=':tada:')

#navigation 
with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Home', 'Projects', 'Contact'],
        icons=['house-fill', 'box', 'envelope-open'],
        default_index=0,
        menu_icon='cast'
    )


# Home
if selected == 'Home':
    with st.container():
        st.subheader('HI, I\'M SHAHZODBEK :wave:')
        st.title('A DATA SCIENTIST FROM UZBEKISTAN')
        st.write('I\' passionate about finding ways to use data science tools in real life to make people\'s life easier and better.')

    with st.container():
        st.write('---')
        left_column, right_column = st.columns(2)
        
        # left_column
        with left_column:
            st.write('What I do')
            st.write('##')
            st.write(
                """
                I am 3-year biotechnology student.
                Recently, I discovered about data science and how awesome it is.
                Now I'm learning to use in my field and more from several recources.
                Check out my projects and Share you thoughts in Contact section.
                """
            )
        
        # right column
        with right_column:
            st_lottie(lottie_code)

# Projects
if selected == 'Projects':
    navbar = option_menu(
        menu_title= None,
        options=['Birds Classifier', 'Absenteeism'], # project labels
        icons=['camera-reels', 'person-x'],
        default_index=0,
        menu_icon='cast',
        orientation='horizontal'
    )

    # Birds model
    if navbar == 'Birds Classifier':
        model = load_learner('birds_model2.pkl')

        st.subheader('This model can classify 10 different birds.')
        st.write('Birds: Woodpecker Penguin Raven Chicken Eagle Owl Duck Goose Swan Falcon Parrot Turkey')
        st.write('---')
        st.write('To use this model upload your image to upload field')
        file = st.file_uploader('Images:', type=['gif', 'png', 'jpg', 'jpeg', 'svg'])

        if file:
            #PIL corvert
            img = PILImage.create(file)
            st.image(file)

            #prediction
            pred, pred_id, probs = model.predict(img)
            if pred not in ['Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl', 'Duck', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Turkey']:
                st.warning('Please, upload image of birds.')
            else:
                st.success(f"Result: {pred}")
                st.info(f'Probability: {probs[pred_id]*100:.1f}%')
    
    # Absenteeism model
    if navbar == 'Absenteeism':
        st.subheader('This is a absenteeism probability estimater model')
        st.write('Based on data you provided returns data with probability and prediction columns')
        genre = st.radio(
            "Choose your data input type",
            ('Demo', 'Manual'))

        df = pd.read_csv('Absenteeism_new_data.csv')
        from absent_model import *
        model_absent = absenteeism_model('model', 'scaler')

        # Demo
        if genre == 'Demo':
            st.write('This is our data')
            model_absent.load_and_clean_data('Absenteeism_new_data.csv')

            st.dataframe(df)

            st.write('---')
            st.write('This is our result')
            prediction = model_absent.predicted_outputs()
            st.dataframe(prediction)

            # Download button
            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(prediction)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='prediction_df.csv',
                mime='text/csv',
            )
        # Manual
        else:
            st.info('Your file should be the same format as shown on Demo section')
            f = st.file_uploader('Your data', ['csv'])
            if f:
                df_check = pd.read_csv(f)
                st.dataframe(df_check)
                # Check data
                if len(df_check.columns) == len(df.columns):
                    df_check.columns = df.columns.values
                    df_check.to_csv('Working_data.csv', index=False)
                    model_absent.load_and_clean_data('Working_data.csv')
                    st.write('---')
                    st.write('This is our result')
                    prediction = model_absent.predicted_outputs()
                    st.dataframe(prediction)
                    # Download Button
                    @st.cache
                    def convert_df(df):
                        # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(prediction)

                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='prediction_df.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning('Please, upload required file.')

# Contact
if selected == 'Contact':
    st.header(':mailbox: Get in touch with me!')
    contact_form = """
    <form action="https://formsubmit.co/shahtesting101@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder='Your name' required>
        <input type="email" name="email" placeholder='Your email' required>
        <textarea name="message" placeholder="Your thoughts"></textarea>
        <button type="submit">Send</button>
    </form> 
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    # css file
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    local_css('styles.css')
    st.subheader('If you are interested my other projects check out my Github page:')
    st.write('[Github page >](https://github.com/Shahzodbey)')
