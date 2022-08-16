import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_ace import st_ace
from joblib import load

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout='wide')

# Create a sidebar and populate with content
st.sidebar.title("About")
st.sidebar.info(
    """
    [Web App URL](https://inspiritai-tech-demo-heart-disease-0--home-1jfryd.streamlitapp.com/)
    \n
    [GitHub Respository](https://github.com/inspiritai-tech/demo-fake-news-detection)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Inspirit AI: <https://www.inspiritai.com>
    [GitHub](https://github.com/inspiritai-tech) | [LinkedIn](https://www.linkedin.com/company/inspirit-ai/)
    """
)

# Title
st.title("📰 Fake News Detection Powered by Neural Networks")
st.subheader("Interactive News Classification Tool")

# Hero Image
st.image("hero-image.jpg")
st.caption("Photo Credit: [Ludovica Dri](https://unsplash.com/@wanderluly?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/Bc_y35IwUHw). ")

# Introduction
st.header("Anatomy of a News Website")
st.markdown(
'''
Have you ever wondered how websites like google.com and nytimes.com work under the hood?
Using the internet every day, it is easy to forget how magical even the most mundane web browsing experiences
are. Consider, for example this article on The New York Times:
'''
)

# # Motivation

# with st.expander("Why is Detecting Heart Disease Challenging?"):
#   st.markdown(
#     """
#     Typically, patients first learn of their suspected risk from their general practitioners which may involve lengthy
#     assessments, tests, and referrals to specialists like cardiologists. Costly tests including electrocardiograms, echocardiograms, blood tests,
#     MRI scans, CT scans, X-Rays, and more could be run by a patients' medical team to diagnose specific problems.
#     For many, however, access to heathcare poses significant barriers to entry including time, money, insurance, and more.
#     """
#   )

# # Solution
# with st.expander("How AI Can Help"):
#   st.markdown(
#     """
#     What if we could give some autonomy and power back to patients with an AI powered tool that screens individuals
#     based on simple lab results that any doctor or nurse could perform? Here, you will see one such example solution based on a
#     dataset of hundreds of patients with and without heart disease. By training a model to learn the importance of predictors
#     (such as age or cholesterol), this algorithm can help guide patients toward seeking additional care or making lifestyle changes
#     to further reduce their risk.

#     What's more, the larger the dataset becomes, the more accurate this classifer can become, improving predictions across age, ethnicity,
#     race, sex, and more.
#     """
#   )

#   # Demo
# st.header("🫀 Try: AI Risk Assessment Tool 🩺")
# st.markdown(
#   """
#   Enter in your values for **systolic blood pressure** and **maximum heart rate** and let the AI predict whether
#   you are at risk for heart disease.

#   Note: This tool is meant for informational purposes only. Always consult with a medical professional
#   regarding your unique situation. Cardiac risk assessments are not helpful for those who have already
#   had a cardiac event (e.g. heart attack, stroke, or heart failure).
#   """
# )

# # Load DecisionTree Model
# model = load("heart-disease-model.joblib")

# # Collect User Data
# bp = st.number_input("Enter Your Systolic Blood Pressure", min_value=100, max_value = 180, value=120, step=1)
# hr = st.number_input("Enter Your Maximum Heart Rate (during excercise, in beats per minute)", min_value=80, max_value = 230, value=165, step=1)
# input = [[bp, hr]]

# submit = st.button("Submit Response")

# def make_prediction(model, input):
#   return model.predict(input)

# def get_app_response(prediction):
#   if prediction == 1:
#     st.subheader("➡️ You may be at risk for heart disease.")
#     st.warning("Here are some [resources](https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118) to check out to decide on the next best step.")
#   elif prediction == 0:
#     st.subheader("➡️ According to our machine learning model, you are not likely to be at risk for heart disease.")
#     st.success("Here is a [resource](https://health.gov/myhealthfinder/health-conditions/heart-health/keep-your-heart-healthy) on how to maintain your heart health.")
#   else:
#     st.error("Oops, we've run into an error! Try refreshing the page.")

# if submit:
#   prediction = make_prediction(model, input)
#   get_app_response(prediction)
# else:
#   st.subheader("⬆️ Click the Submit Button to Generate Your AI Prediction")

# st.markdown("---")

# # Data Visualization
# st.header("👩‍⚕️ Patient Data Visualization 📋")
# st.markdown(
#   '''
#   Let us take a peek at the dataset that supports our AI model in order visualize some of the key features that are important to consider.
#   Our dataset comes from the [University of California Irvine's Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
#   Below, you can inspect the data itself by either scrolling or viewing in fullscreen mode.
#   You can also sort the data by ascending of descending values for a particular column.
#   '''
# )

# uci_df = pd.read_csv('UCI_data.csv')
# uci_df.rename(columns={'Chest pain type':'CP', 'Cholesterol':'Chol', 'FBS over 120':'FBS', 'EKG results':'EKG', 'Exercise angina':'EA', 'ST depression':'ST', 'Slope of ST':'Slope ST', 'Number of vessels fluro':'Vessels'}, inplace=True)
# st.dataframe(uci_df)

# st.subheader("⚓ Feature Directory")
# st.markdown(
#   '''
#   You may be wondering what each column means. Check out this directory to get familiar with what we're looking at.
#   '''
# )

# with st.expander("Age"):
#   st.markdown("age in years")
# with st.expander("Sex"):
#   st.markdown("sex of patient (1 = male; 0 = female)")
# with st.expander("CP"):
#   st.markdown("chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)")
# with st.expander("BP"):
#   st.markdown("systolic blood pressure")
# with st.expander("Chol"):
#   st.markdown("serum cholesterol in mg/dl")
# with st.expander("FBS"):
#   st.markdown("fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
# with st.expander("EKG"):
#   st.markdown("resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left venticular hypertrophy)")
# with st.expander("Max HR"):
#   st.markdown("maximum heart rate achieved")
# with st.expander("EA"):
#   st.markdown("excercise induced angina (1 = yes; 0 = no)")
# with st.expander("ST"):
#   st.markdown("ST depression induced by excercise relative to rest")
# with st.expander("Slope of ST"):
#   st.markdown("the slope of the peak excercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)")
# with st.expander("Vessels"):
#   st.markdown("number of major vessels (0-3) colored by flourosopy")
# with st.expander("Thallium"):
#   st.markdown("thallium stress test results (3 = normal; 6 = fixed defect; 7 = reversable defect)")
# with st.expander("Heart Disease"):
#   st.markdown("if the patient has been diagnosed with heart disease by a physician")

# # st.markdown(
# #   '''
# #   You may be wondering what each column means. Check out this directory to get familiar with what we're looking at.
# #   1. **Age**: age in years
# #   2. **Sex**: sex of patient (1 = male; 0 = female)
# #   3. **CP**: chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
# #   4. **BP**: systolic blood pressure
# #   5. **Chol**: serum cholesterol in mg/dl
# #   6. **FBS**: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# #   7. **EKG**: resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left venticular hypertrophy)
# #   8. **Max HR**: maximum heart rate achieved
# #   9. **EA**: excercise induced angina (1 = yes; 0 = no)
# #   10. **ST**: ST depression induced by excercise relative to rest
# #   11. **Slope of ST**: the slope of the peak excercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
# #   12. **Vessels**: number of major vessels (0-3) colored by flourosopy 
# #   13. **Thallium**: thallium stress test results (3 = normal; 6 = fixed defect; 7 = reversable defect)
# #   14. **Heart Disease**: if the patient has been diagnosed with heart disease by a physician
# #   '''
# # )

# st.subheader("🔍 At A Glance")
# st.markdown(
#   '''
#   This dataset contains patient data and outcomes from 303 individuals which you can view
#   in the figure below which plots the two features we hightlighted in our AI Heart Disease Detection Tool.
#   The relative size of each circle corresponds to that patient's cholesterol measurement.

#   Tip: You can drag to select an area to zoom in on that data. Try this with your inputted values! 
#   '''
# )

# # Scatter Plot of BP vs Max HR with Size as Chol
# fig1 = px.scatter(uci_df, x= "BP", y="Max HR", color="Heart Disease", size='Chol')
# st.plotly_chart(fig1, use_container_width=True)

# st.markdown(
#   '''
#   Now that we have seen the relationship between blood pressure and maximum heart rate, let's take our other
#   continuous variables and plot them in a matrix scatter plot. These features are age, systolic blood pressure,
#   cholesterol, maximum heart rate, and ST depression induced by excercise relative to rest.
  
#   Note: For your viewing
#   comfort, you may wish to view this visualization in fullscreen mode or utilize the zoom in feature.
#   '''
# )

# # Example 2
# fig2 = px.scatter_matrix(uci_df, dimensions=["Age", "BP", "Chol", "Max HR", "ST"], color="Heart Disease")
# st.plotly_chart(fig2, use_container_width=True)

# st.markdown(
#   '''
#   Another way to inspect the relationship between different features is with this parallel coordinates plot.
#   Each feature in the dataset is included below, and the user is able to visualize the relative dependencies
#   of different features on other features. Every line represents a patient's data.

#   Drag the features along the horizontal axis to reorder the plot and see the relationship between different
#   data points. If you would like to filter and isolate based on a particular feature, click and drag to draw a
#   pink line on the vertical axis of a feature to select only patients who fall within the specified values
#   for those features.

#   Again, for the best user experience, enter full screen mode to play around with the data.
#   '''
# )

# # Example 3
# fig3 = px.parallel_coordinates(uci_df, color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
# st.plotly_chart(fig3, use_container_width=True)

# st.markdown("---")

# # Coding Challenge
# st.header("💻 Coding Challenge: Single Feature Classification 🖱️")
# st.markdown(
#   """
#   Now that you've seen a little bit of the dataset and played around with both the AI model as well
#   as data visualizations for our heart disease problem, try taking a crack at writing a short Python
#   program that make a rudimentary classifier based on a single feature.

#   Complete the function `predict(blood_pressure)` below which takes in a specific value for `blood_pressure`
#   and returns a `1` if the prediction is positive for heart disease and a `0` if the prediction is negative.
#   The prediction is deemed positive or negative depending on whether the value for `blood_pressure` is above
#   or below a `cutoff` value. This value can be determined by you, however the solution assumes an arbitrary `cutoff` value of `130`.
#   """
# )

# st.caption('The code playground below serves as an editing environment and will not compile nor save your work. Check your solution below!')
# code = st_ace(
#         value= 'def predict(blood_pressure): \n ### YOUR CODE HERE ###',
#         language="python",
#         theme="github",
#         font_size=18,
#         tab_size=4,
#         show_gutter=True
#     )

# with st.expander('Code Solution'):
#         st.caption('Caption')
#         st.code(
#           '''
#           def predict(blood_pressure): 
#             cutoff = 130
#             if blood_pressure > cutoff:
#               return 1
#             else:
#               return 0
#           '''
#         )