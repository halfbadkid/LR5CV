import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ---------------------------------------------------------
# Step 1: Create a new Streamlit application using Python and configure an appropriate page title and layout.
# ---------------------------------------------------------
st.set_page_config(page_title="NeuralSight CV", layout="wide")

st.title("VisionNet: CPU-Based Image Classifier")
st.info("This system leverages Deep Learning to categorize visual data on a CPU-only environment.")

# ---------------------------------------------------------
# Step 2: Import the required libraries including Streamlit, PyTorch, Torchvision, PIL, and Pandas.
# Step 3: Configure the application to run only on CPU settings.
# ---------------------------------------------------------
# Target CPU for inference [cite: 484, 906]
inference_device = torch.device("cpu")
st.sidebar.text(f"Processing Unit: {inference_device}")

# Load ResNet18 with default pre-trained parameters [cite: 485, 907]
resnet_weights = models.ResNet18_Weights.DEFAULT
classifier_model = models.resnet18(weights=resnet_weights)
classifier_model.eval()  # Set to evaluation mode [cite: 485, 907]
classifier_model.to(inference_device)

# ---------------------------------------------------------
# Step 4: Load a pre-trained ResNet18 model from torchvision.models and set the model to evaluation mode.
# Step 5: Apply the recommended image preprocessing transformations associated with the ResNet18 pre- trained weights.
# ---------------------------------------------------------
# Retrieve the built-in transformation logic for this model [cite: 486, 908]
img_transform = resnet_weights.transforms()
label_map = resnet_weights.meta["categories"]

# ---------------------------------------------------------
# Step 6: Design a user interface that allows users to upload an image file (e.g., JPG or PNG).
# ---------------------------------------------------------
user_image = st.file_uploader("Select a JPG or PNG image for analysis", type=["jpg", "png", "jpeg"])

if user_image:
    # Load and show the input image [cite: 488, 910]
    raw_img = Image.open(user_image).convert("RGB")
    st.image(raw_img, caption="User Input", width=400)

    # ---------------------------------------------------------
    # Step 7: Convert the uploaded image into a tensor and perform model inference using PyTorch without gradient computation.
    # ---------------------------------------------------------
    with st.spinner("Analyzing visual features..."):
        # Convert to tensor and add batch dimension [cite: 488, 910]
        input_data = img_transform(raw_img).unsqueeze(0).to(inference_device)

        # Disable gradient tracking for efficiency [cite: 488, 910]
        with torch.no_grad():
            raw_scores = classifier_model(input_data)

    # ---------------------------------------------------------
    # Step 8: Apply the softmax function to the model output and display the top-5 predicted classes along with their probabilities.
    # ---------------------------------------------------------
    conf_scores = F.softmax(raw_scores[0], dim=0)
    top_values, top_indices = torch.topk(conf_scores, 5)

    st.divider()
    st.subheader("Top 5 Classification Results")

    # Organize data for display [cite: 489, 911]
    prediction_list = []
    for score, idx in zip(top_values, top_indices):
        prediction_list.append([label_map[idx], float(score)])

    results_df = pd.DataFrame(prediction_list, columns=["Category", "Confidence"])
    st.dataframe(results_df, use_container_width=True)

    # ---------------------------------------------------------
    # Step 9: Visualize the prediction probabilities using a bar chart in Streamlit.
    # ---------------------------------------------------------
    st.subheader("Confidence Distribution")
    st.bar_chart(results_df.set_index("Category"))

    # ---------------------------------------------------------
    # Step 10: Run the Streamlit application and test the system using multiple images. Discuss the classification results obtained. Discuss the level and the process path clearly.
    # ---------------------------------------------------------
    st.subheader("Automated Findings")

    primary_label = label_map[top_indices[0]]
    primary_score = float(top_values[0])

    # Reworded analytical logic [cite: 860, 861]
    if primary_score >= 0.85:
        summary = f"The algorithm displays **high certainty** in identifying this as a **'{primary_label}'** ({primary_score:.1%})."
    elif 0.60 <= primary_score < 0.85:
        summary = f"The algorithm shows **moderate confidence** that the subject is a **'{primary_label}'** ({primary_score:.1%})."
    else:
        summary = f"The prediction is **uncertain**, with **'{primary_label}'** being the most likely candidate despite low scoring ({primary_score:.1%})."

    st.write(summary)
    st.caption("Higher bar values in the chart indicate stronger activation in the final softmax layer.")
else:
    st.warning("Please upload a file to begin the classification process.")