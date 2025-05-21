import streamlit as st
import os
import json
import pandas as pd

# Helper function to extract f1 scores (same as before)
@st.cache_data
def extract_f1_scores(base_path):
    data = []
    for root, _, files in os.walk(base_path):
        if "metrics.json" in files:
            print("Found metrics.json in", root)
            model_name = os.path.basename(root)
            # Get the category path from the folder structure
            folder_parts = os.path.relpath(root, base_path).split(os.sep)
            category = '/'.join(folder_parts[:-1])  # Combine parent folders as category
            
            file_path = os.path.join(root, "metrics.json")
            with open(file_path, "r") as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {file_path}")
                    continue
                f1_score = metrics.get("macro avg", {}).get("f1-score")
                if f1_score is not None:
                    data.append({"Category": category, "Model": model_name, "F1 Score": f1_score})
    return pd.DataFrame(data)

# Helper function to load responses.json
@st.cache_data
def load_responses_json(model_path):
    responses_path = os.path.join(model_path, "responses_0.json")
    print("Loading responses from", responses_path)
    if os.path.exists(responses_path):
        with open(responses_path, "r") as f:
            try:
                responses = json.load(f)
                return responses
            except json.JSONDecodeError:
                print(f"Error reading {responses_path}")
    return None

# Cache the hierarchy creation function
@st.cache_data
def get_hierarchy(df):
    df['Hierarchy'] = df['Category'].apply(lambda x: x.split('/'))
    return df

# Streamlit app setup for the responses page
st.title("View Model Responses")

# Add option to change the results folder
results_path = st.text_input("Enter the path to the results folder", value="../results/curated")

# Check if the folder exists
if not os.path.exists(results_path):
    st.error(f"The folder '{results_path}' does not exist. Please enter a valid path.")
    st.stop()

# Extract and cache the data
df = extract_f1_scores(results_path)

# If no data was found
if df.empty:
    st.write("No data found in the results folder.")
else:
    # Add hierarchy to the DataFrame
    df = get_hierarchy(df)

    # Function to render the hierarchical structure for responses
    def render_hierarchy(df, category_path, level=0):
        filtered_df = df[df['Hierarchy'].apply(lambda x: len(x) > level and x[:level + 1] == category_path)]
        
        if filtered_df.empty:
            return
            
        next_level_categories = filtered_df['Hierarchy'].apply(lambda x: x[level + 1] if len(x) > level + 1 else None).dropna().unique()
        section_title = category_path[level].replace('_', ' ').capitalize()

        # Check if this is the last level in the hierarchy, account for when the hierarchy is shorter than 2 levels
        is_last_level = all(len(x) <= level + 1 for x in filtered_df['Hierarchy']) 

        if level == 1:
            # Stable expander key based on full category path
            expander_key = f"expander_{'_'.join(category_path)}"
            
            # Initialize state only once
            if expander_key not in st.session_state:
                st.session_state[expander_key] = False
                
            # Create expander with stable state
            with st.expander(section_title, expanded=st.session_state[expander_key]):
                _render_content(filtered_df, category_path, level, is_last_level, next_level_categories)
        else:
            st.markdown(f"{'#' * level} {section_title}")
            _render_content(filtered_df, category_path, level, is_last_level, next_level_categories)

    # Helper function to render the content (responses.json)
    def _render_content(filtered_df, category_path, level, is_last_level, next_level_categories):
        # If this is the last level, stop recursion and show the responses
        if is_last_level:
            leaf_nodes_df = filtered_df[filtered_df['Hierarchy'].apply(lambda x: len(x) == level + 1)]
            
            # Retrieve models for this category
            models = leaf_nodes_df['Model'].unique()
            
            # Model selection for this level
            model_key = f"models_{'_'.join(category_path)}"
            if model_key not in st.session_state:
                st.session_state[model_key] = models.tolist()

            show_models = st.multiselect("Select models to display", options=models, key=model_key, default=st.session_state[model_key])

            # Filter data for selected models
            filtered_leaf_nodes_df = leaf_nodes_df[leaf_nodes_df['Model'].isin(show_models)]

            # Render the responses for the selected models
            for model in show_models:
                model_row = filtered_leaf_nodes_df[filtered_leaf_nodes_df["Model"] == model]
                model_path = os.path.join(results_path, model_row.iloc[0]["Category"], model)
                responses = load_responses_json(model_path)

                if responses:
                    st.subheader(f"Responses for {model}")
                    st.dataframe(responses)  # Display the JSON responses in a formatted way
                else:
                    st.write(f"No responses found for model '{model}'.")
        else:
            # Recurse through the next level categories
            for next_category in next_level_categories:
                render_hierarchy(df, category_path + [next_category], level + 1)

    # Get top-level categories (first level in the hierarchy)
    top_categories = df['Hierarchy'].apply(lambda x: x[0]).unique()

    # Category selection dropdown
    selected_category = st.multiselect(
        "Select category to display",
        options=top_categories,
        key="category_selector",
        default=top_categories,
    )

    # Start rendering the hierarchical structure for selected category
    for category in selected_category:
        render_hierarchy(df, [category], level=0)
